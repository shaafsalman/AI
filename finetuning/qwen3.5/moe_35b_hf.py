"""
Qwen3.5-35B-A3B MoE LoRA Fine-Tune  (Standard HuggingFace / TRL stack)
=======================================================================
Hardware  : 8 × A100 80GB  (DDP via torchrun)
Run with  : torchrun --nproc_per_node=8 --master_addr=localhost --master_port=29500 moe_35b_hf.py

Model notes:
- Qwen3.5-35B-A3B is a Mixture-of-Experts model: 35B total params, ~3B active per token
- AutoModelForImageTextToText must be used (not AutoModelForCausalLM) so that vision
  encoder weights are loaded correctly; otherwise vLLM will report 400+ missing keys
- LoRA target modules use fully-qualified paths under the VLM wrapper
  (language_model.layers.*.self_attn.*) so PEFT matches the correct parameter names
- Full processor (not just tokenizer) is saved alongside the adapter to ensure
  preprocessor_config.json and video_preprocessor_config.json are present for vLLM
"""

import os
import glob
import json
import shutil
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig
from huggingface_hub import HfApi

# ── config ─────────────────────────────────────────────────────────────────────
# Override with environment variables, or edit the defaults.
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B")
DATASET_NAME = os.environ.get("DATASET_NAME", "")   # required -- HF dataset id
HF_REPO      = os.environ.get("HF_REPO", "")        # optional -- push skipped if empty
HF_TOKEN     = os.environ.get("HF_TOKEN", "")       # never hardcode a token here

OUTPUT_DIR   = "qwen35_moe_checkpoints"
LORA_DIR     = OUTPUT_DIR + "_lora"
MERGED_DIR   = OUTPUT_DIR + "_merged"

# Sequence & LoRA hyperparameters
MAX_SEQ_LEN   = 2048
LORA_RANK     = 16
LORA_ALPHA    = 32

# Training hyperparameters (tuned for 8 × A100 80GB)
BATCH_SIZE    = 1     # per-device batch size; MoE is memory-intensive
GRAD_ACCUM    = 4     # effective batch = 1 × 4 × 8 GPUs = 32
EPOCHS        = 3
LR            = 2e-4
WARMUP_STEPS  = 100
SEED          = 3407

if not DATASET_NAME:
    raise SystemExit("Set DATASET_NAME (env var or edit the config block above).")

# ── prompt template ────────────────────────────────────────────────────────────
# Qwen3.5 uses the ChatML format.  Instruction/input/output map to
# system / user / assistant roles respectively.
PROMPT_TEMPLATE = (
    "<|im_start|>system\n{instruction}<|im_end|>\n"
    "<|im_start|>user\n{input}<|im_end|>\n"
    "<|im_start|>assistant\n{output}<|im_end|>"
)


def format_dataset(examples):
    """Convert Alpaca-style batch rows into ChatML-formatted text strings."""
    texts = []
    for instruction, inp, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        texts.append(PROMPT_TEMPLATE.format(
            instruction=instruction.strip(),
            input=inp.strip(),
            output=output.strip(),
        ))
    return {"text": texts}


def train():
    # Load dataset and convert to flat text strings
    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.map(
        format_dataset,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=8,     # parallelise preprocessing across 8 CPU workers
    )

    # Use AutoProcessor (not AutoTokenizer) so all processor files are saved
    # alongside the adapter — prevents missing preprocessor_config.json and
    # video_preprocessor_config.json during merge/push
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use AutoModelForImageTextToText (not AutoModelForCausalLM) so the full
    # VLM is loaded including vision encoder weights
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",   # scaled dot-product attention (torch >= 2.0)
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    # LoRA adapter — target attention projections only via fully-qualified paths
    # that match the VLM wrapper's parameter namespace
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "language_model.layers.*.self_attn.q_proj",
            "language_model.layers.*.self_attn.k_proj",
            "language_model.layers.*.self_attn.v_proj",
            "language_model.layers.*.self_attn.o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Resume from the latest checkpoint if one exists
    ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
    resume = ckpts[-1] if ckpts else False

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,                                    # bfloat16 training (A100 native)
        max_grad_norm=1.0,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        optim="adamw_torch_fused",
        seed=SEED,
        dataset_num_proc=8,
        packing=True,                                 # pack short sequences → fewer padding tokens
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    trainer.train(resume_from_checkpoint=resume)

    # Only rank-0 saves the adapter after all processes finish training
    if trainer.is_world_process_zero():
        print("Saving LoRA adapter...")
        model.save_pretrained(LORA_DIR)
        # Save full processor so all config files are present for vLLM
        processor.save_pretrained(LORA_DIR)
        print(f"Adapter saved to {LORA_DIR}")


def merge_and_push():
    """Merge LoRA weights into the base model and push to HuggingFace Hub."""
    print("Loading base model for merge...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, LORA_DIR)
    model = model.merge_and_unload(progressbar=True)

    print(f"Saving merged model to {MERGED_DIR}...")
    model.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="4GB")

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    processor.save_pretrained(MERGED_DIR)

    # Patch tokenizer_class — PEFT adapter save writes "TokenizersBackend"
    # which vLLM rejects; set it to the expected Qwen2TokenizerFast class
    tok_cfg_path = os.path.join(MERGED_DIR, "tokenizer_config.json")
    with open(tok_cfg_path) as f:
        tok_cfg = json.load(f)
    tok_cfg["tokenizer_class"] = "Qwen2TokenizerFast"
    with open(tok_cfg_path, "w") as f:
        json.dump(tok_cfg, f, indent=2)
    print("Patched tokenizer_class -> Qwen2TokenizerFast")

    # Ensure config.json reflects the VLM architecture so vLLM can resolve
    # the correct model class (Qwen3_5MoeForConditionalGeneration)
    cfg_path = os.path.join(MERGED_DIR, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["model_type"] = "qwen3_5_moe"
    cfg["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("Patched config.json architecture")

    if not (HF_TOKEN and HF_REPO):
        print("HF_TOKEN or HF_REPO not set -- skipping push.")
        return

    print(f"Pushing to {HF_REPO}...")
    # Use HfApi.upload_folder to avoid deprecated safe_serialization kwarg
    # issues that arise with model.push_to_hub on large sharded models
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=MERGED_DIR,
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="Add merged Qwen3.5-35B-A3B fine-tuned model",
    )
    print(f"Done -- https://huggingface.co/{HF_REPO}")


def main():
    train()

    # Only rank-0 runs the merge+push step
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        merge_and_push()


if __name__ == "__main__":
    main()

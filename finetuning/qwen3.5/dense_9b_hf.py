"""
Qwen3.5-9B LoRA Fine-Tune  (Standard HuggingFace / TRL stack, DDP)
====================================================================
Hardware  : 8 × A100-SXM4-40GB  (or 80GB)
Run with  : torchrun --nproc_per_node=8 --master_addr=localhost --master_port=29500 dense_9b_hf.py

Design decisions:
- Standard HuggingFace transformers + PEFT + TRL (no Unsloth dependency)
- BF16 training with torch sdpa attention and tf32 matmul for extra speed
- adamw_8bit optimiser cuts VRAM vs standard AdamW
- Sequence packing (packing=True) maximises GPU utilisation on long contexts
- vLLM config/tokenizer patches applied post-merge so the served model loads
  cleanly as a text-only dense model (strips multimodal and mrope fields)
"""

import os
import glob
import json
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig
from huggingface_hub import HfApi

# Enable TF32 for matrix multiplications — speeds up BF16 training on A100
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Per-rank GPU identity set by torchrun
local_rank = int(os.environ.get("LOCAL_RANK", 0))

# ── config ─────────────────────────────────────────────────────────────────────
# Override with environment variables, or edit the defaults.
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-9B")
DATASET_NAME = os.environ.get("DATASET_NAME", "")   # required -- HF dataset id
HF_REPO      = os.environ.get("HF_REPO", "")        # optional -- push skipped if empty
HF_TOKEN     = os.environ.get("HF_TOKEN", "")       # never hardcode a token here

OUTPUT_DIR = "qwen35_9b_checkpoints"
LORA_DIR   = OUTPUT_DIR + "_lora"
MERGED_DIR = OUTPUT_DIR + "_merged"

# Sequence & LoRA hyperparameters
MAX_SEQ_LEN      = 2048
LORA_RANK        = 16
LORA_ALPHA       = 32

# Training hyperparameters (tuned for 8 × A100)
BATCH_SIZE       = 2     # per-device; effective = 2 × 4 accum × 8 GPUs = 64
GRAD_ACCUM       = 4
EPOCHS           = 1
LR               = 2e-4
WARMUP_STEPS     = 5
SEED             = 3407
DATASET_NUM_PROC = 8     # CPU workers for dataset preprocessing

if not DATASET_NAME:
    raise SystemExit("Set DATASET_NAME (env var or edit the config block above).")

# ── prompt template ────────────────────────────────────────────────────────────
# Qwen3.5 uses ChatML format.  Instruction/input/output map to
# system / user / assistant roles respectively.
PROMPT_TEMPLATE = (
    "<|im_start|>system\n{instruction}<|im_end|>\n"
    "<|im_start|>user\n{input}<|im_end|>\n"
    "<|im_start|>assistant\n{output}<|im_end|>"
)


# ── helpers ────────────────────────────────────────────────────────────────────

def format_dataset(examples):
    """Convert Alpaca-style batch rows into ChatML-formatted text strings."""
    texts = []
    for instruction, inp, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        texts.append(
            PROMPT_TEMPLATE.format(
                instruction=instruction.strip(),
                input=inp.strip(),
                output=output.strip(),
            )
        )
    return {"text": texts}


def patch_config_for_vllm(merged_dir):
    """
    Fix config.json so vLLM serves the merged model correctly:
      1. Force text-only architecture (Qwen3_5ForCausalLM)
      2. Force correct model_type (qwen3_5)
      3. Strip mrope_* keys that cause rotary-embedding shape crashes under TP
      4. Remove leftover vision/audio/media keys from the base model checkpoint
    """
    config_path = os.path.join(merged_dir, "config.json")
    import shutil
    shutil.copy2(config_path, config_path + ".bak")   # keep a backup

    with open(config_path) as f:
        config = json.load(f)

    config["architectures"] = ["Qwen3_5ForCausalLM"]
    config["model_type"]    = "qwen3_5"

    # mrope fields cause rotary-embedding shape mismatches under tensor parallelism
    MROPE_KEYS = {"mrope_section", "mrope_interleaved"}
    for rope_field in ("rope_scaling", "rope_parameters"):
        if rope_field in config and isinstance(config[rope_field], dict):
            for k in MROPE_KEYS:
                config[rope_field].pop(k, None)
    for k in MROPE_KEYS:
        config.pop(k, None)

    # Remove multimodal keys — not needed for text-only vLLM serving
    for key in ("vision_config", "audio_config", "media_config",
                "image_token_id", "video_token_id", "audio_token_id"):
        config.pop(key, None)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[patch_config] arch={config['architectures']}  model_type={config['model_type']}")


def patch_tokenizer_for_vllm(merged_dir):
    """
    Remove processor_class (multimodal) and set a plain tokenizer class
    so vLLM doesn't attempt to load an image processor that doesn't exist.
    """
    tok_config_path = os.path.join(merged_dir, "tokenizer_config.json")
    with open(tok_config_path) as f:
        tok_config = json.load(f)

    tok_config.pop("processor_class", None)
    tok_config["tokenizer_class"] = "PreTrainedTokenizerFast"

    with open(tok_config_path, "w") as f:
        json.dump(tok_config, f, indent=2)

    print("[patch_tokenizer] processor_class removed, tokenizer_class=PreTrainedTokenizerFast")


def rebuild_safetensors_index(merged_dir):
    """
    Rebuild model.safetensors.index.json from the actual shard files on disk.
    Required when the index is stale or was overwritten by a base-model download.
    """
    from safetensors import safe_open

    shards = sorted([
        f for f in os.listdir(merged_dir)
        if f.endswith(".safetensors")
    ])
    if not shards:
        raise RuntimeError(f"No .safetensors files found in {merged_dir}")

    weight_map = {}
    total_size = 0

    for shard in shards:
        path        = os.path.join(merged_dir, shard)
        total_size += os.path.getsize(path)
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                weight_map[key] = shard

    index = {
        "metadata":   {"total_size": total_size},
        "weight_map": weight_map,
    }

    out = os.path.join(merged_dir, "model.safetensors.index.json")
    with open(out, "w") as f:
        json.dump(index, f, indent=2)

    print(f"[rebuild_index] {len(weight_map)} tensors across {len(shards)} shards → {out}")


# ── train ──────────────────────────────────────────────────────────────────────

def train():
    if local_rank == 0:
        print(f"Loading dataset: {DATASET_NAME}")

    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.map(
        format_dataset,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=DATASET_NUM_PROC,
    )

    if local_rank == 0:
        print("Dataset size:", len(dataset))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",   # scaled dot-product attention (torch >= 2.0)
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    # LoRA adapter — targets attention + FFN projections
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    if local_rank == 0:
        model.print_trainable_parameters()

    # Resume from the latest checkpoint if one exists
    ckpts  = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
    resume = ckpts[-1] if ckpts else None
    if local_rank == 0:
        print("Resuming from:", resume)

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
        bf16=True,
        tf32=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        optim="adamw_8bit",
        seed=SEED,
        dataset_num_proc=DATASET_NUM_PROC,
        packing=True,
        dataloader_num_workers=8,
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

    # Synchronise all ranks before saving
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Only rank-0 saves to avoid concurrent writes
    if trainer.is_world_process_zero():
        print("Saving LoRA adapter...")
        model.save_pretrained(LORA_DIR)
        tokenizer.save_pretrained(LORA_DIR)
        print("Saved adapter:", LORA_DIR)


# ── merge & push ───────────────────────────────────────────────────────────────

def merge_and_push():
    """Merge LoRA weights into the base model, patch for vLLM, and push to Hub."""
    print("Loading base model for merge...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Merging LoRA weights...")
    model = PeftModel.from_pretrained(model, LORA_DIR)
    model = model.merge_and_unload(progressbar=True)

    print(f"Saving merged model to {MERGED_DIR}...")
    model.save_pretrained(
        MERGED_DIR,
        safe_serialization=True,
        max_shard_size="4GB",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.save_pretrained(MERGED_DIR)

    # Apply patches required for clean vLLM serving
    patch_config_for_vllm(MERGED_DIR)
    patch_tokenizer_for_vllm(MERGED_DIR)
    rebuild_safetensors_index(MERGED_DIR)

    print("\n" + "=" * 60)
    print("Merge complete. Serve with:")
    print(f"  vllm serve {MERGED_DIR} \\")
    print(f"    --port 8000 \\")
    print(f"    --tensor-parallel-size 8 \\")
    print(f"    --gpu-memory-utilization 0.85 \\")
    print(f"    --max-model-len 32768 \\")
    print(f"    --language-model-only")
    print("=" * 60 + "\n")

    if not (HF_TOKEN and HF_REPO):
        print("HF_TOKEN or HF_REPO not set -- skipping push.")
        return

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=MERGED_DIR,
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="Add merged Qwen3.5-9B fine-tuned model (vLLM-ready)",
        ignore_patterns=["*.bak"],
    )
    print(f"Pushed to HuggingFace: https://huggingface.co/{HF_REPO}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    train()
    if local_rank == 0:
        merge_and_push()


if __name__ == "__main__":
    main()

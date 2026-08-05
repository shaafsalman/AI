"""
Qwen3.5-4B LoRA Fine-Tune  (Unsloth + TRL, multi-GPU)
======================================================
Hardware  : 8 × A100 80GB
Run with  : torchrun --nproc_per_node=8 dense_4b_unsloth.py

Notes:
- Uses Unsloth FastLanguageModel for optimised BF16 LoRA
- PartialState ensures only rank-0 downloads/processes the dataset;
  all other ranks wait, preventing 8 simultaneous HuggingFace Hub requests
- Sequence packing (packing=True) fills each context window fully,
  significantly increasing throughput on A100s
- NEFTune noise (neftune_noise_alpha=5) provides mild regularisation
  which helps instruction-following generalisation
- adamw_8bit saves ~2 GB VRAM per GPU versus standard AdamW
"""

import unsloth  # MUST be imported first so Unsloth patches are applied early

import os
import shutil

import torch
from accelerate import PartialState
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

# ── config ─────────────────────────────────────────────────────────────────────
# Override with environment variables, or edit the defaults.
MODEL_NAME   = os.environ.get("MODEL_NAME", "unsloth/Qwen3.5-4B")
DATASET_NAME = os.environ.get("DATASET_NAME", "")   # required -- HF dataset id
HF_REPO      = os.environ.get("HF_REPO", "")        # optional -- push skipped if empty

# Sequence settings
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT   = False
LOAD_IN_16BIT  = True   # BF16 full-precision LoRA; fits on A100 80GB for 4B

# LoRA hyperparameters
LORA_R         = 32
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.0    # 0 recommended by Unsloth for best throughput

# Training hyperparameters (tuned for 8 × A100 80GB)
PER_DEVICE_BATCH = 16   # with 80 GB per GPU, 4B model leaves headroom for large batches
GRAD_ACCUM       = 1    # effective batch = 16 × 1 × 8 GPUs = 128
MAX_STEPS        = -1   # -1 = train for NUM_EPOCHS full passes
NUM_EPOCHS       = 1
LR               = 2e-4
WARMUP_STEPS     = 100
LR_SCHEDULER     = "cosine"
WEIGHT_DECAY     = 0.01
SEED             = 3407

OUTPUT_DIR  = "./outputs/qwen35-4b"
MERGED_DIR  = "./outputs/qwen35-4b-merged"

if not DATASET_NAME:
    raise SystemExit("Set DATASET_NAME (env var or edit the config block above).")

# ── 1. load model + tokenizer ──────────────────────────────────────────────────
print("Loading model …")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = MODEL_NAME,
    max_seq_length= MAX_SEQ_LENGTH,
    load_in_4bit  = LOAD_IN_4BIT,
    load_in_16bit = LOAD_IN_16BIT,
    full_finetuning = False,          # LoRA; set True for full fine-tune (more VRAM)
)

# ── 2. LoRA adapters ───────────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r                          = LORA_R,
    target_modules             = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",        # FFN
    ],
    lora_alpha                 = LORA_ALPHA,
    lora_dropout               = LORA_DROPOUT,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",   # ~30% less VRAM vs standard GC
    random_state               = SEED,
    use_rslora                 = False,
    loftq_config               = None,
)

# ── 3. load & format dataset ───────────────────────────────────────────────────
state = PartialState()

with state.main_process_first():
    print("Loading dataset …")
    dataset = load_dataset(DATASET_NAME, split="train")
    print("Dataset columns:", dataset.column_names)

    def format_example(example):
        """
        Convert Alpaca-style row into a chat-template string.
        Expected columns: instruction, input (optional), output.
        The output should already contain any reasoning tokens (e.g. <think>...</think>).
        """
        user_content = example["instruction"]
        if example.get("input", "").strip():
            user_content += "\n\n" + example["input"]
        assistant_content = example["output"]
        messages = [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize              = False,
            add_generation_prompt = False,
        )
        return {"text": text}

    dataset = dataset.map(
        format_example,
        remove_columns = dataset.column_names,
        num_proc       = 8,
    )

# ── 4. trainer ─────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model          = model,
    tokenizer      = tokenizer,
    train_dataset  = dataset,
    max_seq_length = MAX_SEQ_LENGTH,
    args = SFTConfig(
        dataset_text_field          = "text",
        packing                     = True,
        per_device_train_batch_size = PER_DEVICE_BATCH,
        gradient_accumulation_steps = GRAD_ACCUM,
        num_train_epochs            = NUM_EPOCHS,
        max_steps                   = MAX_STEPS,
        learning_rate               = LR,
        warmup_steps                = WARMUP_STEPS,
        lr_scheduler_type           = LR_SCHEDULER,
        weight_decay                = WEIGHT_DECAY,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        logging_steps               = 10,
        save_steps                  = 200,
        save_total_limit            = 3,
        output_dir                  = OUTPUT_DIR,
        seed                        = SEED,
        report_to                   = "none",
        dataset_num_proc            = 8,
        ddp_find_unused_parameters  = False,
        dataloader_num_workers      = 4,
        dataloader_pin_memory       = True,
        neftune_noise_alpha         = 5,
        optim                       = "adamw_8bit",
    ),
)

# ── 5. train ───────────────────────────────────────────────────────────────────
print("Starting training …")
trainer_stats = trainer.train()
print(f"Training complete. Steps: {trainer_stats.global_step}")
print(f"Peak VRAM per GPU: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

# Everything below writes to disk or to the Hub, so only rank 0 runs it.
IS_MAIN = int(os.environ.get("LOCAL_RANK", 0)) == 0

# ── 6. save LoRA adapter ───────────────────────────────────────────────────────
if IS_MAIN:
    adapter_dir = os.path.join(OUTPUT_DIR, "final-adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved -> {adapter_dir}")

# ── 7. merge LoRA into base model for vLLM serving ────────────────────────────
if IS_MAIN:
    print("Merging LoRA into base model for vLLM ...")
    model.save_pretrained_merged(
        MERGED_DIR,
        tokenizer,
        save_method = "merged_16bit",
    )
    print(f"Merged model saved -> {MERGED_DIR}")

# ── 8. push to HuggingFace Hub ─────────────────────────────────────────────────
if IS_MAIN and HF_REPO:
    print(f"Pushing to HF Hub: {HF_REPO} ...")
    model.push_to_hub_merged(
        HF_REPO,
        tokenizer,
        save_method = "merged_16bit",
    )
    print(f"Done -> https://huggingface.co/{HF_REPO}")

    # ── 9. fix tokenizer: replace with base model tokenizer files ─────────────
    # Unsloth writes a custom TokenizersBackend tokenizer that vLLM can't load.
    # Fix: overwrite with the base model's standard tokenizer files.
    print("Fixing tokenizer: replacing with base model tokenizer files ...")
    base_tokenizer_dir = os.path.join(OUTPUT_DIR, "base_tokenizer_tmp")
    snapshot_download(
        MODEL_NAME,
        local_dir       = base_tokenizer_dir,
        ignore_patterns = ["*.safetensors", "*.bin", "*.pt"],
    )

    api = HfApi()
    api.upload_folder(
        folder_path    = base_tokenizer_dir,
        repo_id        = HF_REPO,
        repo_type      = "model",
        commit_message = "Fix tokenizer: replace with base model tokenizer files",
    )

    shutil.rmtree(base_tokenizer_dir)
    print(f"Tokenizer fixed -> https://huggingface.co/{HF_REPO}")

elif IS_MAIN:
    print("HF_REPO not set -- skipping push.")

# ── 10. vLLM serve command ─────────────────────────────────────────────────────
if IS_MAIN:
    print(f"""
To serve with vLLM:
  vllm serve {HF_REPO or MERGED_DIR} \\
      --tensor-parallel-size 8 \\
      --max-model-len 32768 \\
      --reasoning-parser qwen3 \\
      --enable-prefix-caching
""")

"""
Qwen3.5-27B Dense LoRA Fine-Tune  (Unsloth + TRL, DDP)
========================================================
Hardware  : 8 × A100-SXM4-80GB
Run with  : UNSLOTH_COMPILE_TRANSFORMERS=0 torchrun --nproc_per_node 8 dense_27b_unsloth.py

Key design decisions:
- FastLanguageModel (text-only, not vision) for maximum throughput on dense models
- BF16 full-precision LoRA (no 4-bit quantisation; 27B fits on 80GB A100 in BF16)
- response_part includes "<think>" so reasoning/CoT tokens are trained
- Unsloth gradient checkpointing ("unsloth" mode) reduces VRAM ~30% on long sequences
- DDP: proper barrier placement, dist.is_initialized() check before barrier call
- Save: only rank-0 writes checkpoints and uploads to HuggingFace Hub
"""

import os, sys, glob, shutil
import torch
import torch.distributed as dist

# ── per-rank GPU assignment ────────────────────────────────────────────────────
# torchrun sets LOCAL_RANK; each process sees only its own GPU
local_rank = int(os.environ.get("LOCAL_RANK", 0))
os.environ["CUDA_VISIBLE_DEVICES"]         = str(local_rank)
os.environ["PYTORCH_CUDA_ALLOC_CONF"]      = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"]          = "1"   # avoid torch.compile conflicts with Unsloth
os.environ["UNSLOTH_MOE_BACKEND"]          = "triton"
os.environ["UNSLOTH_DISABLE_AUTOTUNE"]     = "1"
os.environ["TRITON_CACHE_DIR"]             = f"/tmp/triton_cache_{local_rank}"   # per-rank cache
os.environ["UNSLOTH_COMPILE_TRANSFORMERS"] = "0"

# ── config ─────────────────────────────────────────────────────────────────────
# Override with environment variables, or edit the defaults.
MODEL_NAME       = os.environ.get("MODEL_NAME", "unsloth/Qwen3.5-27B")  # Unsloth mirror downloads faster
DATASET_NAME     = os.environ.get("DATASET_NAME", "")   # required -- HF dataset id
DATASET_SPLIT    = "train"
MAX_SEQ_LENGTH   = 2048
LORA_RANK        = 16
LORA_ALPHA       = 16

# Training hyperparameters (tuned for 8 × A100 80GB)
NUM_TRAIN_EPOCHS = 1
BATCH_SIZE       = 2     # per-device; effective = 2 × 4 accum × 8 GPUs = 64
GRAD_ACCUM       = 4
WARMUP_STEPS     = 5
LEARNING_RATE    = 2e-4
SEED             = 3407
LOGGING_STEPS    = 10
DATASET_NUM_PROC = 4     # CPU workers for dataset preprocessing

OUTPUT_DIR  = "outputs_qwen35_27b"
MERGED_DIR  = "qwen35_27b_merged"
HF_REPO     = os.environ.get("HF_REPO", "")     # optional -- push skipped if empty
HF_TOKEN    = os.environ.get("HF_TOKEN", "")    # never hardcode a token here

if not DATASET_NAME:
    raise SystemExit("Set DATASET_NAME (env var or edit the config block above).")

# ── clear stale compiled cache (rank-0 only, before dist init) ────────────────
# Stale Unsloth compiled cache can cause dtype mismatch errors on restart
if local_rank == 0:
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unsloth_compiled_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleared stale compiled cache: {cache_dir}")

# ── unsloth imports ────────────────────────────────────────────────────────────
# Must be imported after env vars are set so patches apply correctly
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

if local_rank == 0:
    print(f"GPU {local_rank}: {torch.cuda.get_device_name(0)}")

# ── load model ─────────────────────────────────────────────────────────────────
# BF16 full precision (load_in_4bit=False, load_in_8bit=False)
# 27B in BF16 ≈ 54 GB; fits comfortably on a single A100 80GB
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit   = False,
    load_in_8bit   = False,
    full_finetuning= False,   # LoRA; set True to train all weights (much more VRAM)
)

# ── LoRA adapters ──────────────────────────────────────────────────────────────
# out_proj is also included for 27B; increases trainable params slightly
# use_gradient_checkpointing="unsloth" saves ~30% VRAM on long sequences
model = FastLanguageModel.get_peft_model(
    model,
    r                       = LORA_RANK,
    lora_alpha              = LORA_ALPHA,
    lora_dropout            = 0,
    bias                    = "none",
    target_modules          = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",        # FFN
        "out_proj",
    ],
    use_gradient_checkpointing = "unsloth",
    random_state            = SEED,
    use_rslora              = False,
    loftq_config            = None,
)
model.config.use_cache = False

# ── dataset formatting ─────────────────────────────────────────────────────────
# Qwen3.5 uses ChatML format: <|im_start|>role\ncontent<|im_end|>
# Including <think>...</think> in the output teaches chain-of-thought reasoning
PROMPT_TEMPLATE = """\
<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""

def format_dataset(examples):
    """Convert Alpaca-style batch rows into ChatML-formatted text strings."""
    texts = []
    for instruction, inp, output in zip(
        examples["instruction"],
        examples["input"],
        examples["output"],
    ):
        texts.append(PROMPT_TEMPLATE.format(
            instruction = instruction.strip(),
            input       = inp.strip(),
            output      = output.strip(),
        ))
    return {"text": texts}

if local_rank == 0:
    print(f"Loading dataset: {DATASET_NAME} ...")

dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
dataset = dataset.map(
    format_dataset,
    batched        = True,
    remove_columns = dataset.column_names,
    num_proc       = DATASET_NUM_PROC,
)

if local_rank == 0:
    print(f"Dataset size: {len(dataset):,} examples")

# ── trainer ────────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field          = "text",
        max_seq_length              = MAX_SEQ_LENGTH,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        num_train_epochs            = NUM_TRAIN_EPOCHS,
        warmup_steps                = WARMUP_STEPS,
        learning_rate               = LEARNING_RATE,
        lr_scheduler_type           = "cosine",
        weight_decay                = 0.01,
        bf16                        = True,
        logging_steps               = LOGGING_STEPS,
        save_steps                  = 200,
        save_total_limit            = 3,
        output_dir                  = OUTPUT_DIR,
        optim                       = "adamw_8bit",
        seed                        = SEED,
        dataset_num_proc            = DATASET_NUM_PROC,
        packing                     = True,    # fill sequences to MAX_SEQ_LENGTH
        dataloader_num_workers      = 4,
        dataloader_pin_memory       = True,
        report_to                   = "none",
        ddp_find_unused_parameters  = False,
    ),
)

# ── train on responses only ────────────────────────────────────────────────────
# Loss is computed only on assistant tokens (not on user/system tokens).
# Including "<think>" in response_part means CoT reasoning tokens are trained.
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n<think>",
)

# ── resume from checkpoint if one exists ──────────────────────────────────────
ckpts  = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
resume = ckpts[-1] if ckpts else False
if local_rank == 0:
    print(f"Resuming from: {resume}" if resume else "Starting fresh")

# ── train ──────────────────────────────────────────────────────────────────────
trainer_stats = trainer.train(resume_from_checkpoint=resume)

# ── save (rank-0 only) ─────────────────────────────────────────────────────────
# Barrier ensures all ranks have finished before rank-0 writes checkpoints
if dist.is_available() and dist.is_initialized():
    dist.barrier()

if local_rank == 0:
    print("Training complete. Saving LoRA adapters...")
    model.save_pretrained(OUTPUT_DIR + "_lora")
    tokenizer.save_pretrained(OUTPUT_DIR + "_lora")

    # Merge LoRA into base weights as a standalone 16-bit model for vLLM serving
    print("Merging to 16-bit...")
    model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
    print(f"Saved merged model to: {MERGED_DIR}")

    if HF_TOKEN and HF_REPO:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_REPO, exist_ok=True, private=False)
        print(f"Uploading to {HF_REPO} ...")
        api.upload_folder(
            folder_path = MERGED_DIR,
            repo_id     = HF_REPO,
            repo_type   = "model",
        )
        print("Upload complete!")
    else:
        print("HF_TOKEN or HF_REPO not set -- skipping upload.")

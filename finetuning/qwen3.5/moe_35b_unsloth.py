"""
Qwen3.5-35B-A3B MoE LoRA Fine-Tune  (Unsloth, single GPU)
===========================================================
Hardware  : 8 × A100 80GB  (use one GPU per run; set CUDA_VISIBLE_DEVICES accordingly)
            Alternatively: single H100 80GB

Requirements:
    pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo
    pip install git+https://github.com/huggingface/transformers.git --no-deps
    pip install --no-deps trl==0.22.2

Run with  : python moe_35b_unsloth.py

Model notes:
- Qwen3.5-35B-A3B is a Mixture-of-Experts model: 35B total params, ~3B active per token
- load_in_16bit=True (BF16 LoRA): 4-bit MoE QLoRA is not recommended per Unsloth docs
- train_on_responses_only is NOT used so that full-sequence loss is applied
- The <think>...</think> blocks in outputs teach chain-of-thought reasoning
- After training, the model is merged to 16-bit for direct vLLM serving

Design notes:
- ALL os.environ calls MUST precede any unsloth import so that env patches are applied
- UNSLOTH_COMPILE_DISABLE=1 avoids BF16/FP32 dtype mismatch inside MoE layers
- Sequences longer than MAX_SEQ_LENGTH are filtered out before training
- Memory is explicitly freed (del / gc.collect) before the merge step
"""

import os
import gc
import torch

# ── env vars must come before any unsloth import ─────────────────────────────
# Change CUDA_VISIBLE_DEVICES to the GPU index you want to use on a multi-GPU node
os.environ["CUDA_VISIBLE_DEVICES"]             = "0"
os.environ["UNSLOTH_COMPILE_DISABLE"]          = "1"   # fixes bf16/fp32 dtype mismatch in MoE
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"]  = "1"
os.environ["TOKENIZERS_PARALLELISM"]           = "false"

# ── imports ────────────────────────────────────────────────────────────────────
from datasets import load_dataset, Dataset
from unsloth import FastModel
from trl import SFTTrainer, SFTConfig

# ── config ─────────────────────────────────────────────────────────────────────
# Override with environment variables, or edit the defaults.
MODEL_NAME     = os.environ.get("MODEL_NAME", "unsloth/Qwen3.5-35B-A3B")
DATASET_NAME   = os.environ.get("DATASET_NAME", "")   # required -- HF dataset id
HF_REPO        = os.environ.get("HF_REPO", "")        # optional -- push skipped if empty
MAX_SEQ_LENGTH = 4096                                 # reduce to 2048 if OOM
LORA_RANK      = 16                                   # try 32 for higher adapter capacity

if not DATASET_NAME:
    raise SystemExit("Set DATASET_NAME (env var or edit the config block above).")

# ── 1. load model ──────────────────────────────────────────────────────────────
# FastModel handles both dense and MoE Qwen3.5 variants automatically.
# load_in_16bit=True: BF16 LoRA; fits on a single A100/H100 80GB for the 35B-A3B MoE.
model, processor = FastModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit   = False,    # 4-bit MoE QLoRA not recommended
    load_in_16bit  = True,     # BF16 LoRA
    full_finetuning= False,
)

# FastModel may return a multimodal processor — extract the text tokenizer
tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

# ── 2. LoRA adapters ───────────────────────────────────────────────────────────
# gate_up_proj covers the MoE fused gate projection (specific to the A3B MoE variant)
# alpha == r is the Unsloth recommendation for MoE models
model = FastModel.get_peft_model(
    model,
    r           = LORA_RANK,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention projections
        "gate_proj", "up_proj", "down_proj",        # dense FFN layers
        "gate_up_proj",                             # MoE fused projection
    ],
    lora_alpha                 = LORA_RANK,    # alpha == r per Unsloth recommendation
    lora_dropout               = 0,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",    # extends effective context; saves VRAM
    random_state               = 3407,
)

# ── 3. load & format dataset ───────────────────────────────────────────────────
# Expected dataset schema (Alpaca-style):
#   instruction : the question or task description
#   input       : optional extra context (may be empty string)
#   output      : answer — should contain <think>...</think> + final answer
#
# Chain-of-thought format (recommended for reasoning tasks):
#   <think>
#     step-by-step reasoning ...
#   </think>
#   Final concise answer.

print("Loading dataset …")
raw_dataset = load_dataset(DATASET_NAME, split="train")
print(f"Raw dataset size : {len(raw_dataset)} rows")
print(f"Columns          : {raw_dataset.column_names}")
print(f"Sample row:\n{raw_dataset[0]}")


def format_to_chat(example: dict) -> dict:
    """
    Convert Alpaca-style row to a Qwen3.5 chat-template string.

    Qwen3.5 uses ChatML format:
        <|im_start|>role\ncontent<|im_end|>

    The assistant turn should begin with <think> so the model learns to
    reason before producing its final answer.
    """
    instruction = example.get("instruction", "").strip()
    inp         = example.get("input", "").strip()
    output      = example.get("output", "").strip()

    # Combine instruction and optional context into a single user message
    user_content = f"{instruction}\n\n{inp}" if inp else instruction

    messages = [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": output},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize              = False,
        add_generation_prompt = False,
    )
    return {"text": text}


print("Formatting dataset …")
formatted = raw_dataset.map(format_to_chat, num_proc=1, desc="Formatting")

# Filter out sequences that exceed the model's max context window
print("Filtering by token length …")
formatted = formatted.map(
    lambda ex: {"length": len(tokenizer.encode(ex["text"]))},
    num_proc=1, desc="Counting tokens",
)
before    = len(formatted)
formatted = formatted.filter(lambda x: x["length"] <= MAX_SEQ_LENGTH)
print(f"Kept {len(formatted)}/{before} rows (≤ {MAX_SEQ_LENGTH} tokens)")

train_dataset = formatted.select_columns(["text"])
print(train_dataset)

# ── 4. train ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model            = model,
    processing_class = tokenizer,
    train_dataset    = train_dataset,
    args = SFTConfig(
        dataset_text_field          = "text",
        max_seq_length              = MAX_SEQ_LENGTH,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,    # effective batch = 4 steps × 1 sample = 4
        num_train_epochs            = 3,
        warmup_ratio                = 0.05,
        learning_rate               = 2e-4,
        logging_steps               = 10,
        save_steps                  = 100,
        save_total_limit            = 2,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "cosine",
        seed                        = 3407,
        dataset_num_proc            = 1,    # single process for single-GPU run
        dataloader_num_workers      = 0,
        bf16                        = True,
        output_dir                  = "checkpoints",
        report_to                   = "none",    # swap to "wandb" if desired
    ),
)

print("Starting training …")
trainer.train()
print("Training complete.")

# ── 5. free memory before merge ────────────────────────────────────────────────
# Releasing dataset objects frees CPU/GPU RAM before the merge step
del train_dataset, formatted, raw_dataset
torch.cuda.empty_cache()
gc.collect()

# ── 6. merge and push to HuggingFace Hub ──────────────────────────────────────
# Merging LoRA adapters into the base weights produces a standalone model
# that vLLM can serve directly without any adapter loading overhead.
# Auth comes from the HF_TOKEN env var or a prior `hf auth login`.
if HF_REPO:
    print(f"Pushing merged 16-bit model to: {HF_REPO} ...")
    model.push_to_hub_merged(HF_REPO, tokenizer, save_method="merged_16bit")
    print("Push complete!")
    print("\nTo serve with vLLM:")
    print(f"  vllm serve {HF_REPO} --dtype bfloat16 --max-model-len {MAX_SEQ_LENGTH}")
else:
    print("HF_REPO not set -- skipping push.")

# ── optional: push LoRA-only adapters (much smaller) ──────────────────────────
# model.push_to_hub_merged(f"{HF_REPO}-lora", tokenizer, save_method="lora")

# ── optional: push GGUF for llama.cpp / Ollama / LM Studio ───────────────────
# model.push_to_hub_gguf(
#     f"{HF_REPO}-gguf",
#     tokenizer,
#     quantization_method=["q4_k_m", "q8_0"],
# )

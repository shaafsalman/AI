"""
Qwen3.5-9B Dense LoRA Fine-Tune  (Unsloth + TRL, single GPU)
=============================================================
Hardware : 1 x A100 80GB  (BF16 LoRA on a 9B model needs ~22 GB for weights)
Run with : python dense_9b_unsloth.py

Key design decisions:
- BF16 LoRA (load_in_16bit). 4-bit is NOT recommended for Qwen3.5.
- lora_alpha == r, per Unsloth's guidance for these models.
- train_on_responses_only masks the system/user turns so loss is computed on
  the assistant turn only.
- response_part stops at "<|im_start|>assistant\\n" and does NOT include
  "<think>", so the model is trained to emit the whole assistant turn --
  the opening <think> tag and the final answer after </think>.
- Unsloth's merge can rewrite tokenizer/processor files into a form vLLM
  rejects, so the base model's config files are restored over the merged
  output before pushing.

Dataset schema (Alpaca-style): instruction, input, output.
The `output` field is expected to already contain <think>...</think> -- do not
pass enable_thinking to apply_chat_template here, or the tags get duplicated.
"""

import os
import shutil

from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

# ── config ────────────────────────────────────────────────────────────────────
# Override with environment variables, or edit the defaults.
MODEL_NAME   = os.environ.get("MODEL_NAME", "unsloth/Qwen3.5-9B")
BASE_MODEL   = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-9B")  # source of the canonical config files
DATASET_NAME = os.environ.get("DATASET_NAME", "")               # required -- HF dataset id
HF_REPO      = os.environ.get("HF_REPO", "")                    # optional -- push skipped if empty
HF_TOKEN     = os.environ.get("HF_TOKEN", "")                   # never hardcode a token here

MAX_SEQ_LEN = 4096
LORA_RANK   = 16
LORA_ALPHA  = 16   # keep equal to rank, per Unsloth docs

OUTPUT_DIR       = "qwen35_9b_checkpoints"
LORA_DIR         = "qwen35_9b_lora"
MERGED_DIR       = "qwen35_9b_merged"
BATCH_SIZE       = 8
GRAD_ACCUM       = 2
EPOCHS           = 1
LR               = 2e-4
WARMUP_STEPS     = 10
SEED             = 3407
DATASET_NUM_PROC = 8

# Set False if your dataset has no <think>...</think> reasoning blocks --
# the format and masking assertions below check for them.
EXPECT_THINK_TAGS = True

if not DATASET_NAME:
    raise SystemExit("Set DATASET_NAME (env var or edit the config block above).")

# ── model ─────────────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit   = False,   # NOT recommended for Qwen3.5
    load_in_8bit   = False,
    load_in_16bit  = True,    # BF16 -- ~22 GB VRAM for 9B
)

model = FastLanguageModel.get_peft_model(
    model,
    r                          = LORA_RANK,
    lora_alpha                 = LORA_ALPHA,
    target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
    lora_dropout               = 0,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = SEED,
)

# ── dataset ───────────────────────────────────────────────────────────────────
print(f"Loading dataset: {DATASET_NAME}")
ds = load_dataset(DATASET_NAME, split="train")
print(f"Dataset size: {len(ds):,} rows")


def format_dataset(examples):
    """Convert Alpaca-style batch rows into chat-template strings."""
    texts = []
    for instruction, inp, output in zip(
        examples["instruction"],
        examples["input"],
        examples["output"],
    ):
        convo = [
            {"role": "system",    "content": instruction.strip()},
            {"role": "user",      "content": inp.strip()},
            {"role": "assistant", "content": output.strip()},
        ]
        # Do NOT pass enable_thinking here:
        # - the output field already has <think>...</think> baked in
        # - enable_thinking is inference-time only (add_generation_prompt=True)
        # - passing it here would produce duplicate <think> tags
        texts.append(
            tokenizer.apply_chat_template(
                convo,
                tokenize              = False,
                add_generation_prompt = False,
            )
        )
    return {"text": texts}


ds = ds.map(
    format_dataset,
    batched        = True,
    num_proc       = DATASET_NUM_PROC,
    remove_columns = ds.column_names,
)

# Verify the rendered format before spending GPU hours on it.
sample = ds[0]["text"]
assert "<|im_start|>assistant" in sample, "chat template did not emit an assistant turn"
if EXPECT_THINK_TAGS:
    assert "<think>" in sample and "</think>" in sample, \
        "no <think> block found -- set EXPECT_THINK_TAGS=False if that is expected"
    assert sample.index("<think>") > sample.index("<|im_start|>assistant"), \
        "<think> appears before the assistant turn"
print("FORMAT CHECK PASSED")
print(sample[:1000], "\n")

# ── trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = ds,
    args          = SFTConfig(
        output_dir                  = OUTPUT_DIR,
        dataset_text_field          = "text",
        max_length                  = MAX_SEQ_LEN,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        num_train_epochs            = EPOCHS,
        warmup_steps                = WARMUP_STEPS,
        learning_rate               = LR,
        lr_scheduler_type           = "cosine",
        weight_decay                = 0.01,
        bf16                        = True,
        logging_steps               = 10,
        save_steps                  = 200,
        save_total_limit            = 3,
        optim                       = "adamw_8bit",
        seed                        = SEED,
        dataset_num_proc            = DATASET_NUM_PROC,
        packing                     = True,
        report_to                   = "none",
    ),
)

# Train on the full assistant turn -- NOT just the <think> block -- so the model
# also learns to produce the final answer after </think>.
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n",
)

# Verify masking: only assistant content should survive in the labels.
decoded = tokenizer.decode(
    [tokenizer.pad_token_id if x == -100 else x
     for x in trainer.train_dataset[0]["labels"]]
).replace(tokenizer.pad_token, " ")
assert "<|im_start|>user" not in decoded, "user turn is NOT masked -- fix response_part"
if EXPECT_THINK_TAGS:
    assert "<think>" in decoded, "assistant content missing from labels"
print("MASKING CHECK PASSED")
print(decoded[:600], "\n")

# ── train ─────────────────────────────────────────────────────────────────────
trainer.train()

# ── save LoRA adapter ─────────────────────────────────────────────────────────
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"LoRA adapter saved to {LORA_DIR}")

# ── merge to 16-bit ───────────────────────────────────────────────────────────
model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
print(f"Merged model saved to {MERGED_DIR}")

# ── restore the base model's config files ─────────────────────────────────────
# Unsloth's merge can alter config.json / tokenizer_config.json into a form vLLM
# refuses to load. Copy the originals from the base repo over the merged output
# so tokenizer, chat template and processor config match the base model exactly.
print("Restoring original base model config files...")

base_cache = snapshot_download(
    repo_id         = BASE_MODEL,
    token           = HF_TOKEN or None,
    ignore_patterns = ["*.safetensors", "*.bin", "*.pt"],   # config files only
)

CONFIG_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
]

for fname in CONFIG_FILES:
    src = os.path.join(base_cache, fname)
    dst = os.path.join(MERGED_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  restored {fname}")
    else:
        print(f"  skipped  {fname} (not in base model)")

print("Config restore complete.")

# ── push to hub ───────────────────────────────────────────────────────────────
if HF_TOKEN and HF_REPO:
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path    = MERGED_DIR,
        repo_id        = HF_REPO,
        repo_type      = "model",
        commit_message = "Add merged Qwen3.5-9B fine-tune (16-bit)",
    )
    print(f"Pushed to https://huggingface.co/{HF_REPO}")
else:
    print("HF_TOKEN or HF_REPO not set -- skipping push.")

print(f"\nServe with vLLM:\n  vllm serve {MERGED_DIR} --max-model-len {MAX_SEQ_LEN}")

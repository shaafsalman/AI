"""
Qwen3.8-27B Dense LoRA Fine-Tune  (Unsloth + TRL, single GPU)
==============================================================
Hardware  : 1 × H100 80GB  (BF16 LoRA peaks under 70GB)
Run with  : python dense_27b_unsloth.py

Key design decisions:
- FastModel, not FastLanguageModel. Qwen3.8 is natively multimodal, so
  from_pretrained returns a processor; the text tokenizer is extracted from it.
- BF16 LoRA only. 4-bit is broken on this architecture -- BitsandBytes
  dequantizes the gated DeltaNet in_proj_z to the wrong shape (right number of
  values, flattened to one row) and the first forward pass fails.
- target_modules is DISCOVERED, not hardcoded. Two thirds of the layers are
  gated linear-attention (DeltaNet) blocks whose projections are not named
  q_proj/k_proj/v_proj. A standard target list silently adapts ~25% of the
  network and underperforms in ways that look like a data problem.
- A guard asserts the adapter reached the DeltaNet blocks before training
  starts, so a wrong target list fails in seconds instead of after 3 hours.
- The merge is verified against the base weights. The causal-LM merge path
  injects LoRA at a different module path than the adapter keys use, matches
  nothing, and writes a directory that is byte-identical to base without
  raising.
"""

import os

# ── environment (must be set before any torch/unsloth import) ─────────────────
os.environ["CUDA_VISIBLE_DEVICES"]     = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTORCH_CUDA_ALLOC_CONF"]  = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]   = "false"
os.environ["UNSLOTH_COMPILE_DISABLE"]  = "1"   # mixed bf16/fp32 kernels on DeltaNet + LoRA
# HF_HOME is read into module constants at import time -- setting it after the
# import is a no-op and the model re-downloads into $HOME.
os.environ["HF_HOME"] = os.environ.get("HF_HOME", os.path.expanduser("~/hf_cache"))

# ── config ─────────────────────────────────────────────────────────────────────
# Override with environment variables, or edit the defaults.
MODEL_NAME       = os.environ.get("MODEL_NAME", "unsloth/Qwen3.8-27B")
DATASET_NAME     = os.environ.get("DATASET_NAME", "")   # required -- HF dataset id
DATASET_SPLIT    = "train"
MAX_SEQ_LENGTH   = 2048
LORA_RANK        = 32
LORA_ALPHA       = 32    # alpha == r, per Unsloth guidance

# Training hyperparameters (tuned for 1 × H100 80GB)
NUM_TRAIN_EPOCHS = 1
BATCH_SIZE       = 8
GRAD_ACCUM       = 1
WARMUP_STEPS     = 5
LEARNING_RATE    = 1e-4
SEED             = 3407
LOGGING_STEPS    = 10
DATASET_NUM_PROC = 1     # >1 forks after the model is loaded and deadlocks

OUTPUT_DIR  = "outputs_qwen38_27b"
MERGED_DIR  = "qwen38_27b_merged"
HF_REPO     = os.environ.get("HF_REPO", "")     # optional -- push skipped if empty
HF_TOKEN    = os.environ.get("HF_TOKEN", "")    # never hardcode a token here

if not DATASET_NAME:
    raise SystemExit("Set DATASET_NAME (env var or edit the config block above).")

import glob
import torch
from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── load model ─────────────────────────────────────────────────────────────────
# FastModel returns a multimodal processor. Passing it straight to SFTTrainer
# makes apply_chat_template parse string content as vision dicts and crash with
# "TypeError: string indices must be integers".
model, processor = FastModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LENGTH,
    load_in_4bit    = False,   # broken on DeltaNet -- see docstring
    load_in_16bit   = True,
    full_finetuning = False,
)
tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

# ── discover LoRA targets ──────────────────────────────────────────────────────
# The gated DeltaNet blocks do not use q_proj/k_proj/v_proj names, and the
# language weights sit one level deeper in the module tree than on a plain
# causal LM. Collect every leaf Linear under the language model instead of
# guessing, and skip the vision tower and the output head.
SKIP = ("lm_head", "visual", "vision", "merger", "embed")

target_modules = sorted({
    name.split(".")[-1]
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear)
    and not any(s in name for s in SKIP)
})

# DeltaNet projections carry their own names; anything matching these is a
# linear-attention projection rather than standard self-attention.
DELTANET_HINTS = ("in_proj", "out_proj", "ba_proj", "conv")
deltanet_targets = [t for t in target_modules if any(h in t for h in DELTANET_HINTS)]

print(f"[lora] target_modules ({len(target_modules)}): {target_modules}")
print(f"[lora] deltanet targets ({len(deltanet_targets)}): {deltanet_targets}")

assert deltanet_targets, (
    "No DeltaNet projections found in target_modules. Two thirds of this "
    "model's layers are gated linear-attention blocks; adapting only q/k/v "
    "leaves them untrained. Print model.named_modules() and widen DELTANET_HINTS."
)

# ── LoRA adapters ──────────────────────────────────────────────────────────────
model = FastModel.get_peft_model(
    model,
    r                          = LORA_RANK,
    lora_alpha                 = LORA_ALPHA,
    lora_dropout               = 0,
    bias                       = "none",
    target_modules             = target_modules,
    use_gradient_checkpointing = "unsloth",
    random_state               = SEED,
    use_rslora                 = False,
    loftq_config               = None,
)
model.config.use_cache = False

# ── guard: the adapter actually landed on the DeltaNet blocks ─────────────────
adapted = [n for n, _ in model.named_modules() if "lora_A" in n]
adapted_deltanet = [n for n in adapted if any(h in n for h in DELTANET_HINTS)]
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())

print(f"[lora] {len(adapted)} adapted modules | {len(adapted_deltanet)} gated-attention")
print(f"[lora] trainable {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

assert adapted_deltanet, "LoRA did not adapt any DeltaNet layer -- fix target_modules"

# ── dataset formatting ─────────────────────────────────────────────────────────
# Alpaca-style columns rendered into ChatML. Reasoning belongs inside `output`
# as <think>...</think>; do not pass enable_thinking=True here, it is an
# inference-time flag and produces duplicate <think> tags in training.
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

print(f"Loading dataset: {DATASET_NAME} ...")
dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
dataset = dataset.map(
    format_dataset,
    batched        = True,
    remove_columns = dataset.column_names,
    num_proc       = DATASET_NUM_PROC,
)
print(f"Dataset size: {len(dataset):,} examples")

# ── trainer ────────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model            = model,
    processing_class = tokenizer,   # not tokenizer= -- deprecated in TRL
    train_dataset    = dataset,
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
        max_grad_norm               = 0.3,
        bf16                        = True,
        logging_steps               = LOGGING_STEPS,
        save_steps                  = 100,
        save_total_limit            = 2,
        output_dir                  = OUTPUT_DIR,
        optim                       = "adamw_8bit",
        seed                        = SEED,
        dataset_num_proc            = DATASET_NUM_PROC,
        dataloader_num_workers      = 0,   # forks deadlock after a 27B load
        packing                     = True,
        report_to                   = "none",
    ),
)

# ── train on responses only ────────────────────────────────────────────────────
# Loss on assistant tokens only. "<think>" in response_part means the opening
# tag is treated as prompt and the reasoning content is trained.
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n<think>",
)

# ── train ──────────────────────────────────────────────────────────────────────
ckpts  = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
resume = ckpts[-1] if ckpts else False
print(f"Resuming from: {resume}" if resume else "Starting fresh")

trainer.train(resume_from_checkpoint=resume)

# ── save ───────────────────────────────────────────────────────────────────────
print("Training complete. Saving LoRA adapters...")
model.save_pretrained(OUTPUT_DIR + "_lora")
tokenizer.save_pretrained(OUTPUT_DIR + "_lora")

print("Merging to 16-bit...")
model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
print(f"Saved merged model to: {MERGED_DIR}")

# ── verify the merge ───────────────────────────────────────────────────────────
# A merge that matched nothing writes a directory identical to base and does not
# raise. Compare the shards against the cached snapshot before trusting it.
# The empty-comparison case must fail: "0 of 0 tensors differ" is not a pass.
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

base_dir = snapshot_download(MODEL_NAME, allow_patterns=["*.safetensors"])
compared = changed = 0
for shard in sorted(glob.glob(os.path.join(MERGED_DIR, "*.safetensors"))):
    base_shard = os.path.join(base_dir, os.path.basename(shard))
    if not os.path.exists(base_shard):
        continue
    merged_w, base_w = load_file(shard), load_file(base_shard)
    for key in merged_w.keys() & base_w.keys():
        compared += 1
        if not torch.equal(merged_w[key], base_w[key]):
            changed += 1

print(f"[verify] {changed} of {compared} tensors differ from base")
assert compared > 0, "verifier compared nothing -- key names did not match, fix before trusting"
assert changed  > 0, "adapter did NOT apply -- merged model is identical to base"
print("[verify] adapter APPLIED")

# ── push ───────────────────────────────────────────────────────────────────────
if HF_TOKEN and HF_REPO:
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO, exist_ok=True, private=False)
    print(f"Uploading to {HF_REPO} ...")
    api.upload_folder(folder_path=MERGED_DIR, repo_id=HF_REPO, repo_type="model")
    print("Upload complete!")
else:
    print("HF_TOKEN or HF_REPO not set -- skipping upload.")

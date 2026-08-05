"""
AWQ W8A16 Quantization for Qwen3.5-9B (VL architecture)
========================================================
Method : Activation-aware Weight Quantization (AWQ), calibration-based
Output : ~13 GB  (vs ~19 GB BF16 original)
Time   : ~35-45 minutes on an A100 40GB

Use this when you can spare the time and have in-domain calibration data --
AWQ picks per-channel scales from real activations, so it holds accuracy
better than the data-free RTN pass in rtn_w8a16.py.

Architecture notes (Qwen3.5-9B):
  - Unified VLM: the vision encoder is fused, not a separate tower
  - Hybrid layers: 8x full-attention + 24x GDN (linear_attn) blocks
  - GDN layers are left in BF16, matching Qwen's own FP8/GPTQ releases
  - offload_device=cpu is required to avoid OOM during the AWQ scale search

Example:
    python awq_w8a16.py \\
        --model_id  username/my-finetuned-qwen35-9b \\
        --dataset_id username/my-calibration-set \\
        --hf_repo   username/my-qwen35-9b-awq-w8a16
"""

import argparse
import os

import torch
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText

parser = argparse.ArgumentParser(description="AWQ W8A16 quantization for Qwen3.5 VL models")
parser.add_argument("--model_id",    required=True, help="HF repo id or local path of the model to quantize")
parser.add_argument("--dataset_id",  required=True, help="HF dataset id used for calibration")
parser.add_argument("--save_path",   default=None,  help="output dir (default: /tmp/<model>-awq-w8a16)")
parser.add_argument("--hf_repo",     default=os.environ.get("HF_REPO", ""),
                    help="destination HF repo; push is skipped if unset")
parser.add_argument("--num_samples", type=int, default=256, help="calibration samples")
parser.add_argument("--max_seq_len", type=int, default=2048, help="calibration sequence length")
parser.add_argument("--no_push",     action="store_true", help="quantize only, never upload")
args = parser.parse_args()

if args.save_path is None:
    args.save_path = f"/tmp/{args.model_id.split('/')[-1]}-awq-w8a16"

os.makedirs(args.save_path, exist_ok=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("=" * 62)
print(f"  Method   : AWQ W8A16 (calibration-based)")
print(f"  Model    : {args.model_id}")
print(f"  Dataset  : {args.dataset_id} ({args.num_samples} samples)")
print(f"  Save to  : {args.save_path}")
print(f"  HF repo  : {args.hf_repo or '(none -- push skipped)'}")
print("=" * 62)

print("\n[1/5] Loading tokenizer + processor...")
tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

print("[2/5] Loading full VL model in BF16...")
model = AutoModelForImageTextToText.from_pretrained(
    args.model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"      Architecture : {type(model).__name__}")

print(f"\n[3/5] Building calibration dataset from {args.dataset_id}...")
raw = load_dataset(args.dataset_id, split="train")
raw = raw.shuffle(seed=42).select(range(args.num_samples))

calib_texts = []
for sample in raw:
    # Adjust field names to match your dataset schema
    instruction  = sample.get("instruction", "").strip()
    user_input   = sample.get("input", "").strip()
    model_output = sample.get("output", "").strip()
    user_content = f"{instruction}\n\n{user_input}" if user_input else instruction
    messages = [
        {"role": "system",    "content": "You are a helpful assistant. Think step by step before answering."},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": model_output},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    calib_texts.append(text.strip())

calib_dataset = Dataset.from_dict({"text": calib_texts})
print(f"      Prepared {len(calib_texts)} samples.")

IGNORE_LAYERS = [
    "lm_head",
    "re:visual.*",
    "re:merger.*",
    "re:.*linear_attn.*",
]

print("\n[4/5] Running AWQ W8A16 quantization (~35-45 min on A100)...")
print("      INT8   : full-attention + MLP layers")
print("      BF16   : lm_head | visual.* | merger.* | linear_attn.*")
print("      OOM fix: offload_device=cpu")

recipe = [
    AWQModifier(
        duo_scaling="both",
        offload_device=torch.device("cpu"),
    ),
    QuantizationModifier(
        targets=["Linear"],
        scheme="W8A16",
        ignore=IGNORE_LAYERS,
    ),
]

oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=calib_dataset,
    recipe=recipe,
    max_seq_length=args.max_seq_len,
    num_calibration_samples=args.num_samples,
    output_dir=args.save_path,
)

processor.save_pretrained(args.save_path)
size = os.popen(f"du -sh {args.save_path}").read().strip()
print(f"\n      Saved : {args.save_path}")
print(f"      Size  : {size}")

if args.hf_repo and not args.no_push:
    print(f"\n[5/5] Pushing to HuggingFace: {args.hf_repo}...")
    api = HfApi()
    api.create_repo(repo_id=args.hf_repo, exist_ok=True)
    api.upload_folder(folder_path=args.save_path, repo_id=args.hf_repo, repo_type="model")
    print(f"\n  Done. https://huggingface.co/{args.hf_repo}")
else:
    print(f"\n[5/5] Push skipped. Model at {args.save_path}")

print(f"""
================================================================
  Serve with vLLM:
================================================================
  VLLM_USE_DEEP_GEMM=0 vllm serve {args.save_path} \\
    --port 8000 \\
    --tensor-parallel-size 1 \\
    --max-model-len 32000 \\
    --gpu-memory-utilization 0.95 \\
    --reasoning-parser deepseek_r1 \\
    --gdn-prefill-backend triton \\
    --default-chat-template-kwargs '{{"enable_thinking": true}}' \\
    --trust-remote-code
================================================================
""")

"""
W8A16 Data-Free Quantization for Qwen3.5-9B (VL architecture)
==============================================================
Method : Round-to-Nearest (RTN) -- no calibration data required
Output : ~11 GB  (vs ~19 GB BF16 original)
Time   : ~3 minutes on any GPU

Start here. If the accuracy drop is acceptable you are done in minutes; if it
isn't, reach for the calibration-based AWQ pass in awq_w8a16.py.

Note this quantizes the GDN (linear_attn) layers too, which is why the output
is smaller than the AWQ result -- and also why it can lose more accuracy.

Architecture notes (Qwen3.5-9B):
  - Unified VLM: the vision encoder is fused, not a separate tower
  - Hybrid layers: 8x full-attention + 24x GDN (linear_attn) blocks
  - Vision encoder + merger + lm_head stay in BF16

Example:
    python rtn_w8a16.py \\
        --model_id username/my-finetuned-qwen35-9b \\
        --hf_repo  username/my-qwen35-9b-w8a16
"""

import argparse
import os

import torch
from huggingface_hub import HfApi
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoProcessor, AutoModelForImageTextToText

parser = argparse.ArgumentParser(description="Data-free W8A16 RTN quantization for Qwen3.5 VL models")
parser.add_argument("--model_id",  required=True, help="HF repo id or local path of the model to quantize")
parser.add_argument("--save_path", default=None,  help="output dir (default: /tmp/<model>-w8a16)")
parser.add_argument("--hf_repo",   default=os.environ.get("HF_REPO", ""),
                    help="destination HF repo; push is skipped if unset")
parser.add_argument("--no_push",   action="store_true", help="quantize only, never upload")
args = parser.parse_args()

if args.save_path is None:
    args.save_path = f"/tmp/{args.model_id.split('/')[-1]}-w8a16"

os.makedirs(args.save_path, exist_ok=True)

print("=" * 62)
print(f"  Method   : W8A16 RTN (data-free)")
print(f"  Model    : {args.model_id}")
print(f"  Save to  : {args.save_path}")
print(f"  HF repo  : {args.hf_repo or '(none -- push skipped)'}")
print("=" * 62)

print("\n[1/4] Loading processor...")
processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

print("[2/4] Loading full VL model in BF16...")
model = AutoModelForImageTextToText.from_pretrained(
    args.model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"      Architecture : {type(model).__name__}")

IGNORE_LAYERS = [
    "lm_head",
    "re:visual.*",
    "re:merger.*",
]

recipe = [
    QuantizationModifier(
        targets=["Linear"],
        scheme="W8A16",
        ignore=IGNORE_LAYERS,
    ),
]

print("\n[3/4] Quantizing (data-free, ~3 min)...")
print("      INT8   : full-attention + MLP + linear_attn layers")
print("      BF16   : lm_head | visual.* | merger.*")
oneshot(model=model, recipe=recipe, output_dir=args.save_path)

processor.save_pretrained(args.save_path)
size = os.popen(f"du -sh {args.save_path}").read().strip()
print(f"\n      Saved : {args.save_path}")
print(f"      Size  : {size}")

if args.hf_repo and not args.no_push:
    print(f"\n[4/4] Pushing to HuggingFace: {args.hf_repo}...")
    api = HfApi()
    api.create_repo(repo_id=args.hf_repo, exist_ok=True)
    api.upload_folder(folder_path=args.save_path, repo_id=args.hf_repo, repo_type="model")
    print(f"\n  Done. https://huggingface.co/{args.hf_repo}")
else:
    print(f"\n[4/4] Push skipped. Model at {args.save_path}")

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

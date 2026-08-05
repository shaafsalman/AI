# Quantization

INT8 weight quantization for Qwen3.5, via
[llm-compressor](https://github.com/vllm-project/llm-compressor), producing
checkpoints vLLM serves directly.

Both scripts use the **W8A16** scheme: 8-bit weights, 16-bit activations. That
halves the weight memory while leaving activations in BF16, which is the
conservative choice — W8A8 saves more but needs activation calibration and
gives up more accuracy.

---

## Which one

| | `rtn_w8a16.py` | `awq_w8a16.py` |
|---|---|---|
| Method | Round-to-nearest | Activation-aware (AWQ) |
| Calibration data | none | required |
| Time | ~3 min | ~35-45 min (A100 40GB) |
| Output (from 9B BF16, ~19 GB) | ~11 GB | ~13 GB |
| Accuracy | good | better |

**Start with RTN.** It's data-free and takes three minutes; if the accuracy
drop is acceptable you're done. Reach for AWQ when it isn't — AWQ derives
per-channel scales from real activations, so it protects the weights that
matter most to your workload, at the cost of needing a few hundred in-domain
samples.

The size difference is not a typo and not in RTN's favour: RTN also quantizes
the GDN (`linear_attn`) layers, which AWQ deliberately leaves in BF16. That
extra coverage is exactly where RTN gives up its accuracy.

---

## Setup

```bash
cd qwen3.5
bash setup.sh
source quant_env/bin/activate
```

`setup.sh` builds an isolated venv with torch 2.11 (CUDA 12.8), `transformers`
from source (Qwen3.5 needs it), vLLM nightly, and llm-compressor from source.
The last step re-pins `transformers` and `huggingface_hub` because installing
llm-compressor downgrades both — don't drop it. The script verifies every
import and prints the resolved versions before exiting.

This environment is separate from the fine-tuning ones on purpose; the pins
conflict.

## Run

Data-free:

```bash
python rtn_w8a16.py \
    --model_id username/my-finetuned-qwen35-9b \
    --hf_repo  username/my-qwen35-9b-w8a16
```

Calibration-based:

```bash
python awq_w8a16.py \
    --model_id   username/my-finetuned-qwen35-9b \
    --dataset_id username/my-calibration-set \
    --hf_repo    username/my-qwen35-9b-awq-w8a16 \
    --num_samples 256
```

`--model_id` is required. `--hf_repo` falls back to `$HF_REPO` and the upload
is skipped when neither is set, so a bare run stays local; `--no_push` forces
that. `--save_path` defaults to `/tmp/<model-name>-<method>`.

Both scripts print a ready-to-paste `vllm serve` command when they finish.

---

## What stays in BF16

Quantizing everything degrades quality for very little extra saving, so some
layers are excluded:

| Layer pattern | RTN | AWQ | Why |
|---|---|---|---|
| `lm_head` | BF16 | BF16 | Output projection; quantization hits token probabilities directly. |
| `visual.*` | BF16 | BF16 | Vision encoder — small, and sensitive. |
| `merger.*` | BF16 | BF16 | Vision-to-text projection. |
| `linear_attn.*` (GDN) | **INT8** | BF16 | AWQ skips these, matching Qwen's own FP8/GPTQ releases. |

Qwen3.5-9B is a unified VLM — the vision encoder is fused into the model rather
than being a separate tower, so it must be loaded with
`AutoModelForImageTextToText`, not `AutoModelForCausalLM`. Its layers are
hybrid: 8 full-attention blocks interleaved with 24 GDN (gated delta net)
blocks.

## Serving

```bash
VLLM_USE_DEEP_GEMM=0 vllm serve <model-path> \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32000 \
    --gpu-memory-utilization 0.95 \
    --reasoning-parser deepseek_r1 \
    --gdn-prefill-backend triton \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --trust-remote-code
```

Two flags are load-bearing for this architecture: `--gdn-prefill-backend
triton` for the GDN layers, and `VLLM_USE_DEEP_GEMM=0`, which avoids a DeepGEMM
path that misbehaves with these quantized weights. Drop `--reasoning-parser`
and the `enable_thinking` kwarg if the model isn't a reasoning fine-tune.

---

## Troubleshooting

**OOM during the AWQ scale search.** Already mitigated by
`offload_device=torch.device("cpu")` in the recipe. If it still OOMs, lower
`--num_samples` or `--max_seq_len`.

**Calibration field names.** `awq_w8a16.py` builds its calibration prompts from
`instruction` / `input` / `output` columns. A dataset with different column
names needs that loop edited — it's a dozen lines, clearly marked.

# LLM Fine-Tuning, Merging and Quantization Scripts

Production scripts for taking an open-weight model from base checkpoint to a
served, quantized endpoint:

```
fine-tune (LoRA)  ──►  merge  ──►  quantize  ──►  serve (vLLM)
```

Everything here currently targets the **Qwen3.5** family, across dense and MoE
variants, on both the Unsloth and the stock HuggingFace training stacks. The
layout is organised by stage and then by model family, so other families slot
in alongside `qwen3.5/` without reshuffling anything.

Each script is **standalone** — a single file you can copy onto a GPU box and
run. That means a little duplication between them (the vLLM config patches, the
dataset formatter), and that is deliberate: no shared package to install, no
import paths to fix, nothing to break when you run one script on a machine that
only has that one file.

---

## Repository layout

```
.
├── finetuning/
│   ├── README.md                    stack comparison, dataset schema, launching
│   └── qwen3.5/
│       ├── dense_4b_unsloth.py      Qwen3.5-4B     · Unsloth  · 8-GPU DDP
│       ├── dense_9b_hf.py           Qwen3.5-9B     · HF/TRL   · 8-GPU DDP
│       ├── dense_9b_unsloth.py      Qwen3.5-9B     · Unsloth  · single GPU
│       ├── dense_27b_unsloth.py     Qwen3.5-27B    · Unsloth  · 8-GPU DDP
│       ├── moe_35b_hf.py            Qwen3.5-35B-A3B MoE · HF/TRL  · 8-GPU DDP
│       ├── moe_35b_unsloth.py       Qwen3.5-35B-A3B MoE · Unsloth · single GPU
│       ├── requirements-hf.txt
│       └── requirements-unsloth.txt
├── merging/
│   ├── README.md                    SLERP vs DARE, choosing a ratio
│   └── qwen3.5/
│       ├── slerp_dare_ties.py       CPU weight-space merge, two outputs
│       └── requirements.txt
└── quantization/
    ├── README.md                    method comparison, layer policy
    └── qwen3.5/
        ├── setup.sh                 builds the quantization venv
        ├── rtn_w8a16.py             data-free, ~3 min
        └── awq_w8a16.py             calibration-based, ~35-45 min
```

---

## Script index

### Fine-tuning — `finetuning/qwen3.5/`

| Script | Model | Stack | GPUs | Launch |
|---|---|---|---|---|
| `dense_4b_unsloth.py` | Qwen3.5-4B dense | Unsloth + TRL | 8 × A100 80GB | `torchrun --nproc_per_node=8` |
| `dense_9b_hf.py` | Qwen3.5-9B dense | HF + PEFT + TRL | 8 × A100 40/80GB | `torchrun --nproc_per_node=8` |
| `dense_9b_unsloth.py` | Qwen3.5-9B dense | Unsloth + TRL | 1 × A100 80GB | `python` |
| `dense_27b_unsloth.py` | Qwen3.5-27B dense | Unsloth + TRL | 8 × A100 80GB | `torchrun --nproc_per_node=8` |
| `moe_35b_hf.py` | Qwen3.5-35B-A3B MoE | HF + PEFT + TRL | 8 × A100 80GB | `torchrun --nproc_per_node=8` |
| `moe_35b_unsloth.py` | Qwen3.5-35B-A3B MoE | Unsloth | 1 × A100/H100 80GB | `python` |

All six train LoRA in BF16, merge the adapter into the base weights, and
optionally push the merged model to the Hub. Five also save the standalone LoRA
adapter alongside it; `moe_35b_unsloth.py` goes straight from training to a
merged push. See [`finetuning/README.md`](finetuning/README.md) for how to pick
between them.

### Merging — `merging/qwen3.5/`

`slerp_dare_ties.py` blends a base and a fine-tuned checkpoint in weight space,
producing a SLERP merge and a DARE merge in one pass so you can evaluate both.
Runs on CPU. Useful for clawing back general capability that a narrow fine-tune
eroded.

### Quantization — `quantization/qwen3.5/`

| Script | Method | Time | Output size | Needs data |
|---|---|---|---|---|
| `rtn_w8a16.py` | Round-to-nearest | ~3 min | ~11 GB | no |
| `awq_w8a16.py` | Activation-aware (AWQ) | ~35-45 min | ~13 GB | yes |

Both keep the vision encoder, merger and `lm_head` in BF16 and emit a
`vllm serve` command when they finish.

---

## Quickstart

Fine-tune, then quantize the result:

```bash
git clone https://github.com/shaafsalman/AI.git
cd AI
```

```bash
export HF_TOKEN=hf_...
export DATASET_NAME=username/my-dataset
export HF_REPO=username/my-qwen35-9b

pip install -r finetuning/qwen3.5/requirements-hf.txt
torchrun --nproc_per_node=8 finetuning/qwen3.5/dense_9b_hf.py
```

```bash
cd quantization/qwen3.5 && bash setup.sh && source quant_env/bin/activate
python rtn_w8a16.py --model_id username/my-qwen35-9b --hf_repo username/my-qwen35-9b-w8a16
```

---

## Configuration

Every script reads its configuration from environment variables, falling back
to the defaults in the `── config ──` block near the top of the file. Nothing
is hardcoded and no secret is ever committed.

| Variable | Used by | Meaning |
|---|---|---|
| `DATASET_NAME` | fine-tuning | **Required.** HF dataset id. Scripts exit immediately if unset. |
| `MODEL_NAME` | fine-tuning, merging | Base model to train from. Defaults per script. |
| `HF_REPO` | all | Destination repo. **Push is skipped when unset**, so a run without it stays entirely local. |
| `HF_TOKEN` | all | Hub auth. Also settable via `hf auth login`. |
| `BASE_MODEL` | `dense_9b_unsloth.py`, merging | Canonical repo to pull config files from. |
| `FINETUNED_MODEL` | merging | **Required.** The checkpoint to blend into the base. |

The quantization scripts take CLI flags instead of env vars (`--model_id`,
`--dataset_id`, `--hf_repo`), since they are run ad hoc against a finished
model. `--model_id` is required; `--hf_repo` falls back to `$HF_REPO`.

---

## Notes

**Dataset schema.** The fine-tuning scripts expect Alpaca-style columns —
`instruction`, `input`, `output` — and render them into Qwen's ChatML format.
For reasoning fine-tunes the `output` field should already contain its
`<think>...</think>` block; do not ask the chat template to add one, or you get
duplicate tags.

**vLLM compatibility.** Merged checkpoints frequently need patching before vLLM
will load them — multimodal keys stripped from `config.json`, `processor_class`
removed from the tokenizer config, a rebuilt safetensors index. `dense_9b_hf.py`
carries the most complete set of these fixes and is the best file to copy from
when a merged model refuses to serve.

**Hardware.** Every script is annotated with the hardware it was actually run
on. They will run on less, but the batch sizes and gradient-accumulation steps
assume A100-class GPUs and want lowering otherwise.

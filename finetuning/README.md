# Fine-tuning

LoRA fine-tuning scripts for Qwen3.5 and Qwen3.8. Every script follows the same arc:

```
load base (BF16)  ─►  attach LoRA  ─►  format dataset  ─►  train
                                                            │
                    push to Hub  ◄─  patch for vLLM  ◄─  merge adapter
```

---

## Choosing a stack

Two of these models are covered twice, once per training stack, because the
tradeoff is real and depends on the box you're on.

**Unsloth** (`*_unsloth.py`) is faster and uses noticeably less VRAM —
`use_gradient_checkpointing="unsloth"` alone saves around 30% on long
sequences. The cost is version sensitivity: it patches `transformers` at import
time, so it wants specific pinned versions, and every `os.environ` line has to
execute *before* the first Unsloth import or the patches land wrong. Its merge
step also tends to write tokenizer and processor files that vLLM then refuses
to load, which is why both `dense_9b_unsloth.py` and `dense_4b_unsloth.py`
finish by restoring the base model's config files over the merged output.

**Stock HuggingFace** (`*_hf.py`) — `transformers` + `peft` + `trl` — is slower
and hungrier, but it runs on unpinned upstream versions, fails in ways that are
straightforward to debug, and its stack traces point at code you can read.

Use Unsloth when throughput or VRAM is the binding constraint. Use the HF stack
when you want the run to be reproducible six months from now.

## Choosing a script

| Script | Model | Stack | GPUs | Launch |
|---|---|---|---|---|
| `dense_4b_unsloth.py` | Qwen3.5-4B dense | Unsloth | 8 × A100 80GB | `torchrun --nproc_per_node=8` |
| `dense_9b_hf.py` | Qwen3.5-9B dense | HF/TRL | 8 × A100 40/80GB | `torchrun --nproc_per_node=8` |
| `dense_9b_unsloth.py` | Qwen3.5-9B dense | Unsloth | 1 × A100 80GB | `python` |
| `dense_27b_unsloth.py` | Qwen3.5-27B dense | Unsloth | 8 × A100 80GB | `torchrun --nproc_per_node=8` |
| `moe_35b_hf.py` | Qwen3.5-35B-A3B MoE | HF/TRL | 8 × A100 80GB | `torchrun --nproc_per_node=8` |
| `moe_35b_unsloth.py` | Qwen3.5-35B-A3B MoE | Unsloth | 1 × A100/H100 80GB | `python` |
| `../qwen3.8/dense_27b_unsloth.py` | Qwen3.8-27B dense | Unsloth | 1 × H100 80GB | `python` |

`qwen3.8/dense_27b_unsloth.py` is the odd one out. Qwen3.8-27B is a hybrid
attention model: two thirds of its layers are gated linear-attention (DeltaNet)
blocks whose projections are not named `q_proj` / `k_proj` / `v_proj`. A target
list copied from a standard transformer silently adapts about a quarter of the
network. That script therefore discovers `target_modules` by walking the module
tree, asserts the adapter reached the DeltaNet blocks before training starts,
and verifies the merge against the base weights afterwards. It is also BF16
only — 4-bit dequantizes the DeltaNet `in_proj_z` to the wrong shape and dies on
the first forward pass.

The 35B-A3B is a Mixture-of-Experts model: 35B total parameters, roughly 3B
active per token. It fits on a single 80GB card in BF16 LoRA, which is why the
Unsloth variant is single-GPU — run several independent sweeps in parallel, one
per GPU, via `CUDA_VISIBLE_DEVICES`.

---

## Install

```bash
pip install -r qwen3.5/requirements-hf.txt        # for the *_hf.py scripts
pip install -r qwen3.5/requirements-unsloth.txt   # for the *_unsloth.py scripts
pip install -r qwen3.8/requirements-unsloth.txt   # for qwen3.8/dense_27b_unsloth.py
```

Install them into **separate** virtual environments. Unsloth pins versions that
the stock stack does not want.

`moe_35b_unsloth.py` needs a `--no-deps` install to avoid Unsloth pulling in a
conflicting `transformers`; the exact sequence is in that file's docstring.

## Run

```bash
export HF_TOKEN=hf_...
export DATASET_NAME=username/my-dataset
export HF_REPO=username/my-model      # omit to keep everything local

torchrun --nproc_per_node=8 qwen3.5/dense_9b_hf.py
```

Scripts exit immediately with a clear message if `DATASET_NAME` is unset,
rather than failing thirty seconds later inside `load_dataset`.

**Checkpoint resume.** `dense_9b_hf.py`, `moe_35b_hf.py` and
`dense_27b_unsloth.py` automatically resume from the newest `checkpoint-*`
directory under their output dir — delete it to force a fresh run. The other
three always start from scratch; they still write periodic checkpoints, but
picking one up means passing it to `trainer.train(resume_from_checkpoint=...)`
yourself.

---

## Dataset format

Alpaca-style columns:

| Column | Role |
|---|---|
| `instruction` | task description → **system** turn (or prepended to the user turn) |
| `input` | optional extra context → **user** turn |
| `output` | the target completion → **assistant** turn |

Rendered into Qwen's ChatML:

```
<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

### Reasoning fine-tunes

For chain-of-thought training, `output` should already contain its reasoning
block:

```
<think>
step-by-step reasoning ...
</think>
Final concise answer.
```

Do **not** pass `enable_thinking=True` to `apply_chat_template` during
training. It is an inference-time flag that goes with
`add_generation_prompt=True`; passing it here produces duplicate `<think>` tags
that the model then learns to emit.

`dense_9b_unsloth.py` asserts this format is correct — and that the user turn
is properly masked out of the labels — before training starts, so a
mis-rendered dataset fails in seconds instead of after an epoch. If your
dataset has no reasoning blocks, set `EXPECT_THINK_TAGS = False` in that file.

---

## Loss masking

The Unsloth scripts use `train_on_responses_only` so loss is computed on the
assistant turn only, not on the prompt.

`response_part` deliberately stops at `"<|im_start|>assistant\n"` in
`dense_9b_unsloth.py`, so the whole assistant turn is trained — the opening
`<think>` tag, the reasoning, and the final answer after `</think>`.
`dense_27b_unsloth.py` extends it to `"<|im_start|>assistant\n<think>"`, which
trains the reasoning content but treats the opening tag as part of the prompt.
Either works; be aware which one you picked, because it changes what the model
is expected to generate first.

`moe_35b_unsloth.py` skips masking entirely and trains on the full sequence.

---

## After training

Each script merges the LoRA adapter into the base weights and writes a
standalone 16-bit model that vLLM can serve without adapter-loading overhead.

If the merged model won't load in vLLM, the fix is almost always one of three
things, all implemented in `dense_9b_hf.py`:

- **`config.json`** still carries multimodal or `mrope_*` keys — strip them and
  force the text-only architecture and `model_type`.
- **`tokenizer_config.json`** has a `processor_class`, or a `tokenizer_class`
  of `TokenizersBackend` — remove the former, set the latter to a real class.
- **`model.safetensors.index.json`** is stale — rebuild it from the shards
  actually on disk.

From here, continue to [`../merging`](../merging) to blend the fine-tune back
toward the base model, or straight to [`../quantization`](../quantization).

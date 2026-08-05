# Merging

Weight-space blending of a base checkpoint and a fine-tuned one.

A narrow fine-tune usually costs you general capability: the model gets better
at your task and worse at everything else. Merging some of the base weights
back in trades a little task performance for a lot of that generality. It costs
one CPU pass and no retraining, so it's cheap to try.

`qwen3.5/slerp_dare_ties.py` produces **two** merged models in a single run, so
you can evaluate both and keep the better one.

---

## The two methods

**SLERP** — spherical linear interpolation. Walks the arc between the base and
fine-tuned weight vectors rather than the straight line, so the result keeps a
weight norm close to both parents. A plain linear blend shrinks the norm when
the two vectors diverge, which is what tends to make naive averaging feel
"muddy". Falls back to a linear blend when the two vectors are nearly parallel
and the arc is numerically unstable.

**DARE** — Drop And REscale. Takes the delta (`fine-tuned − base`), randomly
drops `1 − DARE_DENSITY` of it, rescales the survivors by `1/density` to
preserve the delta's expected magnitude, and adds `FT_RATIO ×` that back onto
the base. The insight is that fine-tuning deltas are highly redundant, so most
of them can be discarded with little loss.

> **On the name.** The output directory is `merged-dare-ties/` and the file is
> `slerp_dare_ties.py`, both kept for backwards compatibility, but this
> implements DARE only. There is no TIES sign-election step — that resolves
> sign conflicts across three or more task vectors, and this script merges
> exactly two checkpoints.

---

## Run

```bash
pip install -r qwen3.5/requirements.txt

export BASE_MODEL=Qwen/Qwen3.5-9B
export FINETUNED_MODEL=username/my-finetuned-model

python qwen3.5/slerp_dare_ties.py
```

Outputs:

```
./merged-slerp/          SLERP blend
./merged-dare-ties/      DARE blend
./cache_base/            downloaded base weights (reused across runs)
./cache_ft/              downloaded fine-tuned weights
```

Both outputs get the fine-tuned model's config files copied in, so they are
ready to serve or quantize directly.

## Requirements

CPU-only, but memory-hungry: it holds both checkpoints in RAM at once, so
budget roughly **2× the model size** — about 40 GB for a 9B in BF16. No GPU
needed. Downloads are cached, so re-running with a different `FT_RATIO` skips
straight to the merge.

---

## Tuning the ratio

`FT_RATIO` controls how much fine-tune to blend in — `0.0` is the pure base
model, `1.0` is the pure fine-tune. The default `0.30` means 70% base / 30%
fine-tuned, which is a conservative starting point that preserves most general
capability.

Sweep it. The download cache means each extra ratio costs only the merge and
the eval:

| `FT_RATIO` | Effect |
|---|---|
| 0.1 – 0.2 | Barely moves off the base. Try when the fine-tune badly damaged general ability. |
| 0.3 – 0.5 | The usual useful range. |
| 0.6 – 0.8 | Mostly fine-tune, slight regularisation toward base. |

`DARE_DENSITY` (default `0.70`) is the fraction of delta weights kept before
rescaling, and applies to the DARE output only. Lower values prune harder;
values below ~0.5 start to lose task performance noticeably.

---

## Edge cases

The script handles two mismatches without complaint, worth knowing about
because they're silent:

- **Tensors only in the fine-tuned model** (added tokens, resized embeddings)
  are copied through as-is.
- **Tensors whose shapes disagree** between the two checkpoints take the
  fine-tuned version wholesale — no blending is possible.

Scalar tensors are linearly interpolated under SLERP (an arc through a single
point is meaningless) and taken from the fine-tune under DARE.

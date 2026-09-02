# Pipeline

One driver script that runs the whole arc on a single GPU:

```
data  ──►  train  ──►  merge + verify  ──►  serve  ──►  evaluate  ──►  ship
```

The other three directories in this repo are standalone scripts you copy onto a
box and run. This one is different: `run_pipeline.sh` is an *orchestrator*. It
sequences steps, guards each one, and refuses to continue when a guard fails —
but the steps themselves live in helper scripts alongside it that you supply.
See [Helper contract](#helper-contract) below.

---

## Read this before trusting a number

Stage 1 splits the training mix into rows whose prompts also appear in the
evaluation set and rows that do not, and repeats the overlapping ones
`EVAL_UPSAMPLE` times.

That is contamination, and it works exactly as well as you would expect it to.
Raising `EVAL_UPSAMPLE` moves the graded suites up several points and the
held-out suite down by more — the model is memorising graded prompts rather
than learning the task. The default here is `1`, which is no upsampling at all.

Stage 5 therefore always ends on the held-out evaluation, on prompts the model
has never seen, graded the same way. **That is the number that means
something.** If it moves opposite to the two graded suites, the headline was
bought with generalisation. Report both or report neither.

There is a legitimate version of this knob: upsampling *in-distribution* data
that happens to resemble the eval set, with the overlapping rows removed
outright. If that is what you want, filter `scored` out of the mix rather than
multiplying it.

---

## Stages

| Stage | What it does | Guard |
|---|---|---|
| 1 `data` | Build the training mix — overlap split, upsample, general backfill, shuffle | mix is non-empty; format anomalies are counted and printed |
| 2 `train` | BF16 LoRA, backgrounded under `setsid` with a timestamped log | after 60s, the log must show the adapter reached the DeltaNet projections |
| 3 merge | Merge the adapter, diff the result against the cached base snapshot | `adapter APPLIED` or the run stops — no evaluation on a silently-unmerged model |
| 4 serve | Start vLLM, run four advisory probes | server must bind; probes report but never gate |
| 5 `eval` | Multiple-choice suite, generative suite, then held-out | eval roots must be configured |
| 6 `ship` | Move the merged model to its release directory | refuses to overwrite; shard count must match |

Run one at a time or all of them:

```bash
./run_pipeline.sh            # everything
./run_pipeline.sh data       # rebuild the training mix only
./run_pipeline.sh train      # train only
./run_pipeline.sh eval       # merge + verify + serve + evaluate
./run_pipeline.sh ship       # release
```

`eval` deliberately bundles merge and serve — evaluating a model you have not
verified is how a run gets thrown away twice.

---

## Why the guards are where they are

**The DeltaNet check.** On a hybrid-attention model, attaching LoRA to `q_proj`
/ `k_proj` / `v_proj` alone leaves every gated-delta-net block untouched — two
thirds of the layers on this architecture. Training completes, the loss curve
looks fine, and the model has barely moved. The check reads the trainer's own
`[lora]` lines 60 seconds in and kills the run rather than burning three hours.

**The merge verification.** The PEFT merge path on this architecture produces
either the unmodified base weights or garbage, and does not raise. Everything
downstream — serve, evaluate, ship — is worthless if the adapter did not land,
so `verify_merge.py` diffs the merged weights against the cached base snapshot
and the pipeline stops dead unless it reports `adapter APPLIED`.

**BF16 only.** 4-bit quantized training is broken here: the quantization
library dequantizes the gated-delta-net `in_proj_z` projection to the wrong
shape — the right number of values, flattened into a single row — and the
first forward pass fails. It affects two thirds of the layers. BF16 LoRA on a
27B peaks under 70GB on an 80GB card anyway, so 4-bit buys nothing; if you
need headroom, drop `LORA_R` or the sequence length instead.

**GPU release before every stage.** Each stage that needs the card calls
`free_gpu` first. Without it, a vLLM server left over from a previous stage
will OOM the trainer. It only ever kills processes
owned by the current user, so it is safe on a shared box, and it waits for the
card to actually drain rather than assuming the kill took.

---

## Configuration

Everything reads from the environment with a default in the `── config ──`
block at the top of the script. Nothing is hardcoded, and no token is
committed. `.env` next to the script is sourced before training if present.

| Variable | Default | Meaning |
|---|---|---|
| `WORK_DIR` | script directory | Where the data, logs and merged model live |
| `BASE_MODEL` | `unsloth/Qwen3.8-27B` | Base checkpoint |
| `GPU` | `0` | Single GPU index — this pipeline does not shard |
| `RUN_TAG` | `v1` | Names the training file, merged dir, logs and score records |
| `EVAL_UPSAMPLE` | `1` | Repeats of each eval-overlapping row — **see the warning above** |
| `GENERAL_ROWS` | `1500` | Non-overlapping rows mixed back in |
| `MCQ_TASKS` | three placeholders | Tasks graded by exact-match on a letter |
| `LORA_R` / `LORA_ALPHA` | `32` / `32` | Adapter rank and scaling |
| `LEARNING_RATE` / `EPOCHS` / `BATCH_SIZE` | `1e-4` / `1` / `8` | Training |
| `SERVE_PORT` | `8030` | vLLM port; also the port `free_gpu` reclaims |
| `EVAL_HARNESS` / `GEN_BENCH_ROOT` | unset | Required for stage 5 |
| `EVAL_PYTHON` | `$PYTHON` | Eval harnesses usually want their own venv |
| `SHIP_DIR` | `$WORK_DIR/release` | Release directory; never overwritten |
| `EXPECTED_SHARDS` | `18` | Sanity check on the shipped model |

---

## Helper contract

`run_pipeline.sh` calls the following, expected next to it in `$WORK_DIR`. They
are not in this repo — they are specific to your trainer, harness and grader.
Swap in your own and the driver does not change.

| Called | Stage | Expected behaviour |
|---|---|---|
| `train_lora.py` | 2 | Reads `TRAIN_FILE`, `VAL_FILE`, `BASE_MODEL`, the LoRA and training vars; logs `[cfg]`, `[lora]` (naming each adapted module), `[guard] train`, `[done]`; writes the merged model to `MERGED_DIR` |
| `verify_merge.py` | 3 | `verify_merge.py <merged-dir> <base-snapshot>` — prints `adapter APPLIED` when the weights actually differ |
| `serve.sh` | 4 | `MODEL_DIR=… SERVED_NAME=… ./serve.sh <port>` — starts vLLM, exits non-zero if it fails to bind |
| `probe_cot.py` | 4 | Does the model reason when asked? |
| `probe_general.py` | 4 | General-capability spot check, written to a JSON file |
| `probe_capabilities.py` | 4 | Effort ladder, long context, multi-turn |
| `probe_vision.py` | 4 | Vision tower survived the merge |
| `run_mcq_eval.sh` / `record_mcq.py` | 5a | Multiple-choice suite and score record |
| `run_gen_eval.sh` / `record_gen.py` | 5b | Generative suite; `record_gen.py <baseline-run> <candidate-run> <tag>` |
| `eval_heldout.py` | 5c | `eval_heldout.py <served-name> <tag> <effort>`, with `HELDOUT_TRAIN` pointing at the training file so it can exclude anything the model saw |

Stage 1 additionally expects two inputs, both configurable:

- `train_source.jsonl` — the full trace set. Rows carry `text` (ChatML),
  `completion`, `task`, and `id` shaped `<task>_test_<n>` for generative items.
- `eval_item_keys.json` — a JSON list of the graded prompts. Matching is on the
  first 200 normalised characters; full-string matching misses rows that differ
  only in trailing whitespace or option order.

---

## Where this fits

Stage 2 is doing what [`../finetuning`](../finetuning) does, wrapped in guards
and a data step. Stage 6 hands off to
[`../quantization`](../quantization) — quantize the shipped release before
serving it for real. [`../merging`](../merging) slots in between 3 and 4 if the
fine-tune eroded general capability, which the held-out number in 5c will tell
you about.

#!/usr/bin/env bash
# =============================================================================
# End-to-end LoRA pipeline for a 27B dense hybrid-attention checkpoint.
#
#   data  ──►  train  ──►  merge + verify  ──►  serve  ──►  evaluate  ──►  ship
#
# Single GPU, ~4h wall clock on an 80GB card:
#   ~3h train, ~20m merge and verify, ~10m multiple-choice, ~80m generative.
#
#   ./run_pipeline.sh          full pipeline
#   ./run_pipeline.sh data     rebuild the training mix only
#   ./run_pipeline.sh train    train only
#   ./run_pipeline.sh eval     merge + verify + serve + evaluate
#   ./run_pipeline.sh ship     move the merged model to its release directory
#
# ── READ THIS BEFORE TRUSTING A NUMBER ───────────────────────────────────────
# EVAL_UPSAMPLE in stage 1 repeats training rows whose prompts also appear in
# the evaluation set. It is the single biggest lever on the reported score and
# it is contamination: raising it moves the public benchmarks up and the
# held-out benchmark down, because the model is memorising graded prompts
# rather than learning the task. It defaults to 1 (no upsampling) here.
#
# Stage 5 always runs the held-out evaluation last, and that is the number that
# means anything. If held-out drops while the graded suites climb, you bought
# the headline with generalisation. Report both or report neither.
#
# ── Design decisions ─────────────────────────────────────────────────────────
# - BF16 LoRA only. 4-bit quantized training is broken on this architecture:
#   the quantization library dequantizes the gated-delta-net `in_proj_z`
#   projection to the wrong shape — right number of values, flattened into a
#   single row — and the first forward pass fails. BF16 LoRA peaks under 70GB
#   on an 80GB card anyway, so 4-bit buys nothing.
# - The training guard fails the run if the adapter did not reach the DeltaNet
#   projections. Adapting only q/k/v on a hybrid-attention model silently
#   leaves two thirds of the layers untouched.
# - Merge through the Unsloth path. The PEFT merge path on this architecture
#   produces either the unmodified base weights or garbage, without erroring,
#   so `verify_merge.py` is a hard gate — no adapter, no evaluation.
# - The GPU is force-released before every stage that needs it. A leftover vLLM
#   server from a previous stage will otherwise OOM the trainer.
# - Nothing is hardcoded: every path, model id and hyperparameter reads from
#   the environment with a default below, and no token is ever committed.
# =============================================================================
set -uo pipefail

# ── config ───────────────────────────────────────────────────────────────────
# Override with environment variables, or edit the defaults.
WORK_DIR="${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
EVAL_HARNESS="${EVAL_HARNESS:-}"          # required for stage 5 -- eval harness root
GEN_BENCH_ROOT="${GEN_BENCH_ROOT:-}"      # required for stage 5 -- generative suite root
SHIP_DIR="${SHIP_DIR:-$WORK_DIR/release}" # where the finished model lands

PYTHON="${PYTHON:-python}"                # interpreter for the training venv
EVAL_PYTHON="${EVAL_PYTHON:-$PYTHON}"     # the eval harness usually wants its own
export HF_HOME="${HF_HOME:-$WORK_DIR/hf_cache}"

BASE_MODEL="${BASE_MODEL:-unsloth/Qwen3.8-27B}"
HF_DATASET="${HF_DATASET:-}"              # optional -- push of the training mix
HF_TOKEN="${HF_TOKEN:-}"                  # never hardcode a token here

GPU="${GPU:-0}"                           # single GPU by index
SERVE_PORT="${SERVE_PORT:-8030}"
RUN_TAG="${RUN_TAG:-v1}"
SERVED_NAME="${SERVED_NAME:-lora-27b-$RUN_TAG}"
MERGED_DIR="${MERGED_DIR:-$WORK_DIR/merged_$RUN_TAG}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-18}"

# Data mix. See the contamination note in the header before raising EVAL_UPSAMPLE.
EVAL_UPSAMPLE="${EVAL_UPSAMPLE:-1}"       # repeats of each eval-overlapping row
GENERAL_ROWS="${GENERAL_ROWS:-1500}"      # non-overlapping rows mixed back in
GEN_SCORED_PER_SCENARIO="${GEN_SCORED_PER_SCENARIO:-100}"
SEED="${SEED:-3407}"
SOURCE_MIX="${SOURCE_MIX:-$WORK_DIR/train_source.jsonl}"
EVAL_KEYS="${EVAL_KEYS:-$WORK_DIR/eval_item_keys.json}"
TRAIN_FILE="${TRAIN_FILE:-$WORK_DIR/train_$RUN_TAG.jsonl}"
VAL_FILE="${VAL_FILE:-$WORK_DIR/val_clean.jsonl}"
# Tasks graded by exact-match on a letter answer; everything else is scored by
# position within its scenario. Space separated.
MCQ_TASKS="${MCQ_TASKS:-mcq_general mcq_domain_a mcq_domain_b}"

# Training hyperparameters
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"

STAGE="${1:-all}"
cd "$WORK_DIR"

ts(){ date '+%F %T'; }
say(){ echo "[$(ts)] $*"; }
die(){ say "FATAL: $*"; exit 1; }

# ── GPU release ──────────────────────────────────────────────────────────────
# Only ever kills processes owned by the current user, so this is safe on a
# shared box. Waits for the card to actually drain before returning.
free_gpu(){
  local me; me=$(id -un)
  for pid in $(ss -lptnH "sport = :$SERVE_PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u); do
    [ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" = "$me" ] && kill -9 "$pid" 2>/dev/null
  done
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null); do
    [ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" = "$me" ] && kill -9 "$pid" 2>/dev/null
  done
  local used
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
    [ "$used" -lt 4000 ] && break
    sleep 4
  done
  say "GPU$GPU free: ${used} MiB"
}

# =============================================================================
# 1. DATA
# =============================================================================
# Splits the source mix into rows that overlap the evaluation set and rows that
# do not, repeats the overlapping ones EVAL_UPSAMPLE times, then mixes a capped
# sample of the rest back in so the fine-tune does not collapse onto one format.
build_data(){
  say "===== 1. BUILD TRAINING MIX ====="
  [ -f "$SOURCE_MIX" ] || die "$SOURCE_MIX missing (generate traces first)"
  [ -f "$EVAL_KEYS" ]  || die "$EVAL_KEYS missing (the graded prompt list)"
  [ "$EVAL_UPSAMPLE" -gt 1 ] && \
    say "WARNING: EVAL_UPSAMPLE=$EVAL_UPSAMPLE — graded prompts enter training \
${EVAL_UPSAMPLE}x. Benchmark scores after this are contaminated; read the held-out number."

  SOURCE_MIX="$SOURCE_MIX" EVAL_KEYS="$EVAL_KEYS" TRAIN_FILE="$TRAIN_FILE" \
  EVAL_UPSAMPLE="$EVAL_UPSAMPLE" GENERAL_ROWS="$GENERAL_ROWS" \
  GEN_SCORED_PER_SCENARIO="$GEN_SCORED_PER_SCENARIO" \
  MCQ_TASKS="$MCQ_TASKS" SEED="$SEED" BATCH_SIZE="$BATCH_SIZE" \
  "$PYTHON" - <<'PYEOF'
import json, os, random, re

random.seed(int(os.environ["SEED"]))
src        = os.environ["SOURCE_MIX"]
keys_path  = os.environ["EVAL_KEYS"]
out_path   = os.environ["TRAIN_FILE"]
upsample   = int(os.environ["EVAL_UPSAMPLE"])
general_n  = int(os.environ["GENERAL_ROWS"])
scored_n   = int(os.environ["GEN_SCORED_PER_SCENARIO"])
mcq_tasks  = set(os.environ["MCQ_TASKS"].split())
batch_size = int(os.environ["BATCH_SIZE"])

def norm(s):
    return re.sub(r'[^a-z0-9 ]', '', re.sub(r'\s+', ' ', (s or "").lower())).strip()

def user_turn(row):
    m = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', row["text"], re.S)
    return norm(m.group(1)) if m else ""

# Prompts are matched on their first 200 normalised characters. Full-string
# matching misses rows that differ only in trailing whitespace or option order.
eval_prompts = [e[:200] for e in json.load(open(keys_path))]
rows = [json.loads(line) for line in open(src)]

scored, rest = [], []
for row in rows:
    task = row["task"]
    if task in mcq_tasks:
        # Multiple choice: graded if the prompt itself is in the eval set.
        (scored if any(e in user_turn(row) for e in eval_prompts) else rest).append(row)
    else:
        # Generative: only the first N items of each scenario are graded.
        m = re.match(rf'{re.escape(task)}_test_(\d+)$', str(row.get("id", "")))
        (scored if (m and int(m.group(1)) < scored_n) else rest).append(row)

out = [dict(r) for r in scored for _ in range(upsample)]
random.shuffle(rest)
out += rest[:general_n]
random.shuffle(out)

with open(out_path, "w") as f:
    for row in out:
        f.write(json.dumps(row) + "\n")

# A completion that carries neither a letter answer nor a verdict token will
# train the model toward a format the grader cannot parse.
bad = sum(1 for r in out
          if not re.search(r'ANSWER:\s*[A-J]', r["completion"])
          and "CORRECT" not in r["completion"])

print(f"  overlapping={len(scored)} x{upsample} + {min(general_n, len(rest))} general "
      f"= {len(out)} rows")
print(f"  label/format anomalies: {bad}")
print(f"  est steps @batch{batch_size}, 1 epoch: {len(out) // batch_size}")
PYEOF
  [ -s "$TRAIN_FILE" ] || die "$TRAIN_FILE not written"
}

# =============================================================================
# 2. TRAIN
# =============================================================================
do_train(){
  say "===== 2. TRAIN ====="
  free_gpu
  local log="$WORK_DIR/train_$(date +%Y%m%d_%H%M%S).log"
  set -a; [ -f "$WORK_DIR/.env" ] && . "$WORK_DIR/.env"; set +a

  RUN_TAG="$RUN_TAG" TRAIN_FILE="$TRAIN_FILE" VAL_FILE="$VAL_FILE" \
  BASE_MODEL="$BASE_MODEL" MERGED_DIR="$MERGED_DIR" \
  LORA_R="$LORA_R" LORA_ALPHA="$LORA_ALPHA" LEARNING_RATE="$LEARNING_RATE" \
  EPOCHS="$EPOCHS" BATCH_SIZE="$BATCH_SIZE" \
  HF_DATASET="$HF_DATASET" HF_TOKEN="$HF_TOKEN" HF_HOME="$HF_HOME" \
  CUDA_VISIBLE_DEVICES="$GPU" \
  setsid "$PYTHON" -u train_lora.py > "$log" 2>&1 &
  local pid=$!
  say "trainer pid $pid -> $log"

  # Fail fast rather than 3h later: the adapter must reach the gated-delta-net
  # projections, not just the attention q/k/v.
  sleep 60
  grep -qE '^\[lora\].*DeltaNet' "$log" \
    || die "LoRA did not adapt DeltaNet layers — check target_modules"
  grep -E '^\[cfg\]|^\[guard\] train|^\[lora\]' "$log" | sed 's/^/    /'

  wait $pid
  say "training complete"
  grep -E "train_runtime|\[done\]" "$log" | tail -2 | sed 's/^/    /'
}

# =============================================================================
# 3. MERGE + VERIFY
# =============================================================================
do_merge(){
  say "===== 3. MERGE + VERIFY ====="
  [ -d "$MERGED_DIR" ] || die "trainer did not write $MERGED_DIR"
  local snapshot
  snapshot=$(ls -d "$HF_HOME"/hub/models--"${BASE_MODEL//\//--}"/snapshots/*/ 2>/dev/null | head -1)
  [ -n "$snapshot" ] || die "no cached snapshot for $BASE_MODEL under $HF_HOME"

  "$PYTHON" verify_merge.py "$MERGED_DIR" "$snapshot" | tee /tmp/verify_$RUN_TAG.txt | tail -3
  grep -q "adapter APPLIED" "/tmp/verify_$RUN_TAG.txt" \
    || die "adapter NOT applied — refusing to evaluate"
  say "merge verified"
}

# =============================================================================
# 4. SERVE + SANITY
# =============================================================================
# Probes are advisory — they report, they do not gate. A failing probe is worth
# reading before you trust the scores, but it should not throw away a 3h run.
do_serve(){
  say "===== 4. SERVE + SANITY ====="
  free_gpu
  MODEL_DIR="$MERGED_DIR" SERVED_NAME="$SERVED_NAME" ./serve.sh "$SERVE_PORT" \
    || die "vLLM failed to start on port $SERVE_PORT"

  "$PYTHON" probe_cot.py          "$SERVED_NAME" || true   # reasons when asked?
  "$PYTHON" probe_general.py      "$SERVED_NAME" "general_$RUN_TAG.json" || true
  "$PYTHON" probe_capabilities.py "$SERVED_NAME" || true   # effort ladder, long ctx, multi-turn
  "$PYTHON" probe_vision.py       "$SERVED_NAME" || true   # vision tower still intact?
}

# =============================================================================
# 5. EVALUATE
# =============================================================================
do_eval(){
  [ -n "$EVAL_HARNESS" ]   || die "set EVAL_HARNESS to the eval harness root"
  [ -n "$GEN_BENCH_ROOT" ] || die "set GEN_BENCH_ROOT to the generative suite root"

  say "===== 5a. MULTIPLE CHOICE ====="
  MODEL="openai/$SERVED_NAME" TAG="$RUN_TAG" EFFORT=medium ./run_mcq_eval.sh
  "$EVAL_PYTHON" record_mcq.py "$RUN_TAG"

  say "===== 5b. GENERATIVE ====="
  ./run_gen_eval.sh
  "$EVAL_PYTHON" record_gen.py "baseline_n$GEN_SCORED_PER_SCENARIO" \
                               "${RUN_TAG}_medium_n$GEN_SCORED_PER_SCENARIO" "$RUN_TAG"

  # Last on purpose. Prompts the model has never seen, graded the same way.
  # If this moves opposite to the two suites above, the gain was memorisation.
  say "===== 5c. HELD-OUT (the honest number) ====="
  HELDOUT_TRAIN="$TRAIN_FILE" "$PYTHON" eval_heldout.py "$SERVED_NAME" "$RUN_TAG" medium
}

# =============================================================================
# 6. SHIP
# =============================================================================
do_ship(){
  say "===== 6. SHIP ====="
  free_gpu
  [ -d "$SHIP_DIR" ] && { say "$SHIP_DIR exists — not overwriting"; return; }
  mv "$MERGED_DIR" "$SHIP_DIR" || die "could not move $MERGED_DIR"
  say "shipped -> $SHIP_DIR ($(du -sh "$SHIP_DIR" | cut -f1))"
  local shards; shards=$(ls "$SHIP_DIR"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
  [ "$shards" = "$EXPECTED_SHARDS" ] \
    || die "expected $EXPECTED_SHARDS shards, found $shards"
}

case "$STAGE" in
  data)  build_data ;;
  train) do_train ;;
  eval)  do_merge; do_serve; do_eval ;;
  ship)  do_ship ;;
  all)   build_data; do_train; do_merge; do_serve; do_eval; do_ship ;;
  *)     die "unknown stage: $STAGE (data|train|eval|ship|all)" ;;
esac
say "DONE ($STAGE)"

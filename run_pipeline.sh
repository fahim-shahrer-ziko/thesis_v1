#!/usr/bin/env bash
#
# run_pipeline.sh
#
# Runs the full MRT instruction-tuning pipeline end-to-end, in order:
#   0. (skip if dataset/ already has real data) generate synthetic sample data
#   1. check_data.py            -- validate dataset/*.csv against pipeline assumptions
#   2. generate_instructions.py -- self-instruct: seed task -> OpenAI -> 10 instructions/task
#   3. build_dataset.py         -- CSV + instructions -> per-task train/val/test JSON
#   4. train_lora.py            -- QLoRA fine-tune, one adapter per task (3 models)
#   5. infer.py                 -- run each task's adapter against its test set
#   6. evaluate.py              -- accuracy / tolerance metrics per task
#
# Stops immediately on any failure (set -e). Each stage prints a clear
# header so you can see exactly where you are / where it stopped.
#
# Usage:
#   ./run_pipeline.sh                 # full real run (needs OPENAI_API_KEY + GPU)
#   ./run_pipeline.sh --use-sample-data   # use synthetic data instead of dataset/*.csv
#   ./run_pipeline.sh --skip-generate     # reuse existing utils/instruction/*.json (skip OpenAI calls)
#   ./run_pipeline.sh --skip-train        # skip train_lora.py (e.g. you already trained, just want to re-eval)
#   ./run_pipeline.sh --task dest         # only run the destination task for build/train/infer/eval
#
set -e

USE_SAMPLE_DATA=0
SKIP_GENERATE=0
SKIP_TRAIN=0
TASK="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --use-sample-data) USE_SAMPLE_DATA=1; shift ;;
    --skip-generate) SKIP_GENERATE=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --task) TASK="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

step() {
  echo ""
  echo "============================================================"
  echo "  $1"
  echo "============================================================"
}

if [[ "$TASK" == "all" ]]; then
  TASKS=(ori dest entry_time)
else
  TASKS=("$TASK")
fi

# ---------------------------------------------------------------------------
# Step 0: sample data (optional)
# ---------------------------------------------------------------------------
if [[ "$USE_SAMPLE_DATA" == "1" ]]; then
  step "STEP 0: Generating synthetic sample dataset"
  python3 scripts/generate_sample_data.py
else
  step "STEP 0: Using existing dataset/trip_data.csv + dataset/persona_data.csv"
  if [[ ! -f dataset/trip_data.csv || ! -f dataset/persona_data.csv ]]; then
    echo "ERROR: dataset/trip_data.csv and/or dataset/persona_data.csv not found."
    echo "Either place your real CSVs there, or re-run with --use-sample-data to test with synthetic data."
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Step 1: validate the CSVs
# ---------------------------------------------------------------------------
step "STEP 1: Validating dataset CSVs (check_data.py)"
python3 check_data.py

# ---------------------------------------------------------------------------
# Step 2: generate instructions (self-instruct via OpenAI)
# ---------------------------------------------------------------------------
if [[ "$SKIP_GENERATE" == "1" ]]; then
  step "STEP 2: SKIPPED (--skip-generate) -- reusing existing utils/instruction/*.json"
else
  step "STEP 2: Generating 10 diverse instructions per task (generate_instructions.py)"
  if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "Run: export OPENAI_API_KEY=sk-..."
    echo "Or re-run with --skip-generate if utils/instruction/{task}_instructions.json already exist."
    exit 1
  fi
  for t in "${TASKS[@]}"; do
    python3 generate_instructions.py --task "$t"
  done
fi

# ---------------------------------------------------------------------------
# Step 3: build the instruction-tuning dataset
# ---------------------------------------------------------------------------
step "STEP 3: Building instruction-tuning dataset (build_dataset.py)"
python3 build_dataset.py --task "$TASK"

# ---------------------------------------------------------------------------
# Step 4: train (QLoRA, one adapter per task)
# ---------------------------------------------------------------------------
if [[ "$SKIP_TRAIN" == "1" ]]; then
  step "STEP 4: SKIPPED (--skip-train)"
else
  step "STEP 4: Fine-tuning with QLoRA (train_lora_{ori,dest,entry_time}.py) -- task(s): ${TASKS[*]}"
  for t in "${TASKS[@]}"; do
    python3 "train_lora_${t}.py"
  done
fi

# ---------------------------------------------------------------------------
# Step 5: inference on each task's test set
# ---------------------------------------------------------------------------
step "STEP 5: Running inference (infer.py) -- task(s): ${TASKS[*]}"
for t in "${TASKS[@]}"; do
  python3 infer.py --task "$t" --use_4bit
done

# ---------------------------------------------------------------------------
# Step 6: evaluate
# ---------------------------------------------------------------------------
step "STEP 6: Evaluating predictions (evaluate.py) -- task(s): ${TASKS[*]}"
for t in "${TASKS[@]}"; do
  echo ""
  echo "--- Evaluating task: $t ---"
  python3 evaluate.py --predictions_csv "output/${t}_predictions.csv" --task "$t"
done

step "PIPELINE COMPLETE"
echo "Adapters:    modelSave/llama3_1_mrt_qlora_{task}/final_adapter/"
echo "Predictions: output/{task}_predictions.csv"

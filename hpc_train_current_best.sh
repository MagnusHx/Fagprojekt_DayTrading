#!/bin/bash
#BSUB -J kvant_train_best
#BSUB -q hpc
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[avx512]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -oo logs/kvant_train_best_%J.out
#BSUB -eo logs/kvant_train_best_%J.err
#BSUB -env "LSB_JOB_REPORT_MAIL=N"

set -euo pipefail

PROJECT_DIR="${LS_SUBCWD:-$HOME/Fagprojekt_DayTrading}"
PREPARED_ROOT="src/kvant/ml_framework/prepared"

LOOKBACK="${LOOKBACK:-12}"
BARRIER_WIDTH="${BARRIER_WIDTH:-60}"
BARRIER_HEIGHT_PCT="${BARRIER_HEIGHT_PCT:-0.75}"
TARGET_BARS_PER_DAY="${TARGET_BARS_PER_DAY:-20}"
MAX_FOLDS="${MAX_FOLDS:-1}"
EPOCHS="${EPOCHS:-30}"
SEED="${SEED:-1337}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
NO_RETURN_STATS="${NO_RETURN_STATS:-1}"
DELETE_PREPARED_AFTER="${DELETE_PREPARED_AFTER:-1}"
META_FEATURES="${META_FEATURES:-proba,embedding,prediction_margin,prediction_entropy,time_since_last_event}"

cd "$PROJECT_DIR"
mkdir -p logs artifacts/train_current_best "$PREPARED_ROOT"

module load python3/3.12.11
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline
export OMP_NUM_THREADS="${LSB_DJOB_NUMPROC:-1}"
export MKL_NUM_THREADS="${LSB_DJOB_NUMPROC:-1}"
export OPENBLAS_NUM_THREADS="${LSB_DJOB_NUMPROC:-1}"
export NUMEXPR_NUM_THREADS="${LSB_DJOB_NUMPROC:-1}"
export MPLBACKEND=Agg

run_id="w${BARRIER_WIDTH}_h${BARRIER_HEIGHT_PCT}_tbpd${TARGET_BARS_PER_DAY}_folds${MAX_FOLDS}_ep${EPOCHS}"
run_dir="artifacts/train_current_best/$run_id"
mkdir -p "$run_dir"

echo "host=$(hostname)"
echo "started_at=$(date --iso-8601=seconds)"
echo "project_dir=$PWD"
echo "run_id=$run_id"
echo "seed=$SEED"
python3 --version
uv --version

uv sync --frozen --python "$(command -v python3)"

cleanup_prepared_from_manifest() {
  local manifest_path="$1"
  uv run python - "$manifest_path" "$PREPARED_ROOT" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

manifest_path = Path(sys.argv[1]).resolve()
prepared_root = Path(sys.argv[2]).resolve()
payload = json.loads(manifest_path.read_text())

deleted = []
for fold in payload.get("folds", []):
    exp_dir = Path(fold["exp_dir"]).resolve()
    if exp_dir == prepared_root or prepared_root not in exp_dir.parents:
        raise SystemExit(f"Refusing to delete path outside prepared root: {exp_dir}")
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
        deleted.append(str(exp_dir))

if manifest_path.exists():
    if manifest_path == prepared_root or prepared_root not in manifest_path.parents:
        raise SystemExit(f"Refusing to delete manifest outside prepared root: {manifest_path}")
    manifest_path.unlink()
    deleted.append(str(manifest_path))

for name in ("last_experiment.txt", "last_experiment_cv_manifest.txt"):
    pointer = prepared_root / name
    if pointer.exists():
        pointer.unlink()
        deleted.append(str(pointer))

print("Deleted prepared artifacts:")
for path in deleted:
    print(f"  {path}")
PY
}

snapshot_wandb_runs() {
  local out_path="$1"
  find wandb -maxdepth 1 -type d -name 'offline-run-*' -print 2>/dev/null | sort > "$out_path" || true
}

sync_new_wandb_runs() {
  local before_path="$1"
  local after_path="$2"
  local new_runs_path="$3"

  snapshot_wandb_runs "$after_path"
  comm -13 "$before_path" "$after_path" > "$new_runs_path" || true
  if [ ! -s "$new_runs_path" ]; then
    echo "No new W&B offline run directories found."
    return 0
  fi

  while IFS= read -r wandb_run_dir; do
    echo "Syncing W&B run: $wandb_run_dir"
    uv run wandb sync "$wandb_run_dir" || echo "WARNING: W&B sync failed for $wandb_run_dir; leaving offline run on disk."
  done < "$new_runs_path"
}

rm -f "$PREPARED_ROOT/last_experiment.txt" "$PREPARED_ROOT/last_experiment_cv_manifest.txt"

prepare_args=(
  -m kvant.ml_prepare_data.prepare_experiment
  --lookback "$LOOKBACK"
  --barrier-width "$BARRIER_WIDTH"
  --barrier-height-pct "$BARRIER_HEIGHT_PCT"
  --target-bars-per-day "$TARGET_BARS_PER_DAY"
  --overwrite-existing
)
if [ "$MAX_FOLDS" != "all" ]; then
  prepare_args+=(--max-folds "$MAX_FOLDS")
fi

uv run python "${prepare_args[@]}"

manifest_path="$(tr -d '\n' < "$PREPARED_ROOT/last_experiment_cv_manifest.txt")"
if [ ! -f "$manifest_path" ]; then
  echo "FAILED: expected manifest was not created: $manifest_path" >&2
  exit 1
fi
echo "$manifest_path" > "$run_dir/manifest_path.txt"
du -sh "$PREPARED_ROOT" | tee "$run_dir/prepared_size_before_training.txt"

uv run python -m kvant.ml_framework.scripts.smoke_prepared_experiment \
  --cv-manifest "$manifest_path" \
  --require-market-data | tee "$run_dir/preflight.json"

before_runs="$run_dir/wandb_before.txt"
after_runs="$run_dir/wandb_after.txt"
new_runs="$run_dir/wandb_new_runs.txt"
snapshot_wandb_runs "$before_runs"

train_args=(
  -m kvant.ml_framework.scripts.train_experiment
  --cv-manifest "$manifest_path"
  --baseline
  --epochs "$EPOCHS"
  --seed "$SEED"
  --full-eval-every "$EPOCHS"
  --train-batch-size "$TRAIN_BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --meta-features "$META_FEATURES"
  --checkpoint-out-dir "artifacts/checkpoints/$run_id"
  --wandb-name "train-current-best-${run_id}"
)
if [ "$NO_RETURN_STATS" = "1" ]; then
  train_args+=(--no-return-stats)
fi

train_status=0
uv run python "${train_args[@]}" || train_status=$?

sync_new_wandb_runs "$before_runs" "$after_runs" "$new_runs"
if [ "$DELETE_PREPARED_AFTER" = "1" ]; then
  cleanup_prepared_from_manifest "$manifest_path"
fi
du -sh "$PREPARED_ROOT" || true

if [ "$train_status" -ne 0 ]; then
  echo "FAILED: train_experiment failed with exit status $train_status" | tee "$run_dir/status.txt"
  exit "$train_status"
fi

echo "OK" > "$run_dir/status.txt"
echo "finished_at=$(date --iso-8601=seconds)"

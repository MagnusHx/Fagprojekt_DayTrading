#!/bin/bash
#BSUB -J kvant_label_screen
#BSUB -q hpc
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "select[avx512]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -oo logs/kvant_label_screen_%J.out
#BSUB -eo logs/kvant_label_screen_%J.err
#BSUB -env "LSB_JOB_REPORT_MAIL=N"

set -euo pipefail

PROJECT_DIR="${LS_SUBCWD:-$HOME/Fagprojekt_DayTrading}"
PREPARED_ROOT="src/kvant/ml_framework/prepared"

# Fast screening defaults. MAX_CONFIGS=0 runs the full 27-config grid.
MAX_CONFIGS="${MAX_CONFIGS:-1}"
MAX_FOLDS="${MAX_FOLDS:-1}"
EPOCHS="${EPOCHS:-5}"
LOOKBACK="${LOOKBACK:-12}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
NO_RETURN_STATS="${NO_RETURN_STATS:-1}"
META_FEATURES="${META_FEATURES:-proba,embedding,prediction_margin,prediction_entropy,time_since_last_event}"
SEED="${SEED:-1337}"

WIDTHS=(60 120 180)
HEIGHTS=(0.75 1.0 1.5)
TARGET_BARS_PER_DAY=(20 30 40)

cd "$PROJECT_DIR"
mkdir -p logs artifacts/screen_runs "$PREPARED_ROOT"

module load python3/3.12.11
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline
export OMP_NUM_THREADS="${LSB_DJOB_NUMPROC:-1}"
export MKL_NUM_THREADS="${LSB_DJOB_NUMPROC:-1}"
export OPENBLAS_NUM_THREADS="${LSB_DJOB_NUMPROC:-1}"
export NUMEXPR_NUM_THREADS="${LSB_DJOB_NUMPROC:-1}"
export MPLBACKEND=Agg

echo "host=$(hostname)"
echo "started_at=$(date --iso-8601=seconds)"
echo "project_dir=$PWD"
echo "max_configs=$MAX_CONFIGS max_folds=$MAX_FOLDS epochs=$EPOCHS seed=$SEED meta_features=$META_FEATURES"
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

write_status() {
  local status_run_dir="$1"
  local status_message="$2"

  if [ -z "$status_run_dir" ]; then
    status_run_dir="artifacts/screen_runs/unknown_status"
  fi
  mkdir -p "$status_run_dir"
  printf '%s\n' "$status_message" > "$status_run_dir/status.txt"
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

  while IFS= read -r run_dir; do
    echo "Syncing W&B run: $run_dir"
    uv run wandb sync "$run_dir" || echo "WARNING: W&B sync failed for $run_dir; leaving offline run on disk."
  done < "$new_runs_path"
}

config_count=0
for width in "${WIDTHS[@]}"; do
  for height in "${HEIGHTS[@]}"; do
    for tbpd in "${TARGET_BARS_PER_DAY[@]}"; do
      config_count=$((config_count + 1))
      if [ "$MAX_CONFIGS" -gt 0 ] && [ "$config_count" -gt "$MAX_CONFIGS" ]; then
        echo "Reached MAX_CONFIGS=$MAX_CONFIGS. Stopping screen."
        echo "finished_at=$(date --iso-8601=seconds)"
        exit 0
      fi

      run_id="w${width}_h${height}_tbpd${tbpd}_folds${MAX_FOLDS}_ep${EPOCHS}"
      run_dir="artifacts/screen_runs/$run_id"
      mkdir -p "$run_dir"
      echo "=== screen config ${config_count}: ${run_id} ==="
      echo "config_started_at=$(date --iso-8601=seconds)"
      du -sh "$PREPARED_ROOT" || true
      rm -f "$PREPARED_ROOT/last_experiment.txt" "$PREPARED_ROOT/last_experiment_cv_manifest.txt"

      if ! uv run python -m kvant.ml_prepare_data.prepare_experiment \
        --lookback "$LOOKBACK" \
        --barrier-width "$width" \
        --barrier-height-pct "$height" \
        --target-bars-per-day "$tbpd" \
        --max-folds "$MAX_FOLDS" \
        --overwrite-existing; then
        echo "FAILED: prepare_experiment failed for $run_id"
        write_status "$run_dir" "FAILED: prepare_experiment failed for $run_id"
        continue
      fi

      manifest_path="$(tr -d '\n' < "$PREPARED_ROOT/last_experiment_cv_manifest.txt")"
      if [ ! -f "$manifest_path" ]; then
        echo "FAILED: expected manifest was not created: $manifest_path" >&2
        write_status "$run_dir" "FAILED: expected manifest was not created: $manifest_path"
        continue
      fi
      echo "$manifest_path" > "$run_dir/manifest_path.txt"
      du -sh "$PREPARED_ROOT" | tee "$run_dir/prepared_size_before_training.txt"

      if ! uv run python -m kvant.ml_framework.scripts.smoke_prepared_experiment \
        --cv-manifest "$manifest_path" \
        --require-market-data | tee "$run_dir/preflight.json"; then
        echo "FAILED: smoke_prepared_experiment failed for $run_id"
        write_status "$run_dir" "FAILED: smoke_prepared_experiment failed for $run_id"
        cleanup_prepared_from_manifest "$manifest_path"
        continue
      fi

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
        --wandb-name "label-screen-${run_id}"
      )
      if [ "$NO_RETURN_STATS" = "1" ]; then
        train_args+=(--no-return-stats)
      fi

      train_status=0
      uv run python "${train_args[@]}" || train_status=$?

      sync_new_wandb_runs "$before_runs" "$after_runs" "$new_runs"
      cleanup_prepared_from_manifest "$manifest_path"
      du -sh "$PREPARED_ROOT" || true
      if [ "$train_status" -ne 0 ]; then
        echo "FAILED: train_experiment failed for $run_id with exit status $train_status"
        write_status "$run_dir" "FAILED: train_experiment failed for $run_id with exit status $train_status"
        continue
      fi
      write_status "$run_dir" "OK"
      echo "config_finished_at=$(date --iso-8601=seconds)"
    done
  done
done

echo "finished_at=$(date --iso-8601=seconds)"

#!/bin/bash
#BSUB -J kvant_label_sweep
#BSUB -q hpc
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 9GB
#BSUB -oo logs/kvant_label_sweep_%J.out
#BSUB -eo logs/kvant_label_sweep_%J.err
#BSUB -env "LSB_JOB_REPORT_MAIL=N"

set -euo pipefail

PROJECT_DIR="${LS_SUBCWD:-$HOME/Fagprojekt_DayTrading}"
PREPARED_ROOT="src/kvant/ml_framework/prepared"

# Safety defaults:
#   MAX_CONFIGS=1 runs only the first config for measuring time/space.
#   MAX_CONFIGS=0 runs the full grid below.
MAX_CONFIGS="${MAX_CONFIGS:-1}"
EPOCHS="${EPOCHS:-3}"
LOOKBACK="${LOOKBACK:-12}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
NO_RETURN_STATS="${NO_RETURN_STATS:-1}"

WIDTHS=(60 120 180)
HEIGHTS=(0.75 1.0 1.5)
TARGET_BARS_PER_DAY=(20 30 40)

cd "$PROJECT_DIR"
mkdir -p logs artifacts/sweep_runs "$PREPARED_ROOT"

module load python3/3.12.11
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline
export OMP_NUM_THREADS="${LSB_DJOB_NUMPROC:-1}"
export MPLBACKEND=Agg

echo "host=$(hostname)"
echo "started_at=$(date --iso-8601=seconds)"
echo "project_dir=$PWD"
echo "max_configs=$MAX_CONFIGS epochs=$EPOCHS lookback=$LOOKBACK no_return_stats=$NO_RETURN_STATS"
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
deleted: list[str] = []
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

for pointer_name in ("last_experiment.txt", "last_experiment_cv_manifest.txt"):
    pointer = prepared_root / pointer_name
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
    return 1
  fi

  while IFS= read -r run_dir; do
    echo "Syncing W&B run: $run_dir"
    uv run wandb sync "$run_dir"
  done < "$new_runs_path"
}

config_count=0
for width in "${WIDTHS[@]}"; do
  for height in "${HEIGHTS[@]}"; do
    for tbpd in "${TARGET_BARS_PER_DAY[@]}"; do
      config_count=$((config_count + 1))
      if [ "$MAX_CONFIGS" -gt 0 ] && [ "$config_count" -gt "$MAX_CONFIGS" ]; then
        echo "Reached MAX_CONFIGS=$MAX_CONFIGS. Stopping sweep."
        echo "finished_at=$(date --iso-8601=seconds)"
        exit 0
      fi

      run_id="w${width}_h${height}_tbpd${tbpd}"
      run_dir="artifacts/sweep_runs/${run_id}"
      mkdir -p "$run_dir"
      echo "=== config ${config_count}: ${run_id} ==="
      echo "config_started_at=$(date --iso-8601=seconds)"
      du -sh "$PREPARED_ROOT" || true

      uv run python -m kvant.ml_prepare_data.prepare_experiment \
        --lookback "$LOOKBACK" \
        --barrier-width "$width" \
        --barrier-height-pct "$height" \
        --target-bars-per-day "$tbpd"

      manifest_path="$(tr -d '\n' < "$PREPARED_ROOT/last_experiment_cv_manifest.txt")"
      if [ ! -f "$manifest_path" ]; then
        echo "Expected manifest was not created: $manifest_path" >&2
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
        --train-batch-size "$TRAIN_BATCH_SIZE"
        --eval-batch-size "$EVAL_BATCH_SIZE"
        --wandb-name "label-sweep-${run_id}-ep${EPOCHS}"
      )
      if [ "$NO_RETURN_STATS" = "1" ]; then
        train_args+=(--no-return-stats)
      fi

      uv run python "${train_args[@]}"

      sync_new_wandb_runs "$before_runs" "$after_runs" "$new_runs"
      cleanup_prepared_from_manifest "$manifest_path"
      du -sh "$PREPARED_ROOT" || true
      echo "config_finished_at=$(date --iso-8601=seconds)"
    done
  done
done

echo "finished_at=$(date --iso-8601=seconds)"

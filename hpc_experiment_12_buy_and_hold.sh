#!/bin/bash
#BSUB -J kvant_e0_buyhold
#BSUB -q hpc
#BSUB -n 4
#BSUB -W 02:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -oo logs/kvant_e0_buyhold_%J.out
#BSUB -eo logs/kvant_e0_buyhold_%J.err
#BSUB -env "LSB_JOB_REPORT_MAIL=N"

set -euo pipefail

PROJECT_DIR="${LS_SUBCWD:-$HOME/Fagprojekt_DayTrading}"
SELECTED_GRID_ENV="artifacts/final_plan/selected_grid.env"
BEST_MANIFEST="${BEST_MANIFEST:-src/kvant/ml_framework/prepared/sb_L_12_w240_h2_fixedCUSUM0.01_cv_manifest.json}"
RESULTS_OUT="results/baselines/E0_buy_and_hold.csv"

WANDB_PROJECT="${WANDB_PROJECT:-day-trading-experiments}"
WANDB_MODE="${WANDB_MODE:-offline}"

on_exit() {
  status=$?
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "exit_status=$status"
}
trap on_exit EXIT

cd "$PROJECT_DIR"
mkdir -p logs results/baselines wandb

if [ -f "$SELECTED_GRID_ENV" ]; then
  source "$SELECTED_GRID_ENV"
fi

module load python3/3.12.11

export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT
export WANDB_MODE
export WANDB_DIR="$PROJECT_DIR/wandb"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${LSB_DJOB_NUMPROC:-1}"
export MKL_NUM_THREADS="${LSB_DJOB_NUMPROC:-1}"
export OPENBLAS_NUM_THREADS="${LSB_DJOB_NUMPROC:-1}"
export NUMEXPR_NUM_THREADS="${LSB_DJOB_NUMPROC:-1}"

echo "host=$(hostname)"
echo "started_at=$(date --iso-8601=seconds)"
echo "project_dir=$PWD"
echo "wandb_mode=$WANDB_MODE"
echo "best_manifest=$BEST_MANIFEST"
echo "triple_barrier_height=0.02"
echo "cusum_threshold=0.01"
python3 --version
uv --version

if [ ! -f "$BEST_MANIFEST" ]; then
  echo "Expected selected CV manifest does not exist: $BEST_MANIFEST" >&2
  echo "Prepare the 2% triple-barrier / 1% CUSUM config before submitting this job." >&2
  exit 1
fi

uv sync --frozen --python "$(command -v python3)"

uv run python scripts/buy_and_hold_baseline.py \
  --cv-manifest "$BEST_MANIFEST" \
  --transaction-cost 0 \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-name E0-buy-and-hold \
  --output "$RESULTS_OUT"

echo "results_out=$RESULTS_OUT"

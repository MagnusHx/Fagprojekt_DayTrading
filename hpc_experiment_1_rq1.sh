#!/bin/bash
#BSUB -J kvant_e1_rq1
#BSUB -q hpc
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 9GB
#BSUB -oo logs/kvant_e1_rq1_%J.out
#BSUB -eo logs/kvant_e1_rq1_%J.err
#BSUB -env "LSB_JOB_REPORT_MAIL=N"

set -euo pipefail

PROJECT_DIR="${LS_SUBCWD:-$HOME/Fagprojekt_DayTrading}"

WANDB_PROJECT="${WANDB_PROJECT:-day-trading-experiments}"
WANDB_ENTITY="${WANDB_ENTITY:-s245509-danmarks-tekniske-universitet-dtu}"
WANDB_MODE="${WANDB_MODE:-offline}"

CV_MANIFEST="src/kvant/ml_framework/prepared/E1_timebar_cv_manifest.json"
CHECKPOINT_DIR="artifacts/E1_timebar_conv1d_nometa"
RESULTS_OUT="results/main/E1_timebar_conv1d_nometa.csv"

on_exit() {
  status=$?
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "exit_status=$status"
}
trap on_exit EXIT

cd "$PROJECT_DIR"
mkdir -p logs artifacts results/main src/kvant/ml_framework/prepared wandb

module load python3/3.12.11

export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT
export WANDB_ENTITY
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
python3 --version
uv --version

uv sync --frozen --python "$(command -v python3)"

uv run python -m kvant.ml_prepare_data.prepare_experiment \
  --sampler time_bar \
  --time-bar-minutes 15 \
  --labeler next_bar \
  --cv-manifest "$CV_MANIFEST"

uv run python -m kvant.ml_framework.scripts.smoke_prepared_experiment \
  --cv-manifest "$CV_MANIFEST" \
  --require-market-data

uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest "$CV_MANIFEST" \
  --model conv1d \
  --epochs 20 \
  --seed 1337 \
  --checkpoint-out-dir "$CHECKPOINT_DIR" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-name E1-timebar-conv1d-nometa \
  --transaction-cost 0.001 \
  --bet-sizing fixed \
  --no-meta \
  --fixed-bet-size 1.0 \
  --results-out "$RESULTS_OUT"

echo "manifest=$CV_MANIFEST"
echo "results_out=$RESULTS_OUT"
echo "checkpoint_dir=$CHECKPOINT_DIR"

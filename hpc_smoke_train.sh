#!/bin/bash
#BSUB -J kvant_smoke
#BSUB -q hpc
#BSUB -n 4
#BSUB -W 00:30
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -oo logs/kvant_smoke_%J.out
#BSUB -eo logs/kvant_smoke_%J.err
#BSUB -env "LSB_JOB_REPORT_MAIL=N"

set -euo pipefail

PROJECT_DIR="${LS_SUBCWD:-$HOME/Fagprojekt_DayTrading}"
cd "$PROJECT_DIR"
mkdir -p logs

module load python3/3.12.11
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline
export OMP_NUM_THREADS="${LSB_DJOB_NUMPROC:-1}"

echo "host=$(hostname)"
echo "started_at=$(date --iso-8601=seconds)"
echo "project_dir=$PWD"
python3 --version
uv --version

uv sync --frozen --python "$(command -v python3)"

uv run python -m kvant.ml_framework.scripts.create_smoke_prepared_experiment --overwrite

uv run python -m kvant.ml_framework.scripts.smoke_prepared_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/smoke_one_fold_cv_manifest.json

uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/smoke_one_fold_cv_manifest.json \
  --baseline \
  --epochs 1 \
  --train-batch-size 32 \
  --eval-batch-size 32 \
  --no-return-stats \
  --no-save-best-checkpoint \
  --wandb-name smoke-hpc-cv-test

echo "finished_at=$(date --iso-8601=seconds)"

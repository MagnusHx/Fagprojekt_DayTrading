# Experiment Distribution For 4 People

This document assigns the experiment workflow across four people. The CUSUM/TB grid is run by one person to avoid
missing CSV files, duplicated W&B runs, and mismatched local artifacts. The other people start their follow-up runs only
after the best grid configuration has been selected and shared.

## Shared Setup

Everyone starts from the repository root:

```bash
git pull origin experiment-framework
uv sync

export WANDB_PROJECT=day-trading-experiments
export WANDB_ENTITY=s245509-danmarks-tekniske-universitet-dtu

uv run python -m pytest tests/test_experiment_grid.py
```

For long runs on macOS, keep `caffeinate -dims` in front of the command.

If W&B runs in offline mode, sync after finishing:

```bash
wandb sync wandb/offline-run-*
```

Important: `results/` is ignored by Git. Generated CSV files must be shared manually with the person doing final
aggregation.

Protocol note: all main comparison runs use fixed bet size `1.0`, including meta-selection. Kelly sizing is reserved for
a later explicit sizing ablation, so the timebar-vs-CUSUM and no-meta-vs-meta comparisons are not affected by
position-size differences.

Metric protocol: the no-meta timebar-vs-CUSUM runs compare primary-model predictive quality only. Use macro F1,
accuracy, confusion matrices, and true-vs-predicted class distributions for that comparison. Do not use no-meta
portfolio metrics as the main conclusion. Portfolio and decision metrics become primary again for confidence-threshold
and meta-selection runs, where the experiment explicitly tests trade filtering.

## Phase 1: Person 1 Runs The Full Grid

Person 1 owns the full grid workflow from data preparation to best-config selection.

### 1. Prepare And Train 15-Minute Time-Bar Baseline

```bash
uv run python -m kvant.ml_prepare_data.prepare_experiment \
  --sampler time_bar --time-bar-minutes 15 \
  --labeler next_bar \
  --cv-manifest src/kvant/ml_framework/prepared/E1_timebar_cv_manifest.json

caffeinate -dims uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/E1_timebar_cv_manifest.json \
  --model conv1d --epochs 20 --seed 1337 \
  --checkpoint-out-dir artifacts/E1_timebar_conv1d_nometa \
  --wandb-project day-trading-experiments \
  --wandb-name E1-timebar-conv1d-nometa \
  --transaction-cost 0 \
  --bet-sizing fixed \
  --no-meta \
  --fixed-bet-size 1.0 \
  --results-out results/main/E1_timebar_conv1d_nometa.csv
```

### 2. Prepare Full CUSUM/TB Grid

```bash
uv run python -m kvant.ml_framework.scripts.run_experiment_grid prepare \
  --execute \
  --plan-out artifacts/final_plan/prepare_grid_commands.json
```

### 3. Train Full CUSUM/TB Conv1D Grid

```bash
caffeinate -dims uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-conv1d \
  --execute \
  --wandb-project day-trading-experiments \
  --extra-train-arg=--seed \
  --extra-train-arg 1337 \
  --plan-out artifacts/final_plan/train_grid_commands.json
```

### 4. Select Best Grid Configuration

```bash
uv run python scripts/select_best_grid_config.py \
  --results-glob "results/grid_search/E2-grid-conv1d-w240-*.csv" \
  --primary-metric val_f1_macro \
  --selection-json artifacts/final_plan/selected_grid.json \
  --env-out artifacts/final_plan/selected_grid.env \
  --promising-out reports/promising_grid_configs.json

source artifacts/final_plan/selected_grid.env

echo "$BEST_GRID_RUN"
echo "$BEST_GRID_RESULT"
echo "$BEST_MANIFEST"
```

### 5. Generate Grid Tables And Heatmap

```bash
uv run python scripts/generate_experiment_report.py \
  --results-glob "results/grid_search/E2-grid-conv1d-w240-*.csv" \
  --metric val_f1_macro \
  --metric val_accuracy \
  --metric test_f1_macro \
  --metric test_accuracy \
  --metric test_true_side_class_0_pct \
  --metric test_true_side_class_1_pct \
  --metric test_pred_side_class_0_pct \
  --metric test_pred_side_class_1_pct \
  --grid-heatmap-metric val_f1_macro
```

### Person 1 Delivers

Share these files with everyone before Phase 2 starts:

```text
results/main/E1_timebar_conv1d_nometa.csv
results/grid_search/E2-grid-conv1d-w240-*.csv
artifacts/final_plan/selected_grid.json
artifacts/final_plan/selected_grid.env
reports/promising_grid_configs.json
reports/generated/tables/summary_metrics.csv
reports/generated/tables/summary_metrics.tex
reports/generated/tables/class_distribution_test.csv
reports/generated/figures/grid_heatmap_val_f1_macro.png
reports/generated/figures/grid_heatmap_val_f1_macro.pdf
reports/generated/figures/predicted_vs_true_class_distribution_test.png
reports/generated/figures/predicted_vs_true_class_distribution_test.pdf
```

Everyone else places the shared files at the same paths and runs:

```bash
source artifacts/final_plan/selected_grid.env
```

## Phase 2: Parallel Follow-Up Experiments

Start Phase 2 only after Person 1 has shared the selected grid files.

### Person 1: Density-Matched Time-Bar Control

```bash
source artifacts/final_plan/selected_grid.env

uv run python scripts/make_density_matched_timebar.py \
  --selection-json artifacts/final_plan/selected_grid.json \
  --split train \
  --labeler next_bar \
  --output-manifest src/kvant/ml_framework/prepared/E2_timebar_density_matched_nextbar_cv_manifest.json \
  --execute

caffeinate -dims uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/E2_timebar_density_matched_nextbar_cv_manifest.json \
  --model conv1d --epochs 20 --seed 1337 \
  --checkpoint-out-dir artifacts/E2_timebar_density_matched_nextbar \
  --wandb-project day-trading-experiments \
  --wandb-name E2-timebar-density-matched-nextbar \
  --transaction-cost 0 \
  --bet-sizing fixed \
  --no-meta \
  --fixed-bet-size 1.0 \
  --results-out results/main/E2_timebar_density_matched_nextbar.csv
```

Deliver:

```text
results/main/E2_timebar_density_matched_nextbar.csv
src/kvant/ml_framework/prepared/E2_timebar_density_matched_nextbar_cv_manifest.json
```

### Person 2: Simple Baselines And Buy-And-Hold

These classifier baselines must stay on the same `BEST_MANIFEST` as the selected CUSUM Conv1D and ResNet-LSTM runs.
Compare them with classification metrics only. Buy-and-hold is an economic baseline on the same folds, not a macro-F1
baseline.

```bash
source artifacts/final_plan/selected_grid.env

uv run python scripts/simple_baselines.py \
  --model majority \
  --cv-manifest "$BEST_MANIFEST" \
  --wandb-project day-trading-experiments \
  --wandb-name E0-majority \
  --seed 1337 \
  --output results/baselines/E0_majority.csv

uv run python scripts/simple_baselines.py \
  --model random \
  --cv-manifest "$BEST_MANIFEST" \
  --wandb-project day-trading-experiments \
  --wandb-name E0-random \
  --seed 1337 \
  --output results/baselines/E0_random.csv

uv run python scripts/simple_baselines.py \
  --model logreg \
  --cv-manifest "$BEST_MANIFEST" \
  --wandb-project day-trading-experiments \
  --wandb-name E0-logreg \
  --seed 1337 \
  --output results/baselines/E0_logreg.csv

uv run python scripts/simple_baselines.py \
  --model random_forest \
  --cv-manifest "$BEST_MANIFEST" \
  --wandb-project day-trading-experiments \
  --wandb-name E0-random-forest \
  --seed 1337 \
  --output results/baselines/E0_random_forest.csv

uv run python scripts/buy_and_hold_baseline.py \
  --cv-manifest "$BEST_MANIFEST" \
  --transaction-cost 0 \
  --wandb-project day-trading-experiments \
  --wandb-name E0-buy-and-hold \
  --output results/baselines/E0_buy_and_hold.csv
```

Deliver:

```text
results/baselines/E0_majority.csv
results/baselines/E0_random.csv
results/baselines/E0_logreg.csv
results/baselines/E0_random_forest.csv
results/baselines/E0_buy_and_hold.csv
```

### Person 3: ResNet-LSTM Follow-Up

```bash
source artifacts/final_plan/selected_grid.env

caffeinate -dims uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-resnet \
  --execute \
  --wandb-project day-trading-experiments \
  --extra-train-arg=--seed \
  --extra-train-arg 1337
```

Deliver:

```bash
echo "$BEST_RESNET_RESULT"
```

The printed CSV path should exist locally and must be shared for final aggregation.
This ResNet-LSTM run is intended to be compared against the selected Conv1D CUSUM run and the scikit-learn baselines
using classification metrics only, because they all use the same selected CUSUM manifest/config family.

### Person 4: ResNet-LSTM Confidence Sweep And Meta-Selection Sweep

These sweeps use the selected CUSUM/TB setup with the ResNet-LSTM architecture, so Person 3's no-meta ResNet-LSTM run is
the direct baseline.

These sweeps use fixed bet size `1.0`; the meta-selection sweep tests TAKE/PASS filtering, not Kelly sizing.
For these sweeps, decision and portfolio metrics are part of the main interpretation because the policy now chooses
which signals to trade.

```bash
source artifacts/final_plan/selected_grid.env

caffeinate -dims uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-confidence \
  --execute \
  --wandb-project day-trading-experiments \
  --extra-train-arg=--seed \
  --extra-train-arg 1337

caffeinate -dims uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-meta \
  --execute \
  --wandb-project day-trading-experiments \
  --extra-train-arg=--seed \
  --extra-train-arg 1337
```

Deliver:

```bash
ls $BEST_CONFIDENCE_GLOB
ls $BEST_META_GLOB
```

All printed CSV files must be shared for final aggregation.

## Phase 3: Final Aggregation

One person collects all CSV files into these locations:

```text
results/baselines/
results/grid_search/
results/main/
```

Then run:

```bash
source artifacts/final_plan/selected_grid.env

uv run python scripts/summarize_prepared_manifests.py \
  --manifest Timebar15m=src/kvant/ml_framework/prepared/E1_timebar_cv_manifest.json \
  --manifest BestCUSUMTB="$BEST_MANIFEST" \
  --manifest DensityMatchedNextBar=src/kvant/ml_framework/prepared/E2_timebar_density_matched_nextbar_cv_manifest.json

uv run python scripts/generate_experiment_report.py \
  --result E0-majority=results/baselines/E0_majority.csv \
  --result E0-random=results/baselines/E0_random.csv \
  --result E0-logreg=results/baselines/E0_logreg.csv \
  --result E0-random-forest=results/baselines/E0_random_forest.csv \
  --result E0-buy-and-hold=results/baselines/E0_buy_and_hold.csv \
  --result E1-timebar-conv1d-nometa=results/main/E1_timebar_conv1d_nometa.csv \
  --result E2-timebar-density-matched-nextbar=results/main/E2_timebar_density_matched_nextbar.csv \
  --result "$BEST_GRID_RUN=$BEST_GRID_RESULT" \
  --result "$BEST_RESNET_RUN=$BEST_RESNET_RESULT" \
  --results-glob "$BEST_CONFIDENCE_GLOB" \
  --results-glob "$BEST_META_GLOB" \
  --metric test_accuracy \
  --metric test_f1_macro \
  --metric test_true_side_class_0_pct \
  --metric test_true_side_class_1_pct \
  --metric test_pred_side_class_0_pct \
  --metric test_pred_side_class_1_pct \
  --metric test_trade_signal_rate \
  --metric test_directional_acted_accuracy \
  --metric test_portfolio_total_return_pct \
  --metric test_portfolio_sharpe_ratio_annualized \
  --metric test_portfolio_max_drawdown_pct \
  --metric test_portfolio_n_executed_trades \
  --metric test_trade_signal_class_0_pct \
  --metric test_trade_signal_class_1_pct \
  --metric test_trade_signal_class_2_pct \
  --comparison E1-timebar-conv1d-nometa="$BEST_GRID_RUN" \
  --comparison E2-timebar-density-matched-nextbar="$BEST_GRID_RUN" \
  --comparison "$BEST_GRID_RUN=$BEST_RESNET_RUN"
```

Expected final outputs:

```text
reports/generated/tables/dataset_summary.csv
reports/generated/tables/dataset_summary.tex
reports/generated/tables/summary_metrics.csv
reports/generated/tables/summary_metrics.tex
reports/generated/tables/class_distribution_test.csv
reports/generated/tables/confusion_matrices_test.csv
reports/generated/tables/pairwise_tests.csv
reports/generated/tables/pairwise_tests.tex
reports/generated/figures/*.png
reports/generated/figures/*.pdf
```

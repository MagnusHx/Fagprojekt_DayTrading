# Day Trading as a Multi-Level Decision Problem

This README is the reproducibility runbook. Run the commands in order from the repository root. The intended outcome is:

- prepared data manifests
- W&B runs with stable names
- per-fold result CSV files
- selected best CUSUM/triple-barrier config
- statistical comparisons
- report-ready LaTeX tables
- report-ready PNG/PDF figures

The project is reference-inspired, not a direct reproduction of the crypto paper. We test whether the general pipeline
transfers to intraday U.S. equities.

## Research Questions

```text
RQ1: How does a reference-inspired machine learning trading pipeline perform when applied to intraday U.S. equities?

RQ2: Does CUSUM sampling with triple-barrier labelling improve predictive and economic performance compared with a
     time-based baseline?

RQ3: Does confidence-based selective trading improve the quality and risk-adjusted performance of executed trades?

RQ4: Does a learned meta-selection model improve trade selection beyond the primary model's own confidence?
```

## Fixed Configuration

| Parameter | Value |
| --- | --- |
| Ticker universe size | 20 |
| Ticker selection | Top dollar volume using train data only |
| Walk-forward folds | 5 |
| Train/validation/test | 1 year / 1 quarter / 1 quarter |
| Sequence length | 12 |
| Time-bar baseline | 15 minutes |
| CUSUM thresholds | 1%, 2%, 3% |
| Triple-barrier heights | 1%, 2%, 4%, 6% |
| Vertical barrier width | 240 minutes |
| Transaction cost | 0.001 |
| Random seed | 1337 |
| W&B project | `day-trading-experiments` |

## Output Locations

```text
results/baselines/             simple baselines and buy-and-hold
results/grid_search/           CUSUM/TB grid, ResNet, confidence, and meta sweeps
results/main/                  timebar and density-matched timebar controls
reports/generated/tables/      CSV and LaTeX tables
reports/generated/figures/     PNG and PDF figures
artifacts/final_plan/          generated command plans and selected config files
artifacts/                     checkpoints
```

## Step 0: Install And Smoke Check

```bash
uv sync
uv run python -m pytest tests/test_experiment_grid.py
uv run ruff check .
```

On macOS, long training commands use `caffeinate -dims` so the machine does not sleep.

## Step 0b: Lock W&B Project For Everyone

Every group member must use the same W&B project and entity before running experiments:

```bash
export WANDB_PROJECT=day-trading-experiments
export WANDB_ENTITY=s245509-danmarks-tekniske-universitet-dtu
```

Optional local `.env` setup:

```bash
cp .env.example .env
```

Do not change these values between runs. All experiment scripts use these values by default, and grid-generated training
commands pass the same project/entity through to `train_experiment`.

## Step 1: Prepare The 15-Minute Time-Bar Baseline

```bash
uv run python -m kvant.ml_prepare_data.prepare_experiment \
  --sampler time_bar --time-bar-minutes 15 \
  --labeler next_bar \
  --cv-manifest src/kvant/ml_framework/prepared/E1_timebar_cv_manifest.json
```

Expected output:

```text
src/kvant/ml_framework/prepared/E1_timebar_cv_manifest.json
```

## Step 2: Prepare The Full CUSUM/TB Grid

This prepares all 12 data configurations:

```text
CUSUM h = 0.01, 0.02, 0.03
TB height = 1%, 2%, 4%, 6%
W = 240 minutes
```

First print/write the command plan:

```bash
uv run python -m kvant.ml_framework.scripts.run_experiment_grid prepare \
  --plan-out artifacts/final_plan/prepare_grid_commands.json
```

Then execute all missing preparations:

```bash
uv run python -m kvant.ml_framework.scripts.run_experiment_grid prepare \
  --execute \
  --plan-out artifacts/final_plan/prepare_grid_commands.json
```

Expected outputs:

```text
artifacts/final_plan/prepare_grid_commands.json
src/kvant/ml_framework/prepared/sb_L_12_w240_h1_fixedCUSUM0.01_cv_manifest.json
src/kvant/ml_framework/prepared/sb_L_12_w240_h2_fixedCUSUM0.01_cv_manifest.json
...
src/kvant/ml_framework/prepared/sb_L_12_w240_h6_fixedCUSUM0.03_cv_manifest.json
```

## Step 3: Train The 15-Minute Time-Bar Conv1D Baseline

```bash
caffeinate -dims uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/E1_timebar_cv_manifest.json \
  --model conv1d --epochs 20 --seed 1337 \
  --checkpoint-out-dir artifacts/E1_timebar_conv1d_nometa \
  --wandb-project day-trading-experiments \
  --wandb-name E1-timebar-conv1d-nometa \
  --transaction-cost 0.001 \
  --no-meta \
  --results-out results/main/E1_timebar_conv1d_nometa.csv
```

Expected output:

```text
results/main/E1_timebar_conv1d_nometa.csv
```

## Step 4: Train The CUSUM/TB Conv1D Grid

First print/write the command plan:

```bash
uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-conv1d \
  --wandb-project day-trading-experiments \
  --extra-train-arg=--seed \
  --extra-train-arg 1337 \
  --plan-out artifacts/final_plan/train_grid_commands.json
```

Then execute the full grid:

```bash
caffeinate -dims uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-conv1d \
  --execute \
  --wandb-project day-trading-experiments \
  --extra-train-arg=--seed \
  --extra-train-arg 1337 \
  --plan-out artifacts/final_plan/train_grid_commands.json
```

Expected outputs:

```text
artifacts/final_plan/train_grid_commands.json
results/grid_search/E2-grid-conv1d-w240-tb1-cusum1-nometa.csv
results/grid_search/E2-grid-conv1d-w240-tb2-cusum1-nometa.csv
...
results/grid_search/E2-grid-conv1d-w240-tb6-cusum3-nometa.csv
```

To split the grid across machines, use the same command with different `--start-index` and `--max-runs`. Example:

```bash
caffeinate -dims uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-conv1d \
  --execute \
  --start-index 0 \
  --max-runs 4 \
  --wandb-project day-trading-experiments \
  --extra-train-arg=--seed \
  --extra-train-arg 1337
```

## Step 5: Select The Best Grid Config

This uses validation metrics only. The default primary selection metric is `val_f1_macro`, with validation Sharpe and
validation total return as tie-breakers.

```bash
uv run python scripts/select_best_grid_config.py \
  --results-glob "results/grid_search/E2-grid-conv1d-w240-*.csv" \
  --primary-metric val_f1_macro \
  --selection-json artifacts/final_plan/selected_grid.json \
  --env-out artifacts/final_plan/selected_grid.env \
  --promising-out reports/promising_grid_configs.json
```

Expected outputs:

```text
artifacts/final_plan/selected_grid.json
artifacts/final_plan/selected_grid.env
reports/promising_grid_configs.json
```

Load the selected config into the current terminal:

```bash
source artifacts/final_plan/selected_grid.env
```

Quick check:

```bash
echo "$BEST_GRID_RUN"
echo "$BEST_GRID_RESULT"
echo "$BEST_MANIFEST"
```

## Step 6: Generate Grid Tables And Heatmaps

```bash
uv run python scripts/generate_experiment_report.py \
  --results-glob "results/grid_search/E2-grid-conv1d-w240-*.csv" \
  --metric val_f1_macro \
  --metric val_portfolio_sharpe_ratio_annualized \
  --metric val_portfolio_total_return_pct \
  --metric test_f1_macro \
  --metric test_portfolio_sharpe_ratio_annualized \
  --metric test_portfolio_total_return_pct \
  --grid-heatmap-metric val_f1_macro
```

Expected outputs:

```text
reports/generated/tables/summary_metrics.csv
reports/generated/tables/summary_metrics.tex
reports/generated/figures/grid_heatmap_val_f1_macro.png
reports/generated/figures/grid_heatmap_val_f1_macro.pdf
```

## Step 7: Compare Time-Bar Conv1D Against Best CUSUM/TB Conv1D

Run after `source artifacts/final_plan/selected_grid.env`.

```bash
uv run python scripts/compare_experiments.py \
  --results-a results/main/E1_timebar_conv1d_nometa.csv \
  --results-b "$BEST_GRID_RESULT" \
  --name-a E1-timebar-conv1d-nometa \
  --name-b "$BEST_GRID_RUN" \
  --metrics \
    test_accuracy \
    test_f1_macro \
    test_trade_signal_rate \
    test_directional_acted_accuracy \
    test_portfolio_total_return_pct \
    test_portfolio_sharpe_ratio_annualized \
    test_portfolio_max_drawdown_pct \
    test_true_side_class_0_pct \
    test_true_side_class_1_pct \
    test_pred_side_class_0_pct \
    test_pred_side_class_1_pct \
    test_trade_signal_class_0_pct \
    test_trade_signal_class_1_pct \
    test_trade_signal_class_2_pct \
  --wandb-project day-trading-experiments \
  --wandb-name E2-timebar-vs-best-cusumtb
```

This is the main RQ2 comparison.

## Step 8: Prepare Density-Matched Time-Bar Control

This control tests whether CUSUM/TB helps because of better event definition or merely because it changes sample
density.

Prepare density-matched time bars with next-bar labels:

```bash
uv run python scripts/make_density_matched_timebar.py \
  --selection-json artifacts/final_plan/selected_grid.json \
  --split train \
  --labeler next_bar \
  --output-manifest src/kvant/ml_framework/prepared/E2_timebar_density_matched_nextbar_cv_manifest.json \
  --execute
```

Expected outputs:

```text
src/kvant/ml_framework/prepared/E2_timebar_density_matched_nextbar_cv_manifest.json
```

## Step 9: Train Density-Matched Time-Bar Control

```bash
caffeinate -dims uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/E2_timebar_density_matched_nextbar_cv_manifest.json \
  --model conv1d --epochs 20 --seed 1337 \
  --checkpoint-out-dir artifacts/E2_timebar_density_matched_nextbar \
  --wandb-project day-trading-experiments \
  --wandb-name E2-timebar-density-matched-nextbar \
  --transaction-cost 0.001 \
  --no-meta \
  --results-out results/main/E2_timebar_density_matched_nextbar.csv
```

Expected outputs:

```text
results/main/E2_timebar_density_matched_nextbar.csv
```

## Step 10: Dataset Summary Table And Figure

Run after the selected grid and density-matched manifests exist.

```bash
uv run python scripts/summarize_prepared_manifests.py \
  --manifest Timebar15m=src/kvant/ml_framework/prepared/E1_timebar_cv_manifest.json \
  --manifest BestCUSUMTB="$BEST_MANIFEST" \
  --manifest DensityMatchedNextBar=src/kvant/ml_framework/prepared/E2_timebar_density_matched_nextbar_cv_manifest.json
```

Expected outputs:

```text
reports/generated/tables/dataset_summary.csv
reports/generated/tables/dataset_summary.tex
reports/generated/figures/sample_count_comparison.png
reports/generated/figures/sample_count_comparison.pdf
```

## Step 11: Run Simple Scikit-Learn Baselines On The Selected Config

Run after `source artifacts/final_plan/selected_grid.env`.

```bash
uv run python scripts/simple_baselines.py \
  --model majority \
  --cv-manifest "$BEST_MANIFEST" \
  --wandb-project day-trading-experiments \
  --wandb-name E0-majority \
  --output results/baselines/E0_majority.csv

uv run python scripts/simple_baselines.py \
  --model random \
  --cv-manifest "$BEST_MANIFEST" \
  --wandb-project day-trading-experiments \
  --wandb-name E0-random \
  --output results/baselines/E0_random.csv

uv run python scripts/simple_baselines.py \
  --model logreg \
  --cv-manifest "$BEST_MANIFEST" \
  --wandb-project day-trading-experiments \
  --wandb-name E0-logreg \
  --output results/baselines/E0_logreg.csv

uv run python scripts/simple_baselines.py \
  --model random_forest \
  --cv-manifest "$BEST_MANIFEST" \
  --wandb-project day-trading-experiments \
  --wandb-name E0-random-forest \
  --output results/baselines/E0_random_forest.csv
```

Optional:

```bash
uv run python scripts/simple_baselines.py \
  --model hist_gb \
  --cv-manifest "$BEST_MANIFEST" \
  --wandb-project day-trading-experiments \
  --wandb-name E0-hist-gb \
  --output results/baselines/E0_hist_gb.csv
```

## Step 12: Run Buy-And-Hold Baseline On The Selected Config

```bash
uv run python scripts/buy_and_hold_baseline.py \
  --cv-manifest "$BEST_MANIFEST" \
  --transaction-cost 0.001 \
  --wandb-project day-trading-experiments \
  --wandb-name E0-buy-and-hold \
  --output results/baselines/E0_buy_and_hold.csv
```

## Step 13: Train ResNet-LSTM On The Selected Config

`Step 5` already wrote `reports/promising_grid_configs.json`, so this command uses the selected CUSUM/TB config.

```bash
caffeinate -dims uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-resnet \
  --execute \
  --wandb-project day-trading-experiments \
  --extra-train-arg=--seed \
  --extra-train-arg 1337
```

Expected output:

```bash
echo "$BEST_RESNET_RESULT"
```

## Step 14: Compare Conv1D Against ResNet-LSTM

Run after `source artifacts/final_plan/selected_grid.env`.

```bash
uv run python scripts/compare_experiments.py \
  --results-a "$BEST_GRID_RESULT" \
  --results-b "$BEST_RESNET_RESULT" \
  --name-a "$BEST_GRID_RUN" \
  --name-b "$BEST_RESNET_RUN" \
  --metrics \
    test_accuracy \
    test_f1_macro \
    test_trade_signal_rate \
    test_directional_acted_accuracy \
    test_portfolio_total_return_pct \
    test_portfolio_sharpe_ratio_annualized \
    test_portfolio_max_drawdown_pct \
  --wandb-project day-trading-experiments \
  --wandb-name E3-conv1d-vs-resnet
```

## Step 15: Run Confidence-Based Selective Trading Sweep

This tests RQ3.

```bash
caffeinate -dims uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-confidence \
  --execute \
  --wandb-project day-trading-experiments \
  --extra-train-arg=--seed \
  --extra-train-arg 1337
```

Expected outputs match:

```bash
ls $BEST_CONFIDENCE_GLOB
```

## Step 16: Run Meta-Selection Sweep

This tests RQ4.

```bash
caffeinate -dims uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-meta \
  --execute \
  --wandb-project day-trading-experiments \
  --extra-train-arg=--seed \
  --extra-train-arg 1337
```

Expected outputs match:

```bash
ls $BEST_META_GLOB
```

## Step 17: Generate Final Tables And Figures

Run after all required experiments are complete. If optional `hist_gb`, ResNet, confidence, or meta runs were skipped,
remove those corresponding lines.

```bash
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
  --metric test_trade_signal_rate \
  --metric test_directional_acted_accuracy \
  --metric test_portfolio_total_return_pct \
  --metric test_portfolio_sharpe_ratio_annualized \
  --metric test_portfolio_max_drawdown_pct \
  --metric test_portfolio_n_executed_trades \
  --metric test_true_side_class_0_pct \
  --metric test_true_side_class_1_pct \
  --metric test_pred_side_class_0_pct \
  --metric test_pred_side_class_1_pct \
  --metric test_trade_signal_class_0_pct \
  --metric test_trade_signal_class_1_pct \
  --metric test_trade_signal_class_2_pct \
  --comparison E1-timebar-conv1d-nometa="$BEST_GRID_RUN" \
  --comparison E2-timebar-density-matched-nextbar="$BEST_GRID_RUN" \
  --comparison "$BEST_GRID_RUN=$BEST_RESNET_RUN"
```

Expected outputs:

```text
reports/generated/tables/summary_metrics.csv
reports/generated/tables/summary_metrics.tex
reports/generated/tables/confusion_matrices_test.csv
reports/generated/tables/pairwise_tests.csv
reports/generated/tables/pairwise_tests.tex
reports/generated/figures/*.png
reports/generated/figures/*.pdf
```

## Step 18: Sync W&B Offline Runs

Only run this if W&B was in offline mode.

```bash
wandb sync wandb/offline-run-*
```

## What Gets Logged

Training logs classification, decision, and portfolio metrics. It also logs class distributions:

```text
{split}/distribution/true_side/class_0_pct
{split}/distribution/true_side/class_1_pct
{split}/distribution/pred_side/class_0_pct
{split}/distribution/pred_side/class_1_pct
{split}/distribution/trade_signal/class_0_pct
{split}/distribution/trade_signal/class_1_pct
{split}/distribution/trade_signal/class_2_pct
```

Use these to catch prediction collapse, for example a model predicting only one class.

Training also writes fold-level confusion matrix counts to the result CSV:

```text
{split}_confusion_true0_pred0_count
{split}_confusion_true0_pred1_count
{split}_confusion_true1_pred0_count
{split}_confusion_true1_pred1_count
```

`scripts/generate_experiment_report.py` aggregates those counts across folds and writes report-ready confusion matrix
figures when the columns are present.

## Training Defaults

The training CLI uses cosine learning-rate scheduling by default:

```bash
--lr-scheduler cosine
```

Disable it only for an explicit ablation:

```bash
--lr-scheduler none
```

Model dropout defaults to `0.3`. Early stopping is available but not enabled unless a command explicitly passes:

```bash
--early-stopping-patience 5
```

## Report Outputs To Use

| Output | File |
| --- | --- |
| Dataset/sample count table | `reports/generated/tables/dataset_summary.tex` |
| Main metric table | `reports/generated/tables/summary_metrics.tex` |
| Pairwise statistical tests | `reports/generated/tables/pairwise_tests.tex` |
| Confusion matrix counts | `reports/generated/tables/confusion_matrices_test.csv` |
| Sample count figure | `reports/generated/figures/sample_count_comparison.pdf` |
| Grid heatmap | `reports/generated/figures/grid_heatmap_val_f1_macro.pdf` |
| Confusion matrix figures | `reports/generated/figures/confusion_matrix_test_*.pdf` |
| Metric comparison figures | `reports/generated/figures/*.pdf` |

## Interpretation Rules

1. Choose CUSUM/TB parameters using validation metrics only.
2. Use test metrics only for final reporting.
3. Treat a result as meaningful only if it is both practically relevant and statistically supported across folds.
4. Use classification metrics for model quality.
5. Use decision metrics for accepted-signal quality.
6. Use portfolio metrics for economic outcome.

This is a research codebase, not a live trading system or investment recommendation.

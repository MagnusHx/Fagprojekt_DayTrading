# Day-Trading as a Multilevel Decision Problem

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

## Fresh Clone Quick Start

Use this section if you have not worked in the repository before.

1. Clone the repository and enter the project root:

```bash
git clone https://github.com/MagnusHx/Fagprojekt_DayTrading.git
cd Fagprojekt_DayTrading
```

2. Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the terminal after installing `uv` if the command is still not found.

3. Install the project environment:

```bash
uv sync
```

4. Configure W&B. All group members must use the same project and entity:

```bash
export WANDB_PROJECT=day-trading-experiments
export WANDB_ENTITY=s245509-danmarks-tekniske-universitet-dtu
wandb login
```

5. Configure Hugging Face access if downloads are rate-limited or require authentication:

```bash
export HF_TOKEN=<your-hugging-face-token>
```

`HUGGINGFACE_HUB_TOKEN` also works. The data loader reads either variable.

6. Run every command below from the repository root. After Step 5, run `source artifacts/final_plan/selected_grid.env`
again in every new terminal before using variables such as `$BEST_MANIFEST`, `$BEST_GRID_RESULT`,
`$BEST_RESNET_RESULT`, `$BEST_CONFIDENCE_GLOB`, or `$BEST_META_GLOB`.

## Research Questions

1. Is it possible to translate the methods and pipeline used in the paper
   "Algorithmic crypto trading using information-driven bars, triple barrier
   labeling and deep learning" to a set of financial instruments, i.e.
   intraday U.S. equities?

2. Does a CUSUM-based event sampling and triple-barrier labeling framework
   improve predictive performance over a non-information-driven intraday
   stock trading baseline?

3. Do more complex models improve performance compared to baseline models?

4. Does confidence-based selective trading improve gross risk-adjusted
   backtest diagnostics?

5. Can a learned meta-labeling model improve trade selection beyond the
   primary model's own confidence?

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
| Transaction cost | 0 |
| Random seed | 1337 |
| W&B project | `day-trading-experiments` |

## Output Locations

```text
results/baselines/             simple baselines and buy-and-hold
results/grid_search/           CUSUM/TB grid, ResNet, confidence, and meta sweeps
results/main/                  timebar and density-matched timebar controls
results/prediction_diagnostics/ per-sample prediction CSVs and meta diagnostic plots
reports/generated/tables/      CSV and LaTeX tables
reports/generated/figures/     PNG and PDF figures
artifacts/final_plan/          generated command plans and selected config files
artifacts/                     checkpoints
```

## Step 0: Install And Smoke Check

If you already ran the fresh-clone setup above, `uv sync` can be skipped.

```bash
uv sync
uv run python -m pytest tests/test_experiment_grid.py
uv run ruff check .
```

On macOS, you can optionally prefix long training commands with `caffeinate -dims` to prevent sleep. The commands below omit it so they work on macOS, Linux, and HPC.

## Step 0b: Lock W&B Project For Everyone

Every group member must use the same W&B project and entity before running experiments:

```bash
export WANDB_PROJECT=day-trading-experiments
export WANDB_ENTITY=s245509-danmarks-tekniske-universitet-dtu
wandb login
```

Optional local `.env` setup:

```bash
cp .env.example .env
```

Do not change these values between runs. All experiment scripts use these values by default, and grid-generated training
commands pass the same project/entity through to `train_experiment`.

If Hugging Face returns rate-limit or authentication errors while preparing data, set:

```bash
export HF_TOKEN=<your-hugging-face-token>
```

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
uv run python -m kvant.ml_framework.scripts.train_experiment \
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

Expected output:

```text
results/main/E1_timebar_conv1d_nometa.csv
```

### HPC Run For Steps 1 And 3

Submit this from the repository root on DTU HPC:

```bash
bsub < hpc_experiment_1_rq1.sh
```

The job script follows the LSF batch-job pattern used on DTU HPC: explicit job name, queue, walltime, single-node CPU
span, per-slot memory request and limit, non-appending log files, and module loading inside the batch job.

## Step 4: Train The CUSUM/TB Conv1D Grid

Grid runs use fixed bet size `1.0`, so the time-bar/CUSUM and meta-selection comparisons are not affected by Kelly
sizing. Kelly is reserved for a later explicit sizing ablation.

The no-meta time-bar/CUSUM comparison is a primary-model comparison. It should be interpreted using macro F1, accuracy,
confusion matrices, and true-vs-predicted class distributions. Portfolio metrics may still be logged for diagnostics,
but they are not the main evidence for RQ2 because no-meta trades every signal and portfolio constraints can dominate
the economic result.

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
uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-conv1d \
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
uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-conv1d \
  --execute \
  --start-index 0 \
  --max-runs 4 \
  --wandb-project day-trading-experiments \
  --extra-train-arg=--seed \
  --extra-train-arg 1337
```

## Step 5: Select The Best Grid Config

This uses validation metrics only. For the no-meta CUSUM/TB grid, select by predictive quality, not portfolio outcome.
The default primary selection metric is `val_f1_macro`, with `val_accuracy` as the default tie-breaker.

This step creates the environment variables used by later commands. If you open a new terminal after this step, run the
`source` command again before continuing.

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
  --metric val_accuracy \
  --metric test_f1_macro \
  --metric test_accuracy \
  --metric test_true_side_class_0_pct \
  --metric test_true_side_class_1_pct \
  --metric test_pred_side_class_0_pct \
  --metric test_pred_side_class_1_pct \
  --grid-heatmap-metric val_f1_macro
```

Expected outputs:

```text
reports/generated/tables/summary_metrics.csv
reports/generated/tables/summary_metrics.tex
reports/generated/tables/class_distribution_test.csv
reports/generated/figures/grid_heatmap_val_f1_macro.png
reports/generated/figures/grid_heatmap_val_f1_macro.pdf
reports/generated/figures/predicted_vs_true_class_distribution_test.png
reports/generated/figures/predicted_vs_true_class_distribution_test.pdf
```

## Step 7: Compare Time-Bar Conv1D Against Best CUSUM/TB Conv1D

Run after `source artifacts/final_plan/selected_grid.env`.

This comparison is intentionally classification-only. Do not use no-meta portfolio returns, Sharpe, drawdown, or
executed-trade count as the main CUSUM-vs-timebar conclusion.

```bash
uv run python scripts/compare_experiments.py \
  --results-a results/main/E1_timebar_conv1d_nometa.csv \
  --results-b "$BEST_GRID_RESULT" \
  --name-a E1-timebar-conv1d-nometa \
  --name-b "$BEST_GRID_RUN" \
  --metrics \
    test_accuracy \
    test_f1_macro \
    test_true_side_class_0_pct \
    test_true_side_class_1_pct \
    test_pred_side_class_0_pct \
    test_pred_side_class_1_pct \
  --wandb-project day-trading-experiments \
  --wandb-name E2-timebar-vs-best-cusumtb
```

This is the main RQ2 comparison. Use the generated confusion matrix figures as the report-level error analysis.

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
uv run python -m kvant.ml_framework.scripts.train_experiment \
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

These baselines are meant to be directly comparable to the selected CUSUM Conv1D and ResNet-LSTM runs. Keep them on
the same `BEST_MANIFEST`, use the same walk-forward folds, and compare them with classification metrics only.

```bash
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
```

Optional:

```bash
uv run python scripts/simple_baselines.py \
  --model hist_gb \
  --cv-manifest "$BEST_MANIFEST" \
  --wandb-project day-trading-experiments \
  --wandb-name E0-hist-gb \
  --seed 1337 \
  --output results/baselines/E0_hist_gb.csv
```

## Step 12: Run Buy-And-Hold Baseline On The Selected Config

This is an economic baseline on the same selected CUSUM folds, not a classifier baseline. Do not rank it against
Conv1D, ResNet-LSTM, or the scikit-learn classifiers using macro F1.

```bash
uv run python scripts/buy_and_hold_baseline.py \
  --cv-manifest "$BEST_MANIFEST" \
  --transaction-cost 0 \
  --wandb-project day-trading-experiments \
  --wandb-name E0-buy-and-hold \
  --output results/baselines/E0_buy_and_hold.csv
```

## Step 13: Train ResNet-LSTM On The Selected Config

`Step 5` already wrote `reports/promising_grid_configs.json`, so this command uses the selected CUSUM/TB config.
For a fair architecture comparison, keep this run no-meta and on the same selected manifest/config family as the
baseline Conv1D and the scikit-learn baselines above.

Run `source artifacts/final_plan/selected_grid.env` first if this is a new terminal.

```bash
uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-resnet \
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
If the ResNet-LSTM run is no-meta, compare architecture quality using classification metrics only.
The same rule applies when informally comparing ResNet-LSTM against the scikit-learn baselines: use macro F1,
accuracy, confusion matrices, and class distributions on the same selected CUSUM manifest.

```bash
uv run python scripts/compare_experiments.py \
  --results-a "$BEST_GRID_RESULT" \
  --results-b "$BEST_RESNET_RESULT" \
  --name-a "$BEST_GRID_RUN" \
  --name-b "$BEST_RESNET_RUN" \
  --metrics \
    test_accuracy \
    test_f1_macro \
    test_true_side_class_0_pct \
    test_true_side_class_1_pct \
    test_pred_side_class_0_pct \
    test_pred_side_class_1_pct \
  --wandb-project day-trading-experiments \
  --wandb-name E3-conv1d-vs-resnet
```

## Step 15: Run Confidence-Based Selective Trading Sweep

This tests RQ3 on the selected CUSUM/TB setup using the ResNet-LSTM architecture from Step 13. These runs introduce a
decision threshold, so they can use decision and portfolio metrics in addition to classification metrics.

Run `source artifacts/final_plan/selected_grid.env` first if this is a new terminal.

The confidence threshold is a decision-time cutoff applied after the primary model has produced class probabilities.
The sweep therefore trains the primary ResNet-LSTM once per config at the first threshold and reuses that checkpoint
for the remaining thresholds with `--init-checkpoint-dir` and `--skip-primary-training`. Every threshold retains its
own W&B run and result CSV while using identical primary-model weights.

```bash
uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-confidence \
  --execute \
  --wandb-project day-trading-experiments \
  --extra-train-arg=--seed \
  --extra-train-arg 1337
```

Expected outputs match:

```bash
ls $BEST_CONFIDENCE_GLOB
```

Keep all four thresholds for a config together when splitting commands with `--start-index` and `--max-runs`; the
three reuse runs depend on the checkpoint produced by the first run.

## Step 16: Run Meta-Selection Sweep

This tests RQ4 on the selected CUSUM/TB setup using the ResNet-LSTM architecture from Step 13. It isolates whether the
meta-model improves TAKE/PASS selection. For meta-selection, report take rate, meta F1, acted directional accuracy,
executed trade count, and portfolio metrics.

Run `source artifacts/final_plan/selected_grid.env` first if this is a new terminal.

The meta-accept threshold is a decision-time cutoff applied to the meta model's TAKE probability. The sweep trains the
primary ResNet-LSTM **once per config** at the first threshold and reuses that fixed primary model for the remaining
thresholds. The first threshold writes its best-checkpoint bundles to its `--checkpoint-out-dir`; the later thresholds
load them via `--init-checkpoint-dir` and `--skip-primary-training`, then re-fit the meta-selection layer on the same
train-fold predictions and evaluate at their own threshold. This keeps every threshold on an identical primary network
and cuts training cost roughly fourfold.

```bash
uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-meta \
  --execute \
  --wandb-project day-trading-experiments \
  --extra-train-arg=--seed \
  --extra-train-arg 1337
```

For per-sample meta-selection diagnostics used to inspect logits, TAKE probabilities, trade decisions, and outliers, add:

```bash
  --extra-train-arg=--prediction-export-dir \
  --extra-train-arg results/prediction_diagnostics
```

Each threshold still produces its own W&B run, result CSV, and diagnostics. Because the reuse runs depend on the
training run that precedes them, keep a config's commands together: when splitting across machines with `--start-index`
and `--max-runs`, do not cut between a config's first (training) threshold and its later (reuse) thresholds.

Expected outputs match:

```bash
ls $BEST_META_GLOB
```

Generate the meta-selection diagnostic plots after the prediction CSVs exist:

```bash
uv run python scripts/plot_prediction_diagnostics.py \
  --input-dir results/prediction_diagnostics \
  --output-dir results/prediction_diagnostics/plots \
  --split val

uv run python scripts/plot_prediction_diagnostics.py \
  --input-dir results/prediction_diagnostics \
  --output-dir results/prediction_diagnostics/plots \
  --split test
```

Expected outputs:

```text
results/prediction_diagnostics/plots/val_logit_margin_vs_meta_take_proba.png
results/prediction_diagnostics/plots/val_meta_take_proba_vs_realized_return.png
results/prediction_diagnostics/plots/val_outlier_overlay.png
results/prediction_diagnostics/plots/test_logit_margin_vs_meta_take_proba.png
results/prediction_diagnostics/plots/test_meta_take_proba_vs_realized_return.png
results/prediction_diagnostics/plots/test_outlier_overlay.png
```

## Step 17: Generate Final Tables And Figures

Run after all required experiments are complete. If optional `hist_gb`, ResNet, confidence, or meta runs were skipped,
remove those corresponding lines.

Run `source artifacts/final_plan/selected_grid.env` first if this is a new terminal.

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

Expected outputs:

```text
reports/generated/tables/summary_metrics.csv
reports/generated/tables/summary_metrics.tex
reports/generated/tables/class_distribution_test.csv
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

For no-meta time-bar/CUSUM runs, the report should focus on:

```text
test_f1_macro
test_accuracy
confusion matrix counts/figures
true-side class distribution
predicted-side class distribution
```

Decision and portfolio metrics become primary only in confidence-threshold and meta-selection experiments, where the
research question is explicitly about trade filtering and execution quality.

Training also writes fold-level confusion matrix counts to the result CSV:

```text
{split}_confusion_true0_pred0_count
{split}_confusion_true0_pred1_count
{split}_confusion_true1_pred0_count
{split}_confusion_true1_pred1_count
```

`scripts/generate_experiment_report.py` aggregates those counts across folds and writes report-ready confusion matrix
figures when the columns are present.

The same report script also writes the class-distribution comparison:

```text
reports/generated/tables/class_distribution_test.csv
reports/generated/figures/predicted_vs_true_class_distribution_test.png
reports/generated/figures/predicted_vs_true_class_distribution_test.pdf
```

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

Fixed position sizing is the default for both no-meta and meta-selection runs:

```bash
--bet-sizing fixed
--fixed-bet-size 1.0
```

This keeps the primary research comparisons focused on sampling, labelling, and selection. Use Kelly only for a later
explicit sizing ablation:

```bash
--bet-sizing kelly
```

## Report Outputs To Use

| Output | File |
| --- | --- |
| Dataset/sample count table | `reports/generated/tables/dataset_summary.tex` |
| Main metric table | `reports/generated/tables/summary_metrics.tex` |
| Pairwise statistical tests | `reports/generated/tables/pairwise_tests.tex` |
| True/predicted class distribution table | `reports/generated/tables/class_distribution_test.csv` |
| Confusion matrix counts | `reports/generated/tables/confusion_matrices_test.csv` |
| Sample count figure | `reports/generated/figures/sample_count_comparison.pdf` |
| Grid heatmap | `reports/generated/figures/grid_heatmap_val_f1_macro.pdf` |
| True/predicted class distribution figure | `reports/generated/figures/predicted_vs_true_class_distribution_test.pdf` |
| Confusion matrix figures | `reports/generated/figures/confusion_matrix_test_*.pdf` |
| Metric comparison figures | `reports/generated/figures/*.pdf` |

## Interpretation Rules

1. Choose CUSUM/TB parameters using validation metrics only.
2. Use test metrics only for final reporting.
3. Treat a result as meaningful only if it is both practically relevant and statistically supported across folds.
4. For no-meta time-bar/CUSUM comparisons, use macro F1, accuracy, confusion matrices, and true-vs-predicted class distributions.
5. Do not use no-meta portfolio metrics as the main RQ2 conclusion.
6. Use decision metrics for accepted-signal quality in confidence-threshold and meta-selection runs.
7. Use portfolio metrics for economic outcome only when the experiment includes an explicit trade-selection policy.

This is a research codebase, not a live trading system or investment recommendation.

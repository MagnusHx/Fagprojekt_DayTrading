# Final Experiment Plan

This document mirrors the runnable workflow in `README.md`. The goal is to keep the report logic, code outputs, and
W&B runs aligned.

## Research Questions

| RQ | Question | Main experiments |
| --- | --- | --- |
| RQ1 | How does a reference-inspired machine learning trading pipeline perform on intraday U.S. equities? | Simple baselines, buy-and-hold, timebar Conv1D, final best model |
| RQ2 | Does CUSUM sampling with triple-barrier labelling improve predictive and economic performance compared with a time-based baseline? | Timebar Conv1D vs fixed-CUSUM/TB Conv1D grid |
| RQ3 | Does confidence-based selective trading improve the quality and risk-adjusted performance of executed trades? | Primary confidence threshold sweep |
| RQ4 | Does a learned meta-selection model improve trade selection beyond the primary model's own confidence? | No-meta vs meta-selection sweep |

## Fixed Setup

| Parameter | Value |
| --- | --- |
| Ticker universe size | 20 |
| Ticker selection | Top dollar volume, train only |
| Walk-forward folds | 5 |
| Train/validation/test | 1 year / 1 quarter / 1 quarter |
| Sequence length | 12 |
| Time bars | 15 minutes |
| CUSUM thresholds | 1%, 2%, 3% |
| Triple-barrier heights | 1%, 2%, 4%, 6% |
| Vertical barrier width | 240 minutes |
| Transaction cost | 0.001 |
| Random seed | 1337 |

All parameters not being tested are frozen.

## Experiment Stages

### E0: Simple Baselines

Purpose: establish whether simple non-neural models already solve the primary side-prediction task.

Runs:

| Run | Model | Output |
| --- | --- | --- |
| `E0-majority` | Majority classifier | `results/baselines/E0_majority.csv` |
| `E0-random` | Stratified random classifier | `results/baselines/E0_random.csv` |
| `E0-logreg` | Logistic regression | `results/baselines/E0_logreg.csv` |
| `E0-random-forest` | Random forest | `results/baselines/E0_random_forest.csv` |
| `E0-hist-gb` | Histogram gradient boosting, optional | `results/baselines/E0_hist_gb.csv` |

### E0b: Buy-And-Hold Baseline

Purpose: provide an economic benchmark independent of classification.

Output: `results/baselines/E0_buy_and_hold.csv`.

### E1: Time-Based Conv1D Baseline

Purpose: establish the simplest neural trading pipeline.

Configuration:

| Sampler | Label | Model | Meta |
| --- | --- | --- | --- |
| 15-minute time bars | Next-bar direction | Conv1D | OFF |

Output: `results/main/E1_timebar_conv1d_nometa.csv`.

### E2: Fixed CUSUM/Triple-Barrier Grid

Purpose: test information-driven sampling and triple-barrier labels without changing model architecture.

Configuration:

| Sampler | Label | Model | Meta |
| --- | --- | --- | --- |
| Fixed CUSUM | Triple barrier | Conv1D | OFF |

Grid:

| CUSUM h | TB height | W |
| --- | --- | --- |
| 1%, 2%, 3% | 1%, 2%, 4%, 6% | 240 minutes |

The best configuration is selected using validation metrics only.

Implementation:

| Task | Script |
| --- | --- |
| Train/prepare fixed grid | `python -m kvant.ml_framework.scripts.run_experiment_grid` |
| Select best grid config | `scripts/select_best_grid_config.py` |
| Save selected config for later steps | `artifacts/final_plan/selected_grid.json` and `artifacts/final_plan/selected_grid.env` |
| Save selected config for ResNet/meta follow-ups | `reports/promising_grid_configs.json` |

### E2b: Density-Matched Time-Bar Control

Purpose: check whether the CUSUM/TB result is caused by the event definition or simply by a different number of
training samples.

The control computes the average usable samples per ticker-day from the selected CUSUM/TB manifest using the training
split only. It then prepares a time-bar baseline with the nearest integer interval that matches this average sample
density.

Recommended control:

| Control | Label | Purpose |
| --- | --- | --- |
| Density-matched timebar | Next-bar direction | Fairer version of the original timebar baseline |

### E3: Model Architecture

Purpose: test whether model complexity improves the selected CUSUM/TB setup.

Comparison:

| Model A | Model B | Data | Meta |
| --- | --- | --- | --- |
| Conv1D | ResNet-LSTM | Best CUSUM/TB config | OFF |

### E4: Confidence-Based Selective Trading

Purpose: test whether accepting only high-confidence primary predictions improves decision and portfolio metrics.

Threshold sweep:

| Thresholds |
| --- |
| 0.45, 0.50, 0.55, 0.60 |

### E5: Meta-Selection

Purpose: test whether a learned meta-model improves accepted-signal quality beyond primary confidence.

Comparison:

| Run | Selection |
| --- | --- |
| No meta | Accept primary signals directly |
| Meta | Logistic meta-model decides TAKE/PASS |

### E6: Time-Period Robustness

Purpose: if time allows, repeat the best setup on another market period.

## Tables And Figures

| Output | Produced by | Purpose |
| --- | --- | --- |
| Dataset summary table | `scripts/summarize_prepared_manifests.py` | Show sample counts across timebar/CUSUM setups |
| Baseline table | `scripts/generate_experiment_report.py` | Compare simple baselines and neural runs |
| Grid heatmap | `scripts/generate_experiment_report.py` | Show validation performance over CUSUM/TB grid |
| Main comparison table | `scripts/generate_experiment_report.py` | Report mean and 95% CI across folds |
| Pairwise tests | `scripts/generate_experiment_report.py` or `scripts/compare_experiments.py` | Report p-values and mean differences |
| Class distributions | W&B + result CSVs | Check label balance and prediction collapse |

## Priority

Must-have:

1. Fixed RQs.
2. README command flow.
3. Timebar Conv1D.
4. Fixed CUSUM/TB grid.
5. Simple baselines.
6. Buy-and-hold.
7. Summary tables and figures.
8. Pairwise statistics.

Should-have:

1. ResNet-LSTM on selected config.
2. Confidence threshold sweep.
3. Meta-selection sweep.

Nice-to-have:

1. Time-period robustness.
2. Kelly and richer sizing analysis.

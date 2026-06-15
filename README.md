# kvant

`kvant` is a quant research project for short-horizon US equity trading on minute-level OHLCV data. The current approach follows a Lopez de Prado-inspired side-plus-meta-labeling workflow: prepared artifacts keep the raw three-class triple-barrier event outcome (`down`, `exit`, `up`), the training pipeline derives a binary primary side target (`down`, `up`), and a meta-label layer decides whether a predicted side should be traded or abstained from.

The project is built around reproducible walk-forward folds. Data preparation computes technical features on minute bars before sampling, fits all learned preparation state on train data only, and then prepares train/validation/test artifacts with sampled OHLCV, labels, metadata, and diagnostics. Training consumes those prepared artifacts with PyTorch models, validates the artifact contract before running, logs metrics through W&B, and evaluates model quality, decision quality, paper-style trade diagnostics, and a stricter budget-constrained portfolio simulator under transaction costs.

## Current pipeline

1. Download/cache minute OHLCV shards from Hugging Face and build walk-forward splits of top-volume US equities.
2. Fit a train-only `TunedCUSUMBarSampler` targeting a configurable bars-per-day density.
3. Compute intraday technical features on minute bars before sampling, standardize them from train data, and select features for the primary side task with train-only F-score selection.
4. Label sampled bars with triple-barrier outcomes and persist the canonical `event_outcome` label space.
5. Enforce live-safe label and backtest timing: the sampled bar is the signal, trades enter on the next sampled bar, and label intervals crossing split boundaries are purged/embargoed.
6. Train a primary side classifier on actionable `down/up` events while retaining `exit` rows for decision and abstention evaluation.
7. Fit/evaluate the meta-label decision policy using model probabilities, embeddings, uncertainty, prepared volatility/return aliases, rolling ticker win/return statistics, and time-since-event context.
8. Report side, meta, decision, execution, paper-economics, and portfolio-account metrics.

## Important paths

```txt
src/kvant/ml_prepare_data/       Data preparation, sampling, features, labeling
src/kvant/ml_framework/          Prepared data loading, training, validation, logging
src/kvant/ml_framework/prepared/ Generated prepared fold artifacts and CV manifests (ignored)
tests/                           Unit and pipeline-contract tests
reports/                         Analysis notes and experiment sheets
docs/                            MkDocs documentation sources
references/                      Source papers used by reports
artifacts/                       Generated diagnostics, plots, checkpoints, and smoke reports (ignored)
```

## Running the 8-Day Experiment Plan

Follow the exact commands below to reproduce experiments E1–E4 from [experiment_plan_8day.md](reports/experiment_plan_8day.md), then E0 as a final summary.

### Overview

Experiments answer four research questions via a ladder of comparisons:
- **E1** (L1 & L2): Time bars vs CUSUM, **RQ2** — Does information-driven sampling improve performance?
- **E2** (L3): Model complexity (Conv1D vs ResNet-LSTM) — bonus insight
- **E3** (L4): Selective trading / confidence thresholds, **RQ3** — Does selective trading improve risk-adjusted returns?
- **E4** (L5): Meta-selection ablation, **RQ4** — Does meta-selection add value?
- **E0** (RQ1, final summary): Can we translate the crypto method to stocks? Compare best model vs baselines.

All use: seed `1337`, sequence length `12`, Conv1D (default), transaction cost `0.001`, all 5 folds (final runs).
Fixed triple-barrier parameters: hb=2.5%, W=240 min.

### Running experiments in order

Recommended sequence:

1. **Prepare all data** (B1–B5): create E1_timebar, E1_cusum, E2 manifests (~20 min)
2. **E1** (RQ2): E1-timebar + E1-cusum in parallel or sequential (~2 hours per arm, 5 folds)
3. **E2** (Model complexity): E2-conv1d, then E2-resnet if E2-conv1d beats baselines (~2 hours)
4. **E3** (RQ3): Threshold sweep on best E2 checkpoint (~30 min)
5. **E4** (RQ4): 3 meta ablation arms on same E2 manifest (~3 hours)
6. **E0** (RQ1, final): Compare best model from E1–E4 vs trivial baselines (~10 min)
7. **Analysis**: aggregate results, run statistical comparisons, write report

### Prepare all data first (B1–B5 prerequisite)

Run these preparation steps once, then all experiments use the outputs:

```bash
# Prepare E1-timebar data (B1+B2: time-bar sampler + next-bar labeler)
uv run python -m kvant.ml_prepare_data.prepare_experiment \
  --sampler time_bar --time-bar-minutes 15 \
  --labeler next_bar \
  --cv-manifest src/kvant/ml_framework/prepared/E1_timebar_cv_manifest.json

# Prepare E1-cusum data (CUSUM + triple-barrier with fixed params)
uv run python -m kvant.ml_prepare_data.prepare_experiment \
  --sampler tuned_cusum --target-bars-per-day 15 \
  --labeler triple_barrier --barrier-height-pct 2.5 --barrier-width 240 \
  --cv-manifest src/kvant/ml_framework/prepared/E1_cusum_cv_manifest.json

# Prepare E2 data (same as E1-cusum)
uv run python -m kvant.ml_prepare_data.prepare_experiment \
  --sampler tuned_cusum --target-bars-per-day 15 \
  --labeler triple_barrier --barrier-height-pct 2.5 --barrier-width 240 \
  --cv-manifest src/kvant/ml_framework/prepared/E2_cv_manifest.json

# Verify all build items exist
test -f scripts/simple_baselines.py && echo "✓ B3: simple_baselines.py"
uv run python -m kvant.ml_framework.scripts.train_experiment --help | grep -q "no-meta" && echo "✓ B4: --no-meta flag"
test -f scripts/reconcile_metrics.py && echo "✓ B5: reconcile_metrics.py"
```

### E1: RQ2 head-to-head (L1 vs L2) — Table 2 + equity curves

**Does information-driven sampling + triple-barrier labeling improve performance?**

Compare time bars (simple baseline) vs CUSUM + triple-barrier (information-driven method). Same model (Conv1D), same features — only data pipeline differs. Data prepared above.

```bash
# Train E1-timebar (all 5 folds)
uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/E1_timebar_cv_manifest.json \
  --model conv1d --epochs 20 --seed 1337 \
  --output-dir artifacts/E1_timebar \
  --wandb-name E1-timebar \
  --log-portfolio-metrics --transaction-cost 0.001

# Train E1-cusum (all 5 folds) — can run in parallel
uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/E1_cusum_cv_manifest.json \
  --model conv1d --epochs 20 --seed 1337 \
  --output-dir artifacts/E1_cusum \
  --wandb-name E1-cusum \
  --log-portfolio-metrics --transaction-cost 0.001
```

### E2: Model complexity (L3) — Table 4

Trains Conv1D and optionally ResNet-LSTM. All 5 folds. Data prepared above.

```bash
# E2-conv1d (mandatory)
uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/E2_cv_manifest.json \
  --model conv1d --epochs 20 --seed 1337 \
  --output-dir artifacts/E2_conv1d \
  --wandb-name E2-conv1d \
  --log-portfolio-metrics --transaction-cost 0.001

# E2-resnet (only if E2-conv1d validation F1 > E0-logreg)
uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/E2_cv_manifest.json \
  --model resnet_lstm --epochs 30 --seed 1337 \
  --output-dir artifacts/E2_resnet \
  --wandb-name E2-resnet \
  --log-portfolio-metrics --transaction-cost 0.001
```

### E3: Selective trading / confidence thresholds (RQ3) — Table 5 + figure

No retraining. Use best E2 checkpoint, sweep thresholds {0.0, 0.55, 0.65}.

```bash
# Determine E2_BEST_CHECKPOINT from W&B (highest test Sharpe or lowest drawdown)
export E2_BEST_CHECKPOINT=artifacts/E2_conv1d/best_checkpoint.pth

# Threshold sweep at eval time (via reconcile_metrics.py)
uv run python scripts/reconcile_metrics.py \
  --checkpoint ${E2_BEST_CHECKPOINT} \
  --thresholds 0.0 0.55 0.65 \
  --output results/E3_threshold_sweep.csv \
  --wandb-project day-trading-experiments \
  --wandb-name E3-threshold-sweep
```

### E4: Meta-selection ablation (RQ4) — Table 6

**Can a learned meta-model improve trade selection beyond simple thresholds?**

Compare meta-selection ON vs OFF, and test feature importance. All 5 folds.

```bash
# E4-nometa: Every signal, fixed bet size (no meta, no Kelly)
uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/E2_cv_manifest.json \
  --model conv1d --epochs 20 --seed 1337 \
  --no-meta --bet-size fixed \
  --output-dir artifacts/E4_nometa \
  --wandb-name E4-nometa \
  --log-portfolio-metrics --transaction-cost 0.001

# E4-meta-min: Meta with minimal feature set (proba, embedding)
uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/E2_cv_manifest.json \
  --model conv1d --epochs 20 --seed 1337 \
  --meta-features proba,embedding \
  --output-dir artifacts/E4_meta_min \
  --wandb-name E4-meta-min \
  --log-portfolio-metrics --transaction-cost 0.001

# E4-meta-full: Meta with full feature set (default)
uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/E2_cv_manifest.json \
  --model conv1d --epochs 20 --seed 1337 \
  --meta-features default \
  --output-dir artifacts/E4_meta_full \
  --wandb-name E4-meta-full \
  --log-portfolio-metrics --transaction-cost 0.001
```

### E0: Final Summary (RQ1) — Table 1

**Can we translate the crypto trading pipeline to US equities?**

After E1–E4, compare your best model against trivial baselines:

```bash
# E0-majority: Predict majority class
uv run python scripts/simple_baselines.py \
  --model majority \
  --cv-manifest src/kvant/ml_framework/prepared/E1_timebar_cv_manifest.json \
  --wandb-project day-trading-experiments \
  --wandb-name E0-majority

# E0-logreg: Logistic regression (simple ML baseline)
uv run python scripts/simple_baselines.py \
  --model logreg \
  --cv-manifest src/kvant/ml_framework/prepared/E1_timebar_cv_manifest.json \
  --wandb-project day-trading-experiments \
  --wandb-name E0-logreg
```

**Results summary:** Once E1–E4 are complete, create Table 1 by comparing:
- E0-majority (worst baseline)
- E0-logreg (simple ML baseline)
- Best model from E1–E4 (e.g., E1-cusum if it beats E0-logreg, or E4-meta-full if meta adds value)

**Interpretation:** If your best model beats E0-logreg **statistically significantly** (p < 0.05), RQ1 is answered YES — the method transfers to stocks.

### Results and reporting

After E1–E4 and E0:

```bash
# Aggregate results from W&B
wandb export --project day-trading-experiments --output results/wandb_export.json

# Generate tables (aggregate per fold: mean ± std, 95% CI)
python scripts/generate_tables.py --all --output results/report_tables.csv

# Verify all numbers against W&B
python scripts/verify_results.py --report results/report_tables.csv

# Archive and commit
git add reports/ results/
git commit -m "Experiments E1-E4 + E0 complete: $(date +%Y-%m-%d)"
```

### Statistical analysis

All metrics are logged with **95% confidence intervals** computed across folds using t-distribution. To make statistical comparisons between experiments:

```bash
# RQ2: Does information-driven beat timebars?
uv run python scripts/compare_experiments.py \
  --results-a results/E1_timebar.csv \
  --results-b results/E1_cusum.csv \
  --name-a "E1-timebar" \
  --name-b "E1-cusum" \
  --metrics test_accuracy test_f1_macro portfolio/sharpe_ratio_annualized \
  --wandb-project day-trading-experiments \
  --wandb-name E1-comparison

# Model complexity: Conv1D vs ResNet
uv run python scripts/compare_experiments.py \
  --results-a results/E2_conv1d.csv \
  --results-b results/E2_resnet.csv \
  --name-a "E2-conv1d" \
  --name-b "E2-resnet" \
  --metrics test_accuracy test_f1_macro portfolio/sharpe_ratio_annualized \
  --wandb-project day-trading-experiments \
  --wandb-name E2-comparison

# RQ4: Does meta-selection add value?
uv run python scripts/compare_experiments.py \
  --results-a results/E4_nometa.csv \
  --results-b results/E4_meta_full.csv \
  --name-a "E4-nometa" \
  --name-b "E4-meta-full" \
  --metrics test_accuracy test_f1_macro portfolio/sharpe_ratio_annualized \
  --wandb-project day-trading-experiments \
  --wandb-name E4-meta-comparison

# RQ1: Does our best model beat trivial baselines?
uv run python scripts/compare_experiments.py \
  --results-a results/E0_logreg.csv \
  --results-b results/<BEST_MODEL>.csv \
  --name-a "E0-logreg (baseline)" \
  --name-b "E0-best (our method)" \
  --metrics test_accuracy test_f1_macro portfolio/sharpe_ratio_annualized \
  --wandb-project day-trading-experiments \
  --wandb-name E0-final-comparison
```

**Statistical outputs:**
- **95% CI**: logged to W&B for each metric (e.g., `{metric}/ci_lower`, `{metric}/ci_upper`)
- **Paired t-tests**: p-value < 0.05 marks statistical significance (same folds, so paired)
- **Mean difference**: effect size with 95% CI, lets you assess practical significance even when p > 0.05

**Interpretation:** A finding is "signal" only if both:
1. It is practically significant (beats the comparison by a meaningful amount)
2. It is statistically significant (p < 0.05) across folds (not just noise)

## Common commands

```bash
uv run pytest tests/
uv run ruff check .
uv run python -m kvant.ml_prepare_data.prepare_experiment
```

To prepare a fixed-threshold CUSUM ablation instead of the tuned bars/day sampler:

```bash
uv run python -m kvant.ml_prepare_data.prepare_experiment --sampler fixed_cusum --cusum-h 0.01
```

To set up the current CUSUM/barrier calibration grid, use the grid runner. It covers CUSUM thresholds
`0.005`, `0.01`, `0.02`; barrier heights `0.005`, `0.01`, `0.015`; barrier widths `60`, `120`, `180`;
and meta thresholds `0.45`, `0.50`, `0.55`, `0.60`.

```bash
# Write/print the full command plan without running it.
uv run python -m kvant.ml_framework.scripts.run_experiment_grid plan

# Prepare missing fold manifests for the 27 data configurations.
uv run python -m kvant.ml_framework.scripts.run_experiment_grid prepare --execute

# Run Conv1D first. Use --start-index/--max-runs to batch the 108 training commands.
uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-conv1d --execute --max-runs 4

# After selecting promising Conv1D configs, create and edit the ResNet-LSTM follow-up list.
uv run python -m kvant.ml_framework.scripts.run_experiment_grid write-promising-template
uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-resnet --execute
```

After preparing data, validate and train using the generated manifest:

```bash
uv run python -m kvant.ml_framework.scripts.smoke_prepared_experiment --cv-manifest src/kvant/ml_framework/prepared/sb_L_12_w180_h1.5_TBPD30_cv_manifest.json
uv run python -m kvant.ml_framework.scripts.train_experiment --baseline --epochs 3
```

## Shared training configurations

Use these checked-in `invoke` presets when team members need directly comparable runs. Pass the same explicit
`--cv-manifest` to every command for reproducibility:

```bash
# One-epoch Conv1D startup check without return simulations or checkpoint output
uv run invoke smoke --cv-manifest=src/kvant/ml_framework/prepared/<experiment>_cv_manifest.json

# Conv1D baselines, differing only by transaction cost
uv run invoke baseline-no-cost --cv-manifest=src/kvant/ml_framework/prepared/<experiment>_cv_manifest.json
uv run invoke baseline-cost --cv-manifest=src/kvant/ml_framework/prepared/<experiment>_cv_manifest.json

# Main ResNet-LSTM candidates, differing only by transaction cost
uv run invoke main-no-cost --cv-manifest=src/kvant/ml_framework/prepared/<experiment>_cv_manifest.json
uv run invoke main-cost --cv-manifest=src/kvant/ml_framework/prepared/<experiment>_cv_manifest.json
```

The baseline presets use Conv1D for 30 epochs. The main presets use the current conservative ResNet-LSTM candidate for
30 epochs. Both use `lr=0.001`, fractional Kelly at `0.25`, a `2%` maximum position fraction, and full validation
evaluation every 3 epochs. The cost variants use
`transaction_cost=0.001`; no-cost variants use `0`. The ResNet-LSTM settings are shared candidate parameters, not
claimed optimal parameters until validation experiments establish that.

Add a deliberate one-off override with `--extra-args`, for example:

```bash
uv run invoke main-cost \
  --cv-manifest=src/kvant/ml_framework/prepared/<experiment>_cv_manifest.json \
  --extra-args="--seed 7 --wandb-name main-cost-seed7"
```

For a local smoke run without cloud logging:

```bash
WANDB_MODE=offline uv run python -m kvant.ml_framework.scripts.train_experiment --exp-dir src/kvant/ml_framework/prepared/sb_L_12_w180_h1.5_TBPD30_fold00 --epochs 1 --no-return-stats
```

On CUDA-enabled machines, keep the repo metadata generic and override PyTorch locally instead of committing a CUDA index to `pyproject.toml`. A one-off option is:

```bash
uv sync --index pytorch=https://download.pytorch.org/whl/cu124
```

If you prefer a persistent machine-local override, put it in an ignored `uv.toml`.

Portfolio metrics use a budget-constrained account simulator by default. The simulator applies the same next-sampled-bar entry convention as the backtest, sizes positions from meta bet size, charges entry and exit transaction costs, tracks cash/open positions/exposure, skips trades when budget limits are exhausted, and produces an equity curve. The defaults are `$10,000` initial cash, at most `5%` equity per trade, `100%` total exposure, and at most `10` concurrent positions. Override them with:

```bash
uv run python -m kvant.ml_framework.scripts.train_experiment \
  --portfolio-initial-cash 10000 \
  --portfolio-max-position-fraction 0.02 \
  --portfolio-max-total-exposure 1.0 \
  --portfolio-max-positions 10
```

## Logging and metrics

W&B uses a compact metric set by default and runs the expensive full evaluation on epoch 1, every 3 epochs, and the
final epoch. Normal epochs log training and validation loss. Full-evaluation epochs add the most informative
validation classification, meta-label, decision, execution, and portfolio metrics. Test metrics, reduced paper-trading
diagnostics, confusion matrices, per-ticker results, and equity/profit curves are produced only for the final best model.

Use `--full-eval-every N` to change the evaluation interval.

Metrics are grouped by pipeline layer:

- `training/*`: optimization and validation loss.
- `classification/*`: primary side-model accuracy and macro F1 before meta filtering.
- `meta/*`: TAKE/PASS precision, recall, F1, and take rate.
- `decision/*`: trade rate, acted directional accuracy, and false actions on EXIT truths.
- `execution/*`: raw trade-signal counts before simulation constraints.
- `paper/*`: reduced final-best-model trade diagnostics compatible with the reference-style backtest.
- `portfolio/*`: budget-constrained return, summed calendar-year profit, drawdown, Sharpe, exposure, trade counts, and costs.

Portfolio curves are logged as `perf/portfolio_equity_curve/{split}` and `charts/portfolio_equity/{split}`. Use `portfolio/*` for final economic claims and keep `paper/*` as a diagnostic comparison.

## Notes

The preferred prepared artifacts are the non-`droptexit` `event_outcome` folds, for example `sb_L_12_w180_h1.5_TBPD30_fold00` through `fold04`. Older `droptexit` artifacts are retained on disk for comparison, but the current training entrypoint expects raw three-class event-outcome artifacts and derives side/meta labels downstream.

Prepared data, W&B runs, checkpoints, caches, generated plots, and generated architecture docs are intentionally ignored. They can be regenerated from the source code and commands above.

This is a research codebase, not a live trading system or investment recommendation.

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

Follow the exact commands below to reproduce experiments E0–E5 from [experiment_plan_8day.md](reports/experiment_plan_8day.md).

### Overview

Experiments answer four research questions via a ladder of comparisons:
- **E0** (L0): Majority class + logistic regression baselines
- **E1** (L1 vs L2): Time bars vs CUSUM, RQ1
- **E3** (L3): Model complexity (Conv1D vs ResNet-LSTM)
- **E4** (L4): Selective trading / confidence thresholds, RQ3
- **E5** (L5): Meta-selection ablation, RQ4

All use: seed `1337`, sequence length `12`, Conv1D (default), transaction cost `0.001`, all 5 folds (final runs).
Fixed triple-barrier parameters: hb=2.5%, W=240 min (per Table 1, main experimental configuration).

### Build prerequisites (B1–B5)

These must exist before running any experiment. Verify:

```bash
# B1+B2: Time-bar sampler and next-bar labeler
uv run python -m kvant.ml_framework.scripts.prepare_experiment \
  --sampler time_bar --time-bar-minutes 15 \
  --labeler next_bar \
  --cv-manifest src/kvant/ml_framework/prepared/E1_timebar_cv_manifest.json

# B3: Simple baselines script exists
test -f scripts/simple_baselines.py || echo "Missing B3: create scripts/simple_baselines.py"

# B4: --no-meta flag in train_experiment
uv run python -m kvant.ml_framework.scripts.train_experiment --help | grep -q "no-meta"

# B5: Threshold sweep at eval time (via reconcile_metrics or evaluator extension)
test -f scripts/reconcile_metrics.py || echo "Missing B5: create threshold sweep script"
```

### E0: Floors (L0) — Table 1 in report

Establishes baseline that all deep learning results must beat.

```bash
# E0-majority
uv run python scripts/simple_baselines.py \
  --model majority \
  --prepared-data src/kvant/ml_framework/prepared \
  --output results/E0_majority.csv \
  --wandb-project day-trading-experiments \
  --wandb-name E0-majority

# E0-logreg
uv run python scripts/simple_baselines.py \
  --model logreg \
  --prepared-data src/kvant/ml_framework/prepared \
  --output results/E0_logreg.csv \
  --wandb-project day-trading-experiments \
  --wandb-name E0-logreg
```

### E1: RQ1 head-to-head (L1 vs L2) — Table 2 + equity curves

Compare time bars (baseline) vs CUSUM + triple-barrier (advanced).

```bash
# Step 1: Prepare data for both arms
# L1: Time-bar baseline (15-min bars)
uv run python -m kvant.ml_framework.scripts.prepare_experiment \
  --sampler time_bar --time-bar-minutes 15 \
  --labeler next_bar \
  --output-manifest src/kvant/ml_framework/prepared/E1_timebar_cv_manifest.json

# L2: CUSUM + triple-barrier (fixed: hb=2.5%, W=240 min)
uv run python -m kvant.ml_framework.scripts.prepare_experiment \
  --sampler tuned_cusum --cusum-target-bars 30 \
  --labeler triple_barrier --barrier-height 0.025 --barrier-width 240 \
  --output-manifest src/kvant/ml_framework/prepared/E1_cusum_cv_manifest.json

# Step 2: Train E1-timebar (all 5 folds)
uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/E1_timebar_cv_manifest.json \
  --model conv1d --epochs 20 --seed 1337 \
  --output-dir artifacts/E1_timebar \
  --wandb-name E1-timebar \
  --log-portfolio-metrics --transaction-cost 0.001

# Step 3: Train E1-cusum (all 5 folds)
uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/E1_cusum_cv_manifest.json \
  --model conv1d --epochs 20 --seed 1337 \
  --output-dir artifacts/E1_cusum \
  --wandb-name E1-cusum \
  --log-portfolio-metrics --transaction-cost 0.001
```

### E3: Model complexity (L3) — Table 4

Trains Conv1D and optionally ResNet-LSTM. All 5 folds. Uses fixed triple-barrier parameters.

```bash
# Prepare data (fixed: h=0.5%, w=120 min)
uv run python -m kvant.ml_framework.scripts.prepare_experiment \
  --sampler tuned_cusum --cusum-target-bars 30 \
  --labeler triple_barrier --barrier-height 0.025 --barrier-width 240 \
  --output-manifest src/kvant/ml_framework/prepared/E3_cv_manifest.json

# E3-conv1d (mandatory)
uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/E3_cv_manifest.json \
  --model conv1d --epochs 20 --seed 1337 \
  --output-dir artifacts/E3_conv1d \
  --wandb-name E3-conv1d \
  --log-portfolio-metrics --transaction-cost 0.001

# E3-resnet (only if E3-conv1d validation F1 > E0-logreg)
uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/E3_cv_manifest.json \
  --model resnet_lstm --epochs 30 --seed 1337 \
  --output-dir artifacts/E3_resnet \
  --wandb-name E3-resnet \
  --log-portfolio-metrics --transaction-cost 0.001
```

### E4: Selective trading / confidence thresholds (RQ3) — Table 5 + figure

No retraining. Use best E3 checkpoint, sweep thresholds {0.0, 0.55, 0.65}.

```bash
# Determine E3_BEST_CHECKPOINT from W&B (highest test Sharpe or lowest drawdown)
export E3_BEST_CHECKPOINT=artifacts/E3_conv1d/best_checkpoint.pth

# Threshold sweep at eval time (via B5 reconcile_metrics or evaluator extension)
for threshold in 0.0 0.55 0.65; do
  uv run python -m kvant.ml_framework.scripts.evaluate_checkpoint \
    --checkpoint ${E3_BEST_CHECKPOINT} \
    --confidence-threshold ${threshold} \
    --output results/E4_threshold_${threshold}.csv \
    --wandb-name E4-threshold-${threshold}
done
```

### E5: Meta-selection ablation (RQ4) — Table 6

Compare meta-selection ON vs OFF. All 5 folds.

```bash
# E5-nometa: Every signal, fixed bet size (no meta, no Kelly)
uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/E3_cv_manifest.json \
  --model conv1d --epochs 20 --seed 1337 \
  --no-meta --bet-size fixed \
  --output-dir artifacts/E5_nometa \
  --wandb-name E5-nometa \
  --log-portfolio-metrics --transaction-cost 0.001

# E5-meta-min: Meta with minimal feature set (proba, embedding)
uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/E3_cv_manifest.json \
  --model conv1d --epochs 20 --seed 1337 \
  --meta-features proba,embedding \
  --output-dir artifacts/E5_meta_min \
  --wandb-name E5-meta-min \
  --log-portfolio-metrics --transaction-cost 0.001

# E5-meta-full: Meta with full feature set (default)
uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest src/kvant/ml_framework/prepared/E3_cv_manifest.json \
  --model conv1d --epochs 20 --seed 1337 \
  --meta-features default \
  --output-dir artifacts/E5_meta_full \
  --wandb-name E5-meta-full \
  --log-portfolio-metrics --transaction-cost 0.001
```

### Results and reporting

After all runs:

```bash
# Aggregate results from W&B
wandb export --project day-trading-experiments --output results/wandb_export.json

# Generate tables (aggregate per fold: mean ± std)
python scripts/generate_tables.py --all --output results/report_tables.csv

# Verify all numbers against W&B
python scripts/verify_results.py --report results/report_tables.csv

# Archive and commit
git add reports/ results/
git commit -m "Experiment E0-E5 complete: $(date +%Y-%m-%d)"
```

### Statistical analysis

All metrics are logged with **95% confidence intervals** computed across folds using t-distribution. To make statistical comparisons between experiments:

```bash
# Compare E1-timebar vs E1-cusum with paired t-tests
uv run python scripts/compare_experiments.py \
  --results-a results/E1_timebar.csv \
  --results-b results/E1_cusum.csv \
  --name-a "E1-timebar" \
  --name-b "E1-cusum" \
  --metrics test_accuracy test_f1_macro paper/sharpe_ratio_annualized \
  --wandb-project day-trading-experiments \
  --wandb-name E1-comparison

# Repeat for other comparisons:
# - E0-majority vs E0-logreg
# - E3-conv1d vs E3-resnet
# - E1-cusum vs E3-conv1d (RQ2: economic value of model complexity)
```

**Statistical outputs:**
- **95% CI**: logged to W&B for each metric (e.g., `{metric}/ci_lower`, `{metric}/ci_upper`)
- **Paired t-tests**: p-value < 0.05 marks statistical significance (same folds, so paired)
- **Mean difference**: effect size with 95% CI, lets you assess practical significance even when p > 0.05

**Interpretation:** A metric is "signal" only if both:
1. It beats the floor (E0) by a meaningful amount (practical significance)
2. The difference is statistically significant (p < 0.05) across folds (not just noise)

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

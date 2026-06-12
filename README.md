# kvant

`kvant` is a quant research project for short-horizon US equity trading on minute-level OHLCV data. The current approach follows a Lopez de Prado-inspired side-plus-meta-labeling workflow: prepared artifacts keep the raw three-class triple-barrier event outcome (`down`, `exit`, `up`), the training pipeline derives a binary primary side target (`down`, `up`), and a meta-label layer decides whether a predicted side should be traded or abstained from.

The project is built around reproducible walk-forward folds. Data preparation computes technical features on minute bars before sampling, fits all learned preparation state on train data only, and then prepares train/validation/test artifacts with sampled OHLCV, labels, metadata, and diagnostics. Training consumes those prepared artifacts with PyTorch models, validates the artifact contract before running, logs metrics through W&B, and evaluates model quality, decision quality, paper-style trade diagnostics, and a stricter budget-constrained portfolio simulator under transaction costs.

## Current pipeline

1. Download/cache minute OHLCV shards from Hugging Face and build walk-forward splits of top-volume US equities.
2. Fit a train-only `FixedThresholdCUSUMBarSampler` using the paper-aligned default threshold `0.02`.
3. Compute intraday technical features on minute bars before sampling and standardize them from train data. Optional train-only feature selection remains available as an ablation, but it is not part of the default baseline.
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

## Common commands

```bash
uv run pytest tests/
uv run ruff check .
uv run python -m kvant.ml_prepare_data.prepare_experiment
```

To prepare a specific fixed-threshold CUSUM configuration explicitly:

```bash
uv run python -m kvant.ml_prepare_data.prepare_experiment --sampler fixed_cusum --cusum-h 0.02
```

The default preparation entrypoint now uses a paper-aligned baseline:

- fixed CUSUM threshold `0.02`
- Triple Barrier height `5%`
- vertical barrier `24` sampled periods
- lookback window `96`
- walk-forward warmup `4` quarters

To set up the current CUSUM/barrier calibration grid, use the grid runner. It covers CUSUM thresholds
`0.01`, `0.02`, `0.03`; barrier heights `0.025`, `0.05`, `0.06`; a fixed vertical barrier of `24` sampled periods;
and meta thresholds `0.45`, `0.50`, `0.55`, `0.60`.

```bash
# Write/print the full command plan without running it.
uv run python -m kvant.ml_framework.scripts.run_experiment_grid plan

# Prepare missing fold manifests for the 9 data configurations.
uv run python -m kvant.ml_framework.scripts.run_experiment_grid prepare --execute

# Run Conv1D first. Use --start-index/--max-runs to batch the 36 training commands.
uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-conv1d --execute --max-runs 4

# After selecting promising Conv1D configs, create and edit the ResNet-LSTM follow-up list.
uv run python -m kvant.ml_framework.scripts.run_experiment_grid write-promising-template
uv run python -m kvant.ml_framework.scripts.run_experiment_grid train-resnet --execute
```

After preparing data, validate and train using the generated manifest:

```bash
uv run python -m kvant.ml_framework.scripts.smoke_prepared_experiment --cv-manifest src/kvant/ml_framework/prepared/sb_L_96_wp24_h5_fixedCUSUM0.02_cv_manifest.json
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
WANDB_MODE=offline uv run python -m kvant.ml_framework.scripts.train_experiment --exp-dir src/kvant/ml_framework/prepared/sb_L_96_wp24_h5_fixedCUSUM0.02_fold00 --epochs 1 --no-return-stats
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

The preferred prepared artifacts are the non-`droptexit` `event_outcome` folds produced by the current paper-aligned defaults, for example `sb_L_96_wp24_h5_fixedCUSUM0.02_fold00`. Older `droptexit` and pre-reset artifacts are retained on disk for comparison, but the current training entrypoint expects raw three-class event-outcome artifacts and derives side/meta labels downstream.

Prepared data, W&B runs, checkpoints, caches, generated plots, and generated architecture docs are intentionally ignored. They can be regenerated from the source code and commands above.

This is a research codebase, not a live trading system or investment recommendation.

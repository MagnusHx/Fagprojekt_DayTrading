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

After preparing data, validate and train using the generated manifest:

```bash
uv run python -m kvant.ml_framework.scripts.smoke_prepared_experiment --cv-manifest src/kvant/ml_framework/prepared/sb_L_12_w180_h1.5_TBPD30_cv_manifest.json
uv run python -m kvant.ml_framework.scripts.train_experiment --baseline --epochs 3
```

For a local smoke run without cloud logging:

```bash
WANDB_MODE=offline uv run python -m kvant.ml_framework.scripts.train_experiment --exp-dir src/kvant/ml_framework/prepared/sb_L_12_w180_h1.5_TBPD30_fold00 --epochs 1 --no-return-stats
```

Portfolio metrics use a budget-constrained account simulator by default. The simulator applies the same next-sampled-bar entry convention as the backtest, sizes positions from meta bet size, charges entry and exit transaction costs, tracks cash/open positions/exposure, skips trades when budget limits are exhausted, and produces an equity curve. The defaults are `$10,000` initial cash, at most `5%` equity per trade, `100%` total exposure, and at most `10` concurrent positions. Override them with:

```bash
uv run python -m kvant.ml_framework.scripts.train_experiment \
  --portfolio-initial-cash 10000 \
  --portfolio-max-position-fraction 0.05 \
  --portfolio-max-total-exposure 1.0 \
  --portfolio-max-positions 10
```

## Logging and metrics

W&B logging is grouped by pipeline layer so results are easier to debug:

- `cls/*`: primary side-model performance before meta filtering.
- `meta/*`: TAKE/PASS quality and accept threshold for the meta-label layer.
- `decision/*`: acted/abstained behavior after meta filtering.
- `execution/*`: raw trade signals, overlap suppression, executed trades, and active time.
- `paper/*`: diagnostic trade-level economics compatible with the reference-style backtest.
- `portfolio/*`: budget-constrained account metrics such as final balance, return, drawdown, Sharpe, exposure, executed trades, skipped-budget trades, and transaction costs.

Portfolio curves are logged as `perf/portfolio_equity_curve/{split}` and `charts/portfolio_equity/{split}`. Use `portfolio/*` for final economic claims and keep `paper/*` as a diagnostic comparison.

## Notes

The preferred prepared artifacts are the non-`droptexit` `event_outcome` folds, for example `sb_L_12_w180_h1.5_TBPD30_fold00` through `fold04`. Older `droptexit` artifacts are retained on disk for comparison, but the current training entrypoint expects raw three-class event-outcome artifacts and derives side/meta labels downstream.

Prepared data, W&B runs, checkpoints, caches, generated plots, and generated architecture docs are intentionally ignored. They can be regenerated from the source code and commands above.

This is a research codebase, not a live trading system or investment recommendation.

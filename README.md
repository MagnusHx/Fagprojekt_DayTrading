# kvant

`kvant` is a quant research project for short-horizon US equity trading on minute-level OHLCV data. The current approach follows a Lopez de Prado-inspired side-plus-meta-labeling workflow: prepared artifacts keep the raw three-class triple-barrier event outcome (`down`, `exit`, `up`), the training pipeline derives a binary primary side target (`down`, `up`), and a meta-label layer decides whether a predicted side should be traded or abstained from.

The project is built around reproducible walk-forward folds. Data preparation fits the sampler, feature engineering, and feature selection on train data only, then prepares train/validation/test artifacts with sampled OHLCV, labels, metadata, and diagnostics. Training consumes those prepared artifacts with PyTorch models, validates the artifact contract before running, logs metrics through W&B, and evaluates both model quality and trading-oriented behavior under transaction costs.

## Current pipeline

1. Download/cache minute OHLCV shards from Hugging Face and build walk-forward splits of top-volume US equities.
2. Fit a train-only `TunedCUSUMBarSampler` targeting a configurable bars-per-day density.
3. Compute intraday technical features, standardize them from train data, and select features for the primary side task with train-only F-score selection.
4. Label sampled bars with triple-barrier outcomes and persist the canonical `event_outcome` label space.
5. Train a primary side classifier on actionable `down/up` events while retaining `exit` rows for decision and abstention evaluation.
6. Fit/evaluate the meta-label decision policy and report side, meta, decision, execution, and economics metrics.

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

After preparing data, validate and train using the generated manifest:

```bash
uv run python -m kvant.ml_framework.scripts.smoke_prepared_experiment --cv-manifest src/kvant/ml_framework/prepared/sb_L_12_w180_h1.5_TBPD30_cv_manifest.json
uv run python -m kvant.ml_framework.scripts.train_experiment --baseline --epochs 3
```

For a local smoke run without cloud logging:

```bash
WANDB_MODE=offline uv run python -m kvant.ml_framework.scripts.train_experiment --exp-dir src/kvant/ml_framework/prepared/sb_L_12_w180_h1.5_TBPD30_fold00 --epochs 1 --no-return-stats
```

## Notes

The preferred prepared artifacts are the non-`droptexit` `event_outcome` folds, for example `sb_L_12_w180_h1.5_TBPD30_fold00` through `fold04`. Older `droptexit` artifacts are retained on disk for comparison, but the current training entrypoint expects raw three-class event-outcome artifacts and derives side/meta labels downstream.

Prepared data, W&B runs, checkpoints, caches, generated plots, and generated architecture docs are intentionally ignored. They can be regenerated from the source code and commands above.

This is a research codebase, not a live trading system or investment recommendation.

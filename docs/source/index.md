# kvant

`kvant` is a short-horizon equity trading research project built around a Lopez de Prado-inspired side-plus-meta-labeling pipeline.

The current workflow prepares walk-forward experiment folds from minute OHLCV data, computes features before event sampling, keeps raw three-class triple-barrier event labels (`down`, `exit`, `up`), trains a binary primary side model on actionable outcomes, and evaluates a meta-label decision layer that can abstain from trades. Metrics are organized around the full research question: side-model quality, meta-label acceptance quality, acted/abstained decisions, execution behavior, diagnostic paper economics, and a budget-constrained portfolio account.

## Pipeline overview

1. Cache Hugging Face OHLCV shards and construct top-volume ticker walk-forward splits.
2. Compute intraday features before sampling, then fit fixed-threshold CUSUM sampling and feature standardization on train data only.
3. Persist prepared fold artifacts with features, labels, sampled OHLCV, label metadata, and diagnostics.
4. Validate the prepared artifact contract before any training run.
5. Train a PyTorch side classifier and evaluate the side-plus-meta decision policy.
6. Enforce next-sampled-bar trade entry and purged/embargoed label intervals across split boundaries.
7. Log reproducibility metadata, split summaries, confusion matrices, per-ticker diagnostics, paper metrics, and portfolio-account metrics.

## Logging overview

Training runs publish metrics in layer-specific namespaces:

- `training/*` for optimization and validation loss.
- `classification/*` for primary side-model learning behavior.
- `meta/*` for TAKE/PASS filtering quality.
- `decision/*` for final acted and abstained predictions.
- `execution/*` for raw signal counts.
- `paper/*` for reduced final-best-model trade diagnostics.
- `portfolio/*` for return, summed calendar-year profit, exposure, drawdown, and account constraints.

The portfolio equity curve is logged as `perf/portfolio_equity_curve/{split}` with a companion line chart at `charts/portfolio_equity/{split}`.

## Useful commands

```bash
uv run pytest tests/
uv run ruff check .
uv run python -m kvant.ml_prepare_data.prepare_experiment
```

After preparing data, validate or train from the generated prepared manifest:

```bash
uv run python -m kvant.ml_framework.scripts.smoke_prepared_experiment --cv-manifest src/kvant/ml_framework/prepared/sb_L_96_wp24_h5_fixedCUSUM0.02_cv_manifest.json
WANDB_MODE=offline uv run python -m kvant.ml_framework.scripts.train_experiment --exp-dir src/kvant/ml_framework/prepared/sb_L_96_wp24_h5_fixedCUSUM0.02_fold00 --epochs 1 --no-return-stats
```

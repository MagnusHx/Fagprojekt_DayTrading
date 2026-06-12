# Latest Experiment Analysis

Last updated: 2026-06-12

## Current Status

The repository defaults have been reset to the paper-aligned baseline described in
`references/s40854-025-00866-w.pdf` more closely than before:

- fixed-threshold CUSUM with `h=0.02`
- lookback `L=96`
- Triple Barrier vertical horizon of `24` sampled periods
- symmetric horizontal barriers of `5%`
- walk-forward warmup of `4` quarters
- full default feature set with no default feature selection

The latest code also treats the sampled bar as the signal and enters on the next sampled bar open in both labeling and
backtest code. This makes the default path live-safe relative to aggregated CUSUM bars.

## Important Caveat

There is not yet a completed local training run on disk that uses the new default prepared artifacts
`sb_L_96_wp24_h5_fixedCUSUM0.02_*`.

The latest completed local run still appears to be the older legacy setup:

- W&B directory: `wandb/run-20260427_213934-bm3y428m`
- Model: `resnet_lstm`
- Epochs: `30`
- Manifest: `src/kvant/ml_framework/prepared/sb_L_12_w180_h1.5_fixedCUSUM0.01_cv_manifest.json`
- Transaction cost: `0.001`

That run should now be treated as historical context only. It does not represent the current default methodology.

## What This Means For Interpretation

Older experiment results are useful only as evidence that the previous baseline was weak. They should not be used as
the main evaluation of the project after the methodology reset, because they differ on:

- CUSUM threshold
- Triple Barrier width and height
- lookback length
- warmup window
- prepared artifact naming and metadata contracts
- default meta-feature setup and current portfolio evaluation stack

## Current Recommended Baseline

Regenerate folds and rerun the baseline before drawing new conclusions:

```bash
uv run python -m kvant.ml_prepare_data.prepare_experiment
uv run python -m kvant.ml_framework.scripts.smoke_prepared_experiment --cv-manifest src/kvant/ml_framework/prepared/sb_L_96_wp24_h5_fixedCUSUM0.02_cv_manifest.json
WANDB_MODE=offline uv run python -m kvant.ml_framework.scripts.train_experiment --baseline --epochs 3
```

## Current Search Space

The small paper-aligned grid runner now targets:

| Parameter | Values |
| --- | --- |
| CUSUM threshold | `0.01`, `0.02`, `0.03` |
| Barrier height | `0.025`, `0.05`, `0.06` |
| Barrier width | `24` sampled periods |
| Model sequence | `conv1d` first, then `resnet_lstm` on promising settings |

The current promising template in `reports/promising_grid_configs.json` prioritizes:

- `cusum_h=0.02`, `barrier_height=0.05`, `barrier_width_periods=24`
- `cusum_h=0.03`, `barrier_height=0.06`, `barrier_width_periods=24`

## Bottom Line

The project reports should now be read as follows:

- the code baseline is paper-aligned more closely than before
- the latest completed local experiment is legacy
- a fresh baseline run is still needed before making updated performance claims

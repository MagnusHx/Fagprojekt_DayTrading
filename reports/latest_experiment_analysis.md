# Latest Experiment Analysis

Last updated: 2026-04-27

## Run Identified

The latest completed local experiment is:

- W&B directory: `wandb/run-20260427_213934-bm3y428m`
- Model: `resnet_lstm`
- Epochs: `30`
- Manifest: `src/kvant/ml_framework/prepared/sb_L_12_w180_h1.5_fixedCUSUM0.01_cv_manifest.json`
- Transaction cost: `0.001`
- Meta features in this run: `['proba', 'embedding']`
- Meta accept threshold: `0.6`
- Checkpoint metric: `val/meta/f1`

Important caveat: this run predates the newest portfolio-account evaluation and enriched default meta-feature setup. It does not contain `portfolio/*` metrics, and it did not use the newer prepared volatility/return aliases, rolling ticker win-rate features, prediction margin/entropy, or time-since-event meta features.

## Cross-Fold Summary

| Metric | Mean | Interpretation |
| --- | ---: | --- |
| `best/val/meta/f1` | `0.0907` | Very weak TAKE/PASS selection on validation. |
| `best/test/meta/f1` | `0.1067` | Weak out-of-sample meta-label behavior. |
| `best/test/cls/f1_macro` | `0.4407` | Side model is close to noisy/barely useful. |
| `best/test/accuracy` | `0.5023` | Directional accuracy is approximately random. |
| `best/test/decision/trade_signal_rate` | `0.0585` | The threshold makes the model trade only about 6% of events. |
| `best/test/execution/n_executed_trades` | `186` | Much fewer trades than the Conv1D run, increasing metric instability. |
| `best/test/execution/executed_trade_hit_rate_pct` | `54.99%` | Hit rate is above 50%, but not enough after costs. |
| `best/test/paper/executed_trade_net_return_avg_pct` | `-0.0380%` | Average executed trade loses after costs. |
| `best/test/paper/executed_trade_net_return_total_pct` | `-7.62%` | Total trade-level net return is negative. |
| `best/test/paper/annual_net_profit_loss_pct` | `-24.23%` | Annualized diagnostic return is negative. |
| `best/test/paper/sharpe_ratio_annualized` | `-2.56` | Risk-adjusted result is poor. |
| `best/test/paper/max_drawdown_pct` | `9.30%` | Drawdown is non-trivial despite low take rate. |

## Comparison Against Previous Conv1D Run

Previous local Conv1D run:

- W&B directory: `wandb/run-20260427_185333-guiy48yc`
- Model: `conv1d`
- Epochs: `20`
- Meta threshold: `0.5`
- Same fixed-CUSUM manifest and transaction cost.

| Metric | Conv1D mean | ResNet-LSTM mean | Comment |
| --- | ---: | ---: | --- |
| `best/val/meta/f1` | `0.5101` | `0.0907` | ResNet-LSTM meta selection is much worse under the current threshold/setup. |
| `best/test/meta/f1` | `0.5081` | `0.1067` | Same pattern out of sample. |
| `best/test/trade_signal_rate` | `0.5035` | `0.0585` | ResNet-LSTM trades far less. |
| `best/test/n_executed_trades` | `1464.6` | `186.0` | ResNet-LSTM result is based on fewer trades. |
| `best/test/net_return_avg_pct` | `-0.0058%` | `-0.0380%` | ResNet-LSTM has worse average net trade return. |
| `best/test/annual_net_profit_loss_pct` | `-28.49%` | `-24.23%` | Both are negative; ResNet-LSTM only looks slightly less bad because it trades less. |
| `best/test/sharpe_ratio_annualized` | `-6.20` | `-2.56` | Lower activity softens the Sharpe damage but does not create edge. |

## Diagnosis

The latest experiment does not show an investable signal. The main issue is not only model architecture. The side model remains near random out of sample, while the meta layer becomes extremely conservative at threshold `0.6` and still does not select profitable trades after transaction costs.

Most likely causes:

1. The fixed 1% CUSUM plus `width=180`, `height=1.5%` label setup may be too coarse for minute-level US equities.
2. The ResNet-LSTM is probably overfitting weak/noisy side labels: train loss improves, but validation meta F1 remains very low.
3. The meta threshold of `0.6` is too restrictive for this probability distribution; recall collapses while precision does not rise enough.
4. Transaction costs dominate the edge. Average gross returns are small and inconsistent, so even a modest cost turns the strategy negative.
5. The latest run did not use the newly added meta features or the new portfolio simulator, so it is not the final version of the project approach.

## Recommended Next Step

Do not spend the next iteration on a larger neural architecture. The next most valuable improvement is a label/sampling calibration experiment:

1. Regenerate corrected folds using a small grid of Triple Barrier settings and fixed-CUSUM thresholds.
2. Keep the model simple first, preferably Conv1D, so the comparison isolates data/label quality.
3. Evaluate each configuration with the enriched meta features and the new `portfolio/*` metrics.
4. Select a baseline only if it improves out-of-sample portfolio return, drawdown, and skipped-budget behavior, not only meta F1.

Suggested first grid:

| Parameter | Values |
| --- | --- |
| CUSUM threshold | `0.005`, `0.01`, `0.02` |
| Barrier height | `0.005`, `0.01`, `0.015` |
| Barrier width | `60`, `120`, `180` minutes |
| Model | `conv1d` first, then `resnet_lstm` only on promising settings |
| Meta threshold | tune on validation, compare `0.45`, `0.50`, `0.55`, `0.60` |

The project should aim to find a label/sampling regime where gross trade return is clearly positive before costs. If gross edge is weak, model and meta changes will mostly rearrange negative-cost trades.

# Metric Debugging Contract

This project now treats metrics as four semantic layers:

- `split/cls/*`: classifier quality before abstention
- `split/decision/*`: thresholding and abstention behavior
- `split/execution/*`: signal-to-executed-trade translation
- `split/paper/*`: economic outcomes

The code-level source of truth is [metric_registry.py](/home/magnus/repositories/Fagprojekt_DayTrading/Fagprojekt_DayTrading/src/kvant/ml_framework/train/metric_registry.py:1).

## Primary Debug Metrics

### Learning
- `{split}/cls/accuracy`
- `{split}/cls/f1_macro`
- `{split}/cls/precision_class_{label}`
- `{split}/cls/recall_class_{label}`
- normalized confusion matrices

### Decision
- `{split}/trade_direction_threshold`
- `{split}/decision/abstained_prediction_rate_pct`
- `{split}/decision/acted_prediction_accuracy`
- `{split}/decision/directional_acted_accuracy`
- `{split}/trade_signal_rate`

### Execution
- `{split}/execution/n_trade_signals_raw`
- `{split}/execution/n_trade_signals_skipped_overlap`
- `{split}/paper/n_executed_trades`
- `{split}/paper/share_time_active_pct`
- `{split}/paper/executed_trade_hit_rate_pct`

### Economics
- `{split}/paper/executed_trade_gross_return_avg_pct`
- `{split}/paper/executed_trade_net_return_avg_pct`
- `{split}/paper/transaction_cost_total_pct`
- `{split}/paper/profitable_transactions_pct`
- `{split}/paper/annual_net_profit_loss_pct`
- `{split}/paper/sharpe_ratio_annualized`
- `{split}/paper/max_drawdown_pct`

### Direction Split
- `{split}/paper/long_n_executed_trades`
- `{split}/paper/short_n_executed_trades`
- `{split}/paper/long_hit_rate_pct`
- `{split}/paper/short_hit_rate_pct`
- `{split}/paper/long_net_return_avg_pct`
- `{split}/paper/short_net_return_avg_pct`

## Legacy Metrics

`paper/accuracy_all_predictions` is retained for backward compatibility, but it is not a primary learning metric. In directional-binary runs, abstentions can make it look artificially poor.

`trade_action_probability_*` is structurally uninformative in directional-binary runs because `p_act = 1` by construction. Use `split/decision/trade_action_probability_informative` to decide whether to show those panels.

## Golden-Fold Reconciliation Workflow

1. Train with the default best-checkpoint bundle output or provide `--checkpoint-out-dir`.
2. Use [reconcile_metrics.py](/home/magnus/repositories/Fagprojekt_DayTrading/Fagprojekt_DayTrading/src/kvant/ml_framework/scripts/reconcile_metrics.py:1) with:
   - `--bundle`
   - optional `--wandb-summary`
   - optional `--summary-prefix`
3. Compare the offline recomputed scalars against the W&B summary.

Example:

```bash
uv run python -m kvant.ml_framework.scripts.reconcile_metrics \
  --bundle artifacts/checkpoints/my-run-fold04-best.ckpt.pt \
  --wandb-summary wandb/latest-run/files/wandb-summary.json \
  --summary-prefix fold04/best \
  --output artifacts/metric_debug/fold04-reconciliation.json
```

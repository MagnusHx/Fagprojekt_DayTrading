# Reference Method and Backtest Review

## Scope

This review compares the current `kvant` pipeline with the methods described in:

- `references/s40854-025-00866-w.pdf`
- `references/lopez2018.pdf`

It focuses on why the current project may be producing weak trading results and whether the current backtest is methodologically correct.

## Short Answer

The project contains several sensible pieces from the references: CUSUM-style event sampling, Triple Barrier labeling, train-only fitted preprocessing, side/meta-label separation, transaction costs, and non-overlapping same-ticker trade execution.

However, the backtest is not fully correct in its current form. The largest issue is execution timing with aggregated CUSUM bars. The sampled bar is timestamped at the end of the CUSUM segment, but the labeler and backtest enter at that sampled bar's `open`, which is the first minute of the segment. That open price existed before the model's signal timestamp. This creates a time-consistency problem and can make label/backtest economics different from what could be traded live.

A second important issue is split-boundary leakage. Triple Barrier labels span intervals, but split indices are assigned only by event start timestamp. Training examples close to a validation/test boundary can have labels whose barrier outcome is determined using future validation/test prices. Lopez de Prado's purging/embargo guidance is directly relevant here.

## High-Priority Findings

| Priority | Finding | Impact | Where |
| --- | --- | --- | --- |
| P0 | Aggregated CUSUM bars enter at historical segment open while signal timestamp is segment end | Backtest/labels are not live-trade-consistent | `src/kvant/ml_prepare_data/samplers/sampler_cumsum.py`, `src/kvant/labelling.py`, `src/kvant/ml_framework/train/backtest.py` |
| P0 | Split indices use event start timestamp but do not purge events whose label window crosses split boundaries | Leakage around train/val/test boundaries | `src/kvant/ml_prepare_data/prepare_experiment.py` |
| P1 | Current CUSUM is tuned to bars/day, not fixed-threshold CUSUM as in the 2025 paper's main sensitivity | Paper comparison is weak; may tune away volatility signal | `TunedCUSUMBarSampler` |
| P1 | Current barrier settings are not adapted to US equities and likely produce too many exits | Weak directional sample quality and poor economics | `TripleBarrierLabeler(width=180, height=0.015)` |
| P1 | Model/training stack is weaker than the reference setup | Lower signal quality | Conv1D baseline, limited HPO, no multi-seed ensemble |
| P2 | Feature timing differs from the 2025 paper | Not necessarily wrong, but not paper-faithful | Project computes features before sampling; paper describes computing indicators after sampling |
| P2 | Annualization and portfolio assumptions are simplified | Metrics are useful diagnostics, not live-trading PnL | `compute_paper_trading_metrics` |

## Reference Methods That Matter

### 1. Event-Based Sampling

Lopez de Prado motivates event-based sampling because ML should learn from relevant events, not arbitrary clock times. The 2025 paper tests time bars, dollar bars, volume bars, CUSUM, and range bars. Its strongest conclusion is that CUSUM plus Triple Barrier labeling was the most robust family in their crypto experiments.

Current project status:

- Uses CUSUM-style sampling.
- Tunes `h` per ticker to target a bars-per-day count.
- Does not yet implement fixed-threshold CUSUM sweeps such as 1%, 2%, 3%.
- Does not yet implement range bars, dollar bars, or volume bars as ablations.

Suggested work:

1. Add fixed-threshold CUSUM as a separate sampler mode.
2. Compare fixed CUSUM thresholds against the current tuned-bars/day sampler.
3. Add range bars as the closest alternative to CUSUM.
4. Treat dollar/volume bars as lower priority for US equities unless reliable volume/dollar-volume adjustments are handled.

### 2. Triple Barrier Labeling

Both references support Triple Barrier labeling as more trading-realistic than next-bar labeling. It defines a trade by profit-taking, stop-loss, and vertical time exit.

Current project status:

- Uses Triple Barrier labels.
- Keeps raw `down/exit/up` event outcomes in prepared artifacts.
- Uses a side model plus meta-label decision layer.

Main concern:

- Barrier parameters are probably not calibrated to the US equity minute setting.
- `height=1.5%` and `width=180min` can be too far for many liquid large-cap intraday windows, leading to many time exits and weak actionable signal.

Suggested work:

1. Sweep `height` and `width` on equities.
2. Report tradeable share, not only accuracy.
3. Prefer configurations that balance tradeable share, executed trade hit rate, net return, and drawdown.
4. Try dynamic barriers based on recent volatility, but treat this as an ablation because the 2025 paper found dynamic barriers did not automatically improve results.

### 3. Meta-Labeling

Lopez de Prado's meta-labeling idea is to separate side from size/action. The primary model proposes a side; the meta model decides whether to act.

Current project status:

- Implements a logistic meta-labeler.
- Uses side-model probabilities/embeddings and optional prepared features.
- Uses validation meta F1 as the checkpoint metric.

Concerns:

- The primary side model may be too weak for meta-labeling to rescue.
- The meta model is only as good as the realized-return metadata and backtest timing. If entry timing is wrong, meta labels are also affected.

Suggested work:

1. Fix label/backtest timing first.
2. Then evaluate meta-label calibration and threshold stability.
3. Compare meta-labeling against a simple fixed probability band, such as long if `P(up) > 0.60`, short if `P(up) < 0.40`.

### 4. Purging and Embargo

Lopez de Prado emphasizes that labels derived from overlapping time intervals require purged and embargoed splits. This applies directly to Triple Barrier labels.

Current project status:

- Uses chronological train/validation/test splits.
- Builds split indices by event timestamp only.
- Does not check whether an event's `bar_close_time` crosses into the next split.

Why this matters:

- A training event that starts shortly before validation can have a Triple Barrier outcome determined by validation prices.
- A validation event shortly before test can be determined by test prices.
- This contaminates model selection and backtest interpretation.

Suggested work:

1. While building indices, require the label close time to remain inside the same split as the event start.
2. Drop or embargo events near split boundaries whose label interval overlaps the next split.
3. Add validation tests that assert no train label interval intersects validation/test windows.

## Is the Backtest Correct?

Not fully.

The code does several good things:

- Simulates long and short trades.
- Applies transaction cost on entry and exit.
- Skips overlapping trades for the same ticker.
- Allows concurrent trades across different tickers.
- Computes portfolio compounding, daily returns, Sharpe, drawdown, and active time.

But there are correctness risks that should be fixed before trusting the numbers.

### Backtest Issue 1: Aggregated Bar Entry Uses the Past

`_aggregate_ohlcv_segments` builds a sampled bar with:

- index = timestamp of the segment end
- open = first open in the segment
- high = max high in the segment
- low = min low in the segment
- close = final close in the segment

The model receives features sampled at the segment end timestamp. At that moment, the segment open is already historical. But both the labeler and the backtest use the sampled bar `open` as the entry price.

That means the simulated trade can enter at a price before the signal exists.

Recommended fix:

- For event timestamp `t`, enter at the next tradable price after `t`.
- Conservative options:
  - enter at next sampled bar open,
  - enter at current sampled bar close,
  - or store original minute bars and enter at next minute open after the event timestamp.
- The labeler and backtest must use the same entry convention.

### Backtest Issue 2: Same-Bar Barrier Hits

The backtest checks barrier hits starting at `entry_pos`, including the entry bar. With aggregated bars, the high/low of that entry bar may include prices before the signal timestamp.

Recommended fix:

- If using current bar close as entry, barrier search should begin after the entry timestamp.
- If using next bar open as entry, barrier search should begin at the next bar.
- Do not let the entry bar's pre-signal high/low trigger a barrier.

### Backtest Issue 3: Split Boundary Leakage

The preparation code processes full train+validation+test history per ticker and labels sampled bars. That is acceptable only if index construction prevents training labels from using future split prices.

Current status: fixed in the preparation pipeline. Split index construction now uses label intervals, purges examples that would close outside their split, and applies an embargo before the next boundary.

Implemented fix:

- Store `bar_open_time` and `bar_close_time` for every label.
- Include an example in a split only if both open and close are inside that split.
- Apply an embargo around split boundaries so near-boundary labels do not leak future information.

### Backtest Issue 4: Annualization is Useful but Simplified

The annualized P/L compounds only at trade exits and fills daily portfolio values by last known value. This is reasonable as a diagnostic, but it is not a full execution simulator.

Current status: the simplified `paper/*` metrics are still useful diagnostics, and a stricter `portfolio/*` simulator has been added for final economic claims.

Remaining limitations:

- No slippage.
- No spread model beyond fixed transaction cost.
- Portfolio marks equity at trade entry/exit events rather than every raw minute.
- Daily Sharpe can be unstable when trades are sparse.

Implemented fix:

- Keep current `paper/*` metrics as diagnostics.
- Use the stricter `portfolio/*` metrics for final claims: explicit cash, exposure, concurrent positions, position sizing, transaction costs, skipped-budget trades, and an equity curve.

## Why Results May Be Poor

The most likely reasons are:

1. The project is not a close reproduction of the 2025 paper. It uses US equities, NYSE hours, minute OHLCV data, tuned bars/day CUSUM, and a smaller model stack.
2. Barrier settings likely create an exit-heavy label regime for equities.
3. The primary model is likely underpowered compared with the paper's ResNet-LSTM plus HPO, early stopping, multi-seed averaging, and ensembling.
4. The current sampler objective targets sample density rather than a market-move threshold.
5. Backtest/label timing was previously inconsistent when aggregated bars were used; this has been fixed, but prepared folds must be regenerated.
6. Boundary events previously were not purged by label interval; this has been fixed, but old generated folds should not be trusted for final claims.

## Recommended Fix Order

1. Regenerate prepared folds after the entry-timing and purge/embargo fixes.
2. Run a corrected Conv1D baseline and compare `paper/*` with `portfolio/*`.
3. Run a barrier sweep for equities.
4. Run fixed-threshold CUSUM threshold ablations.
5. Compare simple probability-band trading against the meta-label decision layer.
6. Run ResNet-LSTM with early stopping and at least three seeds.
7. Only then compare economics against the reference paper.

## Practical Acceptance Checks

Before trusting a new backtest, verify:

- No trade enters before the model signal timestamp.
- No barrier hit uses high/low information from before entry.
- No training label closes inside validation or test periods.
- No validation label closes inside test periods.
- Net returns include entry and exit costs.
- Same-ticker overlapping trades are either prevented or explicitly modeled.
- Concurrent cross-ticker exposure is bounded by an explicit portfolio rule.
- Reported Sharpe uses a return series that matches the portfolio exposure assumption.

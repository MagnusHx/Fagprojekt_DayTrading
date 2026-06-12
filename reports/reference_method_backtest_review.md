# Reference Method and Backtest Review

## Scope

This review compares the current `kvant` pipeline with the methods described in:

- `references/s40854-025-00866-w.pdf`
- `references/lopez2018.pdf`

It focuses on why the current project may be producing weak trading results and whether the current backtest is methodologically correct.

## Short Answer

The project now contains the main reference-inspired building blocks in a materially cleaner form: fixed-threshold
CUSUM as the default baseline, Triple Barrier labeling, train-only preprocessing, side/meta-label separation,
transaction costs, purged split construction, next-sampled-bar execution, and non-overlapping same-ticker trade
handling.

The most serious earlier correctness issues have been addressed in source:

- aggregated sampled bars no longer enter at the historical segment open
- split construction now purges label intervals that cross boundaries

The remaining gaps are no longer basic timing correctness problems. They are mainly methodology and calibration gaps:
US equities versus crypto, feature timing differences, no volatility-scaled barriers yet, no range-bar ablation, and
no custom period-aware embargo beyond interval purging.

## High-Priority Findings

| Priority | Finding | Impact | Where |
| --- | --- | --- | --- |
| P1 | Current defaults are now closer to the 2025 paper, but the project still uses US-equity-specific data and a shorter warmup window | Paper comparison is improved but still not literal | `FixedThresholdCUSUMBarSampler(h=0.02)`, `TripleBarrierLabeler(width_periods=24, height=0.05)` |
| P1 | Static paper-style barriers may still be poorly calibrated for US equities | Weak directional sample quality and poor economics remain possible even with paper-aligned defaults | `TripleBarrierLabeler(width_periods=24, height=0.05)` |
| P1 | Period-based barriers currently rely on label-interval purging without an added period-aware embargo heuristic | Possible near-boundary dependence remains worth studying | `src/kvant/ml_prepare_data/prepare_experiment.py` |
| P1 | Model/training stack is weaker than the reference setup | Lower signal quality | Conv1D baseline, limited HPO, no multi-seed ensemble |
| P2 | Feature timing differs from the 2025 paper | Not necessarily wrong, but not paper-faithful | Project computes features before sampling; paper describes computing indicators after sampling |
| P2 | Annualization and portfolio assumptions are simplified | Metrics are useful diagnostics, not live-trading PnL | `compute_paper_trading_metrics` |

## Reference Methods That Matter

### 1. Event-Based Sampling

Lopez de Prado motivates event-based sampling because ML should learn from relevant events, not arbitrary clock times. The 2025 paper tests time bars, dollar bars, volume bars, CUSUM, and range bars. Its strongest conclusion is that CUSUM plus Triple Barrier labeling was the most robust family in their crypto experiments.

Current project status:

- Uses CUSUM-style sampling.
- Defaults to fixed-threshold CUSUM with a `2%` threshold.
- Still supports tuned bars/day CUSUM as a project-specific ablation.
- Does not yet implement range bars, dollar bars, or volume bars as ablations.

Suggested work:

1. Compare fixed `1%`, `2%`, and `3%` CUSUM thresholds against the current `2%` default.
2. Compare fixed CUSUM thresholds against the tuned-bars/day sampler.
3. Add range bars as the closest alternative to CUSUM.
4. Treat dollar/volume bars as lower priority for US equities unless reliable volume/dollar-volume adjustments are handled.

### 2. Triple Barrier Labeling

Both references support Triple Barrier labeling as more trading-realistic than next-bar labeling. It defines a trade by profit-taking, stop-loss, and vertical time exit.

Current project status:

- Uses Triple Barrier labels.
- Keeps raw `down/exit/up` event outcomes in prepared artifacts.
- Uses a side model plus meta-label decision layer.

Main concern:

- The repo now defaults to the paper's `24`-period vertical barrier and `5%` symmetric horizontal barriers.
- Those defaults are plausible as a baseline but are still not necessarily calibrated to the US equity minute setting.

Suggested work:

1. Sweep fixed CUSUM and static Triple Barrier values around the paper defaults on equities.
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

1. Work from regenerated paper-default folds first.
2. Then evaluate meta-label calibration and threshold stability.
3. Compare meta-labeling against a simple fixed probability band, such as long if `P(up) > 0.60`, short if `P(up) < 0.40`.

### 4. Purging and Embargo

Lopez de Prado emphasizes that labels derived from overlapping time intervals require purged and embargoed splits. This applies directly to Triple Barrier labels.

Current project status:

- Uses chronological train/validation/test splits.
- Checks whether a label interval remains inside the split before adding it to the saved index.
- Applies a fixed time embargo only for minute-based barriers; period-based default barriers rely on interval purging alone.

Why this matters:

- A training event that starts shortly before validation can have a Triple Barrier outcome determined by validation prices.
- A validation event shortly before test can be determined by test prices.
- This contaminates model selection and backtest interpretation.

Suggested work:

1. Keep validating that label close time remains inside the same split as the event start.
2. Decide whether period-based barriers need an additional custom embargo rule beyond the existing interval purge.
3. Add validation tests that assert no train label interval intersects validation/test windows.

## Is the Backtest Correct?

Much closer, but still not perfect.

The code does several good things:

- Simulates long and short trades.
- Applies transaction cost on entry and exit.
- Skips overlapping trades for the same ticker.
- Allows concurrent trades across different tickers.
- Computes portfolio compounding, daily returns, Sharpe, drawdown, and active time.

But there are still interpretation risks that should be kept in mind before treating the numbers as production-grade
economics.

### Backtest Issue 1: Aggregated Bar Entry Uses the Past

`_aggregate_ohlcv_segments` builds a sampled bar with:

- index = timestamp of the segment end
- open = first open in the segment
- high = max high in the segment
- low = min low in the segment
- close = final close in the segment

The model receives features sampled at the segment end timestamp. At that moment, the segment open is already
historical.

Current status:

- This has been fixed in the default path.
- Labeling and backtest now both treat the sampled row as the signal and enter at the next sampled bar open.

### Backtest Issue 2: Same-Bar Barrier Hits

The main risk was that barrier search could include pre-signal path information from the same aggregated bar.

Current status:

- The default next-bar entry convention resolves the worst version of this issue.
- A future integration test using aggregated CUSUM segments plus raw minute reconstruction would still be valuable.

### Backtest Issue 3: Split Boundary Leakage

The preparation code processes full train+validation+test history per ticker and labels sampled bars. That is acceptable only if index construction prevents training labels from using future split prices.

Current status: mostly fixed in the preparation pipeline. Split index construction now uses label intervals and purges
examples that would close outside their split. Minute-based barriers also receive a fixed embargo. Period-based
defaults currently rely on interval purging alone.

Implemented fix:

- Store `bar_open_time` and `bar_close_time` for every label.
- Include an example in a split only if both open and close are inside that split.
- Apply an embargo around split boundaries when the barrier uses a fixed wall-clock width.

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

1. The project is not a close reproduction of the 2025 paper. It uses US equities, NYSE hours, minute OHLCV data, a
   4-quarter warmup, and a smaller model stack.
2. Barrier settings likely create an exit-heavy label regime for equities.
3. The primary model is likely underpowered compared with the paper's ResNet-LSTM plus HPO, early stopping, multi-seed averaging, and ensembling.
4. The project still supports non-paper sampler variants and has not yet completed a fresh paper-default rerun, so the
   practical baseline remains unverified.
5. Backtest/label timing was previously inconsistent when aggregated bars were used; this has been fixed, but prepared
   folds must be regenerated.
6. Boundary events previously were not purged by label interval; this has been fixed, but old generated folds should
   not be trusted for final claims.

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

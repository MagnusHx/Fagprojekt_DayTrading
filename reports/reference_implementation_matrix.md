# Reference Implementation Matrix

Last updated: 2026-04-27

This matrix tracks methods and suggestions derived from the project references and review notes:

- `references/lopez2018.pdf`
- `references/s40854-025-00866-w.pdf`
- `reports/reference_method_backtest_review.md`
- `reports/project_performance_gap_analysis.md`

Use this as a living team document. When a method changes, update the implementation status, test status, evidence, and next action in the same pull request or commit.

## Status Legend

| Status | Meaning |
| --- | --- |
| Implemented | Code exists and is part of the current intended pipeline. |
| Partial | Some code exists, but the method is incomplete, not default, or not fully faithful to the reference. |
| Planned | Method is relevant but not implemented yet. |
| Not planned | Method is known but currently not a priority. |
| Tested | Covered by unit tests, contract tests, or an executed experiment with recorded results. |
| Indirectly tested | Covered through broader pipeline tests, but no focused method-specific test exists. |
| Not tested | No meaningful test or experiment evidence yet. |

## Implementation Matrix

| Area | Method or suggestion | Reference motivation | Current implementation | Implementation status | Test status | Evidence in project | Priority | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sampling | Event-based sampling instead of pure clock-time bars | Lopez de Prado recommends sampling informative events instead of every time interval. The 2025 paper compares information-driven bars. | `TunedCUSUMBarSampler` samples events from price moves and can aggregate OHLCV segments. | Implemented | Indirectly tested | `src/kvant/ml_prepare_data/samplers/sampler_cumsum.py`; preparation tests exercise sampled feature alignment. | High | Keep as baseline, but compare with fixed-threshold CUSUM. |
| Sampling | Train-only fitting of sampler parameters | Avoid fitting sampling thresholds on validation/test data. | Sampler is fit during preparation using train data before full-history transform. | Implemented | Indirectly tested | `src/kvant/ml_prepare_data/prepare_experiment.py`; README pipeline description. | High | Add a focused test that sampler `fit` never sees val/test rows. |
| Sampling | Fixed-threshold CUSUM sweeps such as 1%, 2%, 3% | The 2025 paper evaluates fixed CUSUM thresholds and reports sensitivity across thresholds. | `FixedThresholdCUSUMBarSampler` uses one fractional threshold across tickers and is selectable from the preparation CLI. | Implemented | Tested | `src/kvant/ml_prepare_data/samplers/sampler_cumsum.py`; `tests/test_samplers.py`; `uv run python -m kvant.ml_prepare_data.prepare_experiment --sampler fixed_cusum --cusum-h 0.01`. | High | Run threshold sweeps against tuned sampler and record label balance/economics. |
| Sampling | Range bars as an ablation | 2025 paper compares range bars with CUSUM and other bar types. | No range bar sampler. | Planned | Not tested | No implementation found under `src/kvant/ml_prepare_data/samplers/`. | Medium | Implement simple range-bar sampler and compare density, label balance, and economics. |
| Sampling | Dollar bars / volume bars as ablations | 2025 paper compares dollar and volume bars. Lopez de Prado discusses alternative information bars. | No dollar/volume bar sampler. Dollar volume is used only for ticker selection. | Planned | Not tested | `src/kvant/kdata/hf_minute_data.py` uses dollar volume for top ticker selection, not event bars. | Medium | Add only after volume normalization and liquidity handling are clear. |
| Features | TA indicator feature set aligned to the paper | 2025 paper uses technical indicators including EMA, MACD, RSI, stochastic, Williams %R, Bollinger, CMF, MFI, ATR, OBV, ROC. | `IntradayTA10Features` implements a broad TA feature set on minute data. | Implemented | Indirectly tested | `src/kvant/ml_prepare_data/features/feature_engineering.py`. | High | Add direct feature-shape and no-lookahead tests for TA features. |
| Features | Feature engineering before sampling | Current project approach computes features on minute bars first and samples the resulting features at event timestamps. This reduces indicator distortion from irregular event bars. | Implemented as current intended approach. | Implemented | Tested | `tests/test_data.py::test_prepare_experiment_computes_features_before_sampling`. | High | Keep documented because it intentionally differs from the 2025 paper's after-sampling indicator description. |
| Features | Train-only standardization | Prevent validation/test distribution leakage into scaler parameters. | `StandardizedFeatures` is fitted on train chunks. | Implemented | Indirectly tested | `src/kvant/ml_prepare_data/features/feature_engineering.py`; preparation workflow. | High | Add focused test for scaler fit rows. |
| Features | Train-only feature selection | Prevent selection bias from validation/test labels. | `PrimarySideFScoreSelector` can be fit on train labels and metadata is rewritten for selected columns. | Implemented | Tested | `tests/test_data.py::test_prepare_experiment_feature_selector_is_fit_on_train_only_and_rewrites_metadata`. | High | Keep selector metadata in generated artifacts. |
| Labeling | Triple Barrier labeling | Both references support barrier-based labels as more trading-realistic than simple next-bar direction. | `TripleBarrierLabeler` persists canonical `down/exit/up` event outcomes and metadata. | Implemented | Tested | `src/kvant/labelling.py`; `src/kvant/ml_prepare_data/labelling/tripple_bar.py`; `tests/test_labelling.py`. | High | Continue using raw event outcomes in artifacts. |
| Labeling | Calibrate barrier width and height for US equities | Current poor results may come from crypto-friendly or overly wide barrier settings transferred to equities. | Default examples use `width=180` minutes and `height=0.015`; plotting/sweep utility exists. | Partial | Not tested as a systematic experiment | `src/kvant/ml_prepare_data/plot_labelling/vary_labeller_runs.py`; gap report. | High | Run a sweep over width/height and record tradeable share, label balance, net return, drawdown. |
| Labeling | Dynamic volatility barriers | Lopez de Prado often scales barriers by volatility; the 2025 review notes dynamic barriers are worth testing but not guaranteed better. | Not implemented in current labeler. | Planned | Not tested | No dynamic barrier code found in `src/kvant/labelling.py`. | Medium | Add as an ablation after fixed barrier sweep. |
| Labeling | Preserve raw 3-class event outcomes | Side/meta pipeline should not destroy `exit` rows; downstream can derive side labels and abstention metrics. | Prepared artifacts store canonical event labels and metadata. | Implemented | Tested | `src/kvant/labels.py`; `tests/test_data.py`; `tests/test_metric_debugging.py`. | High | Keep backward compatibility validation for label semantics. |
| Leakage control | Purging label intervals across train/val/test boundaries | Lopez de Prado purging is required when labels span time intervals. | Index construction now includes examples only when the label interval is safe for the split. | Implemented | Tested | `src/kvant/ml_prepare_data/prepare_experiment.py`; `tests/test_data.py::test_label_interval_split_safety_purges_boundary_crossing_and_embargoes`. | Critical | Regenerate prepared folds before trusting new validation/test metrics. |
| Leakage control | Embargo near split boundaries | Lopez de Prado recommends embargoing observations around split boundaries to reduce leakage from overlapping labels. | Derived embargo uses `labeler.width_minutes` before the next split boundary. | Implemented | Tested | `src/kvant/ml_prepare_data/prepare_experiment.py`; `tests/test_data.py`. | Critical | Record purged counts from generated ticker metadata after regeneration. |
| Execution timing | Do not enter at historical open of aggregated sampled bar | Aggregated CUSUM bars are timestamped at segment end; entering at the segment open is not live-trade-consistent. | Labeler and backtest now treat sampled row as signal and enter next sampled bar open. | Implemented | Tested | `src/kvant/labelling.py`; `src/kvant/ml_framework/train/backtest.py`; `tests/test_labelling.py`; `tests/test_trading_metrics.py`. | Critical | Regenerate labels/backtests and compare metrics before/after fix. |
| Execution timing | Do not let pre-entry high/low trigger barriers | Barrier path must start at the executable entry bar. | Barrier path begins at next sampled entry bar after signal. | Implemented | Tested | `tests/test_labelling.py::test_triple_barrier_enters_on_next_sampled_bar_open`. | Critical | Add an integration test with aggregated CUSUM segments and raw minute data. |
| Backtest | Transaction costs on entry and exit | Paper economics include trading costs; costs can dominate intraday edge. | Backtest subtracts `2 * transaction_cost` per executed trade. | Implemented | Tested | `tests/test_trading_metrics.py::test_compute_paper_trading_metrics_exposes_cost_drag_when_gross_edge_is_positive`. | High | Keep default cost explicit in experiment names and W&B config. |
| Backtest | Same-ticker non-overlapping trades | Avoid unrealistic simultaneous positions in the same ticker unless explicitly modeled. | Candidate trades are sorted and same-ticker overlaps are skipped. | Implemented | Tested | `src/kvant/ml_framework/train/backtest.py`; `tests/test_trading_metrics.py`. | High | Continue reporting skipped-overlap count. |
| Backtest | Concurrent cross-ticker exposure constraints | Current simulator allows concurrent trades across tickers; this may overstate portfolio usage. | Budget-constrained portfolio simulator now caps per-trade size, total exposure, and max concurrent positions across tickers. | Implemented | Tested | `src/kvant/ml_framework/train/portfolio_simulator.py`; `tests/test_portfolio_simulator.py`. | Medium | Use portfolio metrics for final economic claims and keep paper metrics as diagnostics. |
| Backtest | Daily mark-to-market portfolio returns | More realistic economics require marking open positions and cash/exposure over time. | Portfolio simulator tracks cash, open positions, equity curve, exposure, costs, realized PnL, drawdown, Sharpe, and final balance. | Implemented | Tested | `src/kvant/ml_framework/train/portfolio_simulator.py`; W&B logs `perf/portfolio_equity_curve/{split}`. | Medium | Compare portfolio curves across threshold and model experiments. |
| Decision policy | Side plus meta-labeling | Lopez de Prado meta-labeling separates side prediction from whether to act. | Primary side model plus logistic meta-labeler over probabilities, embeddings, prepared feature aliases, prediction uncertainty, rolling ticker win/return stats, and time since last event. | Implemented | Tested | `src/kvant/ml_framework/train/decision_policy.py`; `src/kvant/ml_prepare_data/data_loading.py`; `tests/test_decision_policy.py`. | High | Evaluate threshold stability after regenerating corrected labels. |
| Decision policy | Simple probability band baseline | 2025 paper uses rules like long if probability exceeds 60%, short if below 40%. | Helper threshold functions exist, but current evaluator defaults to meta-label decision policy. | Partial | Tested as helper behavior | `tests/test_trading_metrics.py` covers threshold helpers. | Medium | Run an experiment comparing meta-labeling against fixed probability bands. |
| Decision policy | Probability calibration | Threshold-based trading depends on calibrated probabilities. | No explicit calibration layer. | Planned | Not tested | No calibration model found. | Medium | Try temperature scaling or isotonic calibration on validation predictions. |
| Modeling | Conv1D baseline | Lightweight baseline for sequence classification. | `Conv1DClassifier` is default CLI model. | Implemented | Tested | `src/kvant/ml_framework/models/conv1d.py`; model tests. | High | Keep as baseline for fast regressions. |
| Modeling | ResNet-LSTM stronger architecture | 2025 paper uses stronger temporal architectures; gap analysis recommends ResNet-LSTM. | `ResNetLSTMClassifier` exists and is selectable with `--model resnet_lstm`. | Implemented | Indirectly tested | `src/kvant/ml_framework/models/resnet_lstm.py`; CLI parser tests. | High | Run full corrected-data experiments with ResNet-LSTM. |
| Modeling | Early stopping | Reference setup uses validation-driven stopping. | Trainer stores best state by validation metric. | Implemented | Indirectly tested | `src/kvant/ml_framework/train/trainer.py`. | High | Confirm checkpoint metric in experiment sheets. |
| Modeling | Multi-seed averaging / ensembling | Reference paper uses stronger evaluation through multiple runs or ensembling. | Not implemented as standard workflow. | Planned | Not tested | Gap report recommends 3-seed averaging. | Medium | Add run matrix script for 3 seeds per fold and aggregate metrics. |
| Modeling | Hyperparameter optimization | Reference setup uses HPO/Hyperband-style tuning. | No HPO system in current project. | Planned | Not tested | No HPO code found. | Medium | Start with small grid over model size, learning rate, dropout, width/height, sampler threshold. |
| Evaluation | Walk-forward folds | More realistic than random splits for time series. | Project builds reproducible fold artifacts and manifests. | Implemented | Tested | README; `tests/test_data.py`; smoke validation tests. | High | Regenerate all folds after critical timing/leakage fixes. |
| Evaluation | Artifact validation before training | Prevent training on stale or malformed prepared data. | `validate_prepared_experiment` and smoke script check artifact contracts. | Implemented | Tested | `src/kvant/ml_framework/run_validation.py`; `tests/test_data.py`. | High | Add validation for label interval safety in stored indices. |
| Evaluation | Metrics grouped by side, meta, decision, execution, economics | Needed to understand whether failure is model, decision, or execution related. | Metric registry and evaluator report grouped metrics. | Implemented | Tested | `src/kvant/ml_framework/train/metric_registry.py`; `tests/test_metric_registry.py`; `tests/test_metric_debugging.py`. | High | Keep experiment sheets updated after each run. |
| Reporting | Compare against reference under fair assumptions | Current US-equity setup is not a direct reproduction of crypto paper results. | Gap report documents differences. | Implemented | Not a code test | `reports/project_performance_gap_analysis.md`. | High | Avoid claiming paper reproduction until fixed-threshold CUSUM, barrier sweeps, and stronger models are run. |

## Immediate Team Checklist

1. Regenerate prepared folds after the entry-timing and purge/embargo fixes.
2. Record label distribution, purged counts, and embargo minutes from generated ticker metadata.
3. Run the current Conv1D baseline on corrected folds to establish a new baseline.
4. Run ResNet-LSTM on the same corrected folds.
5. Add fixed-threshold CUSUM and run threshold ablations.
6. Run barrier width/height sweeps for US equities.
7. Compare meta-labeling with a simple probability-band trading rule.
8. Use the stricter portfolio simulator for final project claims and compare it against diagnostic paper metrics.

## Update Template

When updating this matrix, add a short note here:

| Date | Team member | Rows changed | Evidence added | Follow-up |
| --- | --- | --- | --- | --- |
| 2026-04-27 | Codex | Initial matrix created from reference review and current code state. | Linked code paths and current tests. | Team should fill owner names and experiment result links after reruns. |
| 2026-04-27 | Codex | Updated fixed-threshold CUSUM row from planned to implemented. | Added sampler code path, tests, and preparation CLI command. | Run threshold sweeps for `h=0.01`, `0.02`, and `0.03`. |
| 2026-04-27 | Codex | Updated meta-labeling row with enriched feature set. | Added prediction margin/entropy, prepared feature aliases, rolling ticker stats, and time-since-event evidence. | Rerun fixed-CUSUM experiment with enriched default meta features and compare acted-on-exit rate. |
| 2026-04-27 | Codex | Updated portfolio/backtest rows from partial to implemented. | Added budget-constrained portfolio simulator, metrics, W&B equity curve, and tests. | Run the latest fixed-CUSUM experiment and compare `portfolio/*` against `paper/*`. |

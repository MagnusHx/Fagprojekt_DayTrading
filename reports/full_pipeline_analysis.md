# Full Pipeline Analysis

Last updated: 2026-06-12

## Scope

This report describes the pipeline that is actually implemented in this repository, not only the idealized method in
the README. It covers the source tree, generated artifacts, tests, documentation, and the two local references:

- `references/lopez2018.pdf`: Marcos Lopez de Prado, *Advances in Financial Machine Learning*.
- `references/s40854-025-00866-w.pdf`: *Algorithmic crypto trading using information-driven bars, triple barrier
  labeling and deep learning*, DOI `10.1186/s40854-025-00866-w`.

The project is a short-horizon US equity research pipeline using minute OHLCV data. It is not a live trading system.

## Folder Map

| Path | Role |
| --- | --- |
| `src/kvant/kdata/` | Downloads, filters, caches, and loads Hugging Face minute OHLCV parquet shards. |
| `src/kvant/kmarket_info/` | NYSE calendar and valid trading-window checks. |
| `src/kvant/ml_prepare_data/` | Walk-forward data preparation: sampling, feature engineering, labeling, split indices, diagnostics. |
| `src/kvant/ml_framework/` | Prepared artifact loading, model training, validation, prediction, evaluation, W&B logging. |
| `src/kvant/ml_framework/models/` | `conv1d` baseline and selectable `resnet_lstm`. |
| `tests/` | Contract tests for labels, sampling, feature timing, data loading, metrics, meta-policy, portfolio simulation. |
| `reports/` | Method reviews, experiment analysis, and implementation matrix. |
| `docs/` | Team process documentation and MkDocs source. |
| `references/` | Local PDFs used for methodology comparison. |
| `Fagprojekt_DayTrading/src/kvant/ml_framework/prepared/` | Untracked generated prepared-data artifact currently present on disk. |

The tracked source expects generated prepared data under `src/kvant/ml_framework/prepared/`, because
`src/kvant/ml_prepare_data/__init__.py` defines that root. That directory is not currently present in the tracked
tree. The generated artifact that exists is nested under the untracked `Fagprojekt_DayTrading/` directory.

## Reference Ideas Used

The Lopez de Prado reference motivates the structure of this project: information-driven/event sampling,
triple-barrier labels, purging and embargo for labels that span time intervals, side-plus-meta-labeling, and careful
backtest interpretation.

The 2025 Financial Innovation article motivates comparing information-driven bars, triple-barrier labeling, technical
features, and deep learning models. That paper is crypto-focused, so this project is not a direct reproduction: the
repository uses US equities, NYSE trading-window constraints, Hugging Face minute OHLCV shards, and a stricter local
portfolio simulator.

## True Pipeline

### 1. Data Source and Walk-Forward Split Construction

Data comes from the Hugging Face dataset `mito0o852/OHLCV-1m`. Monthly parquet shards are loaded through
`src/kvant/kdata/hf_minute_data.py`.

The default preparation entrypoint calls `get_huggingface_top_20_normal_splits()`, which builds top-20 ticker splits
using `get_huggingface_top_n_tiny_splits(n=20, warmup_quarters=4)`. Internally, `available_datasets(first_year=2020,
warmup_quarters=4)` creates seventeen rolling walk-forward configurations:

1. 4 training quarters.
2. The next quarter for validation.
3. The following quarter for test.

Ticker selection is based on dollar volume computed from the training months for each fold. Validation and test are
then filtered to the same selected tickers. The default blacklist excludes several broad ETFs and legacy symbols such
as `SPY`, `QQQ`, `SQQQ`, `TQQQ`, `LQD`, `HYG`, `FB`, and `TLT`.

### 2. Sampler Fit and Event-Bar Construction

The implemented default sampler is `FixedThresholdCUSUMBarSampler` in
`src/kvant/ml_prepare_data/samplers/sampler_cumsum.py`, with the preparation CLI now defaulting to
`--sampler fixed_cusum --cusum-h 0.02` to match the paper's main CUSUM baseline more closely.

After fitting, the sampler transforms the full per-ticker history. With `aggregate_ohlcv=True`, each CUSUM segment is
collapsed into one sampled bar:

- timestamp: segment end timestamp
- open: first segment open
- high: segment high
- low: segment low
- close: final segment close
- volume: segment volume sum

There is also a `TunedCUSUMBarSampler`, selected with `--sampler tuned_cusum --target-bars-per-day <n>`, as a
project-specific ablation when bar-density targeting is desired.

### 3. Feature Engineering

Features are computed before sampling, on the full minute-resolution dataframe. This is intentional and important:
technical indicators are calculated on regular one-minute bars, then sampled at the CUSUM event timestamps.

The default feature engineer is:

- `IntradayTA10Features`
- wrapped by `StandardizedFeatures`
- by default uses the full feature set without train-only feature selection

The TA feature set includes OHLCV, log return, EMA/EWM standard deviation groups, MACD, RSI, stochastic oscillator,
Williams %R, Bollinger features, CMF, MFI, and sine/cosine time features.

The scaler is fit on minute-resolution training chunks only. Optional feature selection is still available through
`--feature-selection-top-k`, but it is no longer part of the default paper-aligned baseline.

### 4. Triple-Barrier Labeling

Labels are produced by `TripleBarrierLabeler` in `src/kvant/ml_prepare_data/labelling/tripple_bar.py`, which delegates
to `tripple_bar_label()` in `src/kvant/labelling.py`.

The canonical event-outcome label space is:

| Label | Meaning |
| ---: | --- |
| `0` | down barrier hit |
| `1` | vertical/time exit |
| `2` | up barrier hit |

The current source default is `lookback_L=96`, a static symmetric barrier height of `5%`, and a vertical barrier of
`24` sampled periods. This matches the paper's main Triple Barrier setup more closely than the older
`width=180 minutes`, `height=1.5%` baseline. The implementation still supports legacy minute-based vertical barriers,
but the default path is now period-based.

The labeler uses a live-safe execution convention: the sampled bar timestamp is treated as the signal time, and entry
is at the next sampled bar open. Barrier scanning starts from that executable entry bar. This fixes the common mistake
of entering at the historical open of the already-completed aggregated CUSUM segment.

Labels persist metadata aligned with each sampled row, including signal time, entry time, close time, event label, and
realized PnL fields. Rows without a valid label receive `-1`.

### 5. Split Indices, Purging, and Embargo

The preparation code concatenates train, validation, and test for each ticker before transforming, but split membership
is enforced later through index construction.

Valid target positions must have:

- a non-`-1` label
- enough lookback history: `position >= lookback_L`
- a label interval that remains inside the relevant split

The pipeline derives an embargo from the labeler width only when the vertical barrier is minute-based. For the new
paper-aligned period-based default, purging is still enforced from the realized label interval metadata, but there is
no additional fixed time embargo because a sampled-period horizon does not map to one constant wall-clock duration.

The final prepared artifact stores:

- `config.json`
- `tickers_all.json`, `tickers_train.json`, `tickers_val.json`, `tickers_test.json`
- per-ticker `features.npy`, `labels.npy`, `timestamps.npy`, `market_data.npy`, `label_metadata.jsonl`, `meta.json`
- `index_train.npy`, `index_val.npy`, `index_test.npy`
- sampler metadata and density/reporting outputs
- a CV manifest when folds are prepared through the CLI

### 6. Runtime Validation

Before training, `validate_prepared_experiment()` checks artifact integrity:

- required files exist
- split indices have shape `(N, 2)`
- split indices do not overlap
- ticker/position indices are in bounds
- labels match declared semantics
- timestamps are monotonic
- market data is available when return metrics are requested
- feature metadata lengths match feature matrix width

For fold runs, `validate_cv_manifest()` validates every fold listed in the manifest. The smoke script
`src/kvant/ml_framework/scripts/smoke_prepared_experiment.py` can materialize a batch and run a model forward pass.

### 7. Primary Side Training

Training is launched from `src/kvant/ml_framework/scripts/train_experiment.py`.

The current training pipeline requires prepared artifacts with raw three-class event outcomes. At runtime it builds
primary-side datasets:

- raw event `0` maps to side label `0` (`down`)
- raw event `2` maps to side label `1` (`up`)
- raw event `1` maps to `-1` and is ignored by the primary-side loss

Model input windows are shaped as `(features, lookback_L)` by `IndexWindowDataset`. The model is trained as a binary
side classifier using cross-entropy with `ignore_index=-1` and class weights from the training dataset.

Available models:

- `conv1d`: default and baseline model
- `resnet_lstm`: selectable stronger model

The `--baseline` preset forces `model=conv1d` and `transaction_cost=0.0`. Otherwise the default transaction cost is
`0.001`.

### 8. Meta-Label Decision Layer

The evaluator implements Lopez de Prado-style side plus meta-labeling. The primary model proposes a side, then a
logistic meta-labeler estimates whether taking that proposed side would have produced a positive realized return.

Default meta features include:

- side probabilities
- learned embedding
- prepared last volatility alias
- prepared last recent return alias
- rolling ticker win-rate features
- prediction margin
- prediction entropy
- time since last event

The default accept threshold is `0.5`. Accepted trades are converted into canonical trade labels (`down` or `up`);
rejected trades become `exit`/abstain. Bet sizes are derived from the meta take probability through a Kelly-style
fraction with clipping.

Important implementation detail: `evaluate_all()` fits the meta model on train predictions and applies it to train,
validation, and test predictions. The single-split helper `evaluate_split()` fits on the split being evaluated, but the
training loop uses `evaluate_all()`.

### 9. Metrics, Backtest, and Portfolio Simulation

Metrics are grouped by pipeline layer:

- `cls/*`: primary side classification
- `meta/*`: TAKE/PASS meta-label quality
- `decision/*`: acted and abstained behavior
- `execution/*`: signal count, executed trades, skipped overlaps
- `paper/*`: diagnostic trade-level economics
- `portfolio/*`: stricter account-level economics

The paper-style backtest uses sampled raw OHLCV, enters on the next sampled bar open, searches barriers after entry,
charges entry and exit transaction costs, and prevents overlapping same-ticker positions.

The portfolio simulator is stricter and should be used for final economic claims. It tracks cash, long and short
positions, transaction costs, exposure, max concurrent positions, skipped-budget trades, equity curve, return,
drawdown, Sharpe, Sortino, and profit factor.

Default portfolio settings:

- initial cash: `$10,000`
- max position fraction: `5%`
- max total exposure: `100%`
- max positions: `10`

### 10. Logging and Artifacts

W&B logging is handled by `src/kvant/ml_framework/logging/wandb_logger.py`. The training CLI stores run metadata,
preflight diagnostics, metric namespacing, optional media, and best checkpoints under `artifacts/checkpoints` unless
checkpoint saving is disabled.

The project also writes local diagnostics under `artifacts/run_debug`. These generated paths are intended to be
regenerable and ignored by git.

## What Is Currently Materialized on Disk

An older untracked prepared artifact currently present is:

`Fagprojekt_DayTrading/src/kvant/ml_framework/prepared/sb_L_12_w120_h1.5_TBPD30`

Its `config.json` indicates:

- sampler: tuned CUSUM, target `30` bars/day
- labeler: width `120` minutes, height `0.015`
- lookback: `12`
- feature engineer: standardized `intraday_ta10`
- no persisted current `pipeline_stage`, `label_semantics`, `label_spaces`, or feature-selector metadata in the
  top-level config

The per-ticker metadata still shows the full TA feature list and sampler diagnostics. For example, AAPL uses CUSUM
threshold `h=0.0025`, has 834,422 raw rows, 48,518 sampled rows, and a sampled density of about 42.26 bars/day. Its
valid target labels are heavily time-exit dominated: 1,864 down, 18,603 exit, and 1,671 up.

This means the source code and generated artifact are not perfectly aligned. The checked-in code now targets a
paper-aligned baseline such as `sb_L_96_wp24_h5_fixedCUSUM0.02_*`. Regenerate prepared folds from the current source
before trusting any new experiment or report.

## Alignment With References

| Method | Reference motivation | Current implementation |
| --- | --- | --- |
| Event-based sampling | Lopez de Prado and the 2025 article emphasize informative events over clock bars. | Implemented with fixed and tuned CUSUM samplers; fixed `2%` CUSUM is now the default baseline. |
| Triple-barrier labeling | Both references use barrier-based event outcomes instead of simple next-bar direction. | Implemented with down/exit/up labels and metadata. |
| Side plus meta-labeling | Lopez de Prado separates side prediction from whether/size to trade. | Implemented as binary side model plus logistic TAKE/PASS meta-labeler. |
| Purging and embargo | Lopez de Prado warns that interval labels leak across splits. | Implemented in split index construction. |
| Deep learning classifier | The 2025 article uses deep learning over engineered trading features. | Implemented with Conv1D and ResNet-LSTM. |
| Transaction-cost-aware evaluation | Intraday costs can dominate gross edge. | Implemented in paper diagnostics and portfolio simulation. |
| Reference-faithful bar ablations | The 2025 article compares multiple bar types. | Fixed CUSUM exists; range, dollar, and volume bars are not implemented. |
| Dynamic volatility barriers | Lopez de Prado often scales barriers by volatility. | Not implemented; the current default remains the paper-style static `5%` barrier. |

## Key Caveats

1. The project is not a direct reproduction of the 2025 crypto paper. It applies related ideas to US equities.
2. Current defaults are closer to the paper than before: fixed `2%` CUSUM, `24` sampled periods, `5%` barrier height,
   and `L=96`.
3. Technical indicators are computed before sampling. This is defensible for one-minute equity indicators but differs
   from pipelines that compute indicators on event bars after sampling.
4. The walk-forward default now uses a 1-year training warmup instead of the paper's long expanding history. This is an
   intentional project choice and should be kept in mind when comparing results.
5. Old generated artifacts should not be trusted if they predate the current next-entry, purging/embargo, and metadata
   contracts.
6. `paper/*` metrics are diagnostic. Use `portfolio/*` metrics for final economic claims.

## Recommended End-to-End Command Flow

Prepare current folds:

```bash
uv run python -m kvant.ml_prepare_data.prepare_experiment
```

Validate/smoke the generated manifest:

```bash
uv run python -m kvant.ml_framework.scripts.smoke_prepared_experiment --cv-manifest src/kvant/ml_framework/prepared/<manifest>.json
```

Run the baseline:

```bash
WANDB_MODE=offline uv run python -m kvant.ml_framework.scripts.train_experiment --baseline --epochs 3
```

Run a stricter economic experiment:

```bash
WANDB_MODE=offline uv run python -m kvant.ml_framework.scripts.train_experiment \
  --model resnet_lstm \
  --epochs 10 \
  --transaction-cost 0.001 \
  --portfolio-initial-cash 10000 \
  --portfolio-max-position-fraction 0.05 \
  --portfolio-max-total-exposure 1.0 \
  --portfolio-max-positions 10
```

Run tests:

```bash
uv run pytest tests/
```

## Bottom Line

The true implemented pipeline is:

minute OHLCV shards -> walk-forward top-volume ticker splits -> train-only CUSUM sampler fit -> minute-level TA feature
computation and train-only scaling -> CUSUM event-bar sampling -> optional train-only feature selection ->
triple-barrier event-outcome labeling -> purged/embargoed split indices -> binary primary side model -> logistic
meta-label TAKE/PASS policy -> Kelly-sized trade decisions -> next-sampled-bar backtest -> budget-constrained
portfolio metrics.

The strongest methodological pieces are already present: event sampling, triple-barrier labels, side/meta separation,
split leakage controls, live-safe next-bar entry, and portfolio-level metrics. The main remaining research risk is not
missing plumbing; it is whether the US-equity sampling and barrier regime produces enough gross edge before costs.

# Project Performance Gap Analysis

## Scope

This memo compares the current repository against the paper in `s40854-025-00866-w.pdf` and ranks the most likely reasons the project is not producing stronger trading results.

Evidence used:

- Paper text from `../s40854-025-00866-w.pdf`
- Project code under `src/kvant`
- Prepared artifacts under `src/kvant/ml_framework/prepared`
- Stored W&B runs under `wandb/`

The goal here is not blind replication. The goal is to identify which gaps are:

- not comparable to the paper
- likely true project weaknesses
- possible implementation issues

## Executive Summary

### Top 3 root causes

| Rank | Root cause | Confidence | Expected impact | Category |
| --- | --- | --- | --- | --- |
| 1 | The repo is solving a materially different market problem than the paper | High | Very high | Not comparable to paper |
| 2 | The current label regime is too `EXIT`-heavy, so classification metrics can look acceptable while directional trading quality stays weak | High | High | Likely true project weakness |
| 3 | The training stack is much weaker than the paper’s setup: tiny Conv1D, fixed hyperparameters, no early stopping, no multi-seed averaging, no ensemble | High | High | Likely true project weakness |

### Short conclusion

The biggest single reason the project does not match the paper is that it is not a close reproduction. The paper uses Binance BTC/ETH tick data in a 24/7 crypto market and reports results from a tuned ResNet-LSTM pipeline with Hyperband, early stopping, and top-3 model ensembling. This repo uses top-20 US equities minute data, a tuned-per-ticker CUSUM sampler targeting bars/day, a 3-class `down/exit/up` target with a dominant `EXIT` class, and a small Conv1D classifier trained with fixed settings. Those differences are large enough that a direct “why don’t I get the paper’s returns?” comparison is not fair.

Inside the project’s own setup, the clearest weakness is the label/task design. In the main prepared dataset family `sb_L_12_w180_h1.5_TBPD30_foldXX`, the mean test `EXIT` share is about `68.96%`, which means the mean tradeable share is only `31.04%`. That makes raw accuracy a weak proxy for economic signal quality.

## Evidence Map

| Claim area | Source |
| --- | --- |
| Paper market, split, model-selection, and trading-rule claims | `../s40854-025-00866-w.pdf` |
| Equities data source and expanding-quarter split logic | `src/kvant/kdata/hf_minute_data.py` |
| Current preparation settings | `src/kvant/ml_prepare_data/prepare_experiment.py` |
| Current CUSUM sampler behavior | `src/kvant/ml_prepare_data/samplers/sampler_cumsum.py` and prepared `sampler_*meta.json` files |
| Current model architecture | `src/kvant/ml_framework/models/conv1d.py` |
| Current training policy | `src/kvant/ml_framework/train/trainer.py` |
| Current optimizer and weighted loss setup | `src/kvant/ml_framework/scripts/train_experiment.py` and `src/kvant/ml_framework/train/utils.py` |
| Main label-distribution evidence | prepared artifacts under `src/kvant/ml_framework/prepared/sb_L_12_w180_h1.5_TBPD30_foldXX` |
| Current run-level economics | `wandb/run-20260408_223253-lxn4p1ym`, `wandb/run-20260418_170817-x71xflhg` |
| Concrete high-accuracy but economically bad example | `../wandb/run-20260415_232318-6sbt4jie` |
| Confusion-matrix evidence for the same example | `../wandb/run-20260415_232318-6sbt4jie/files/media/table/fold00/perf/confusion_matrix_normalized/test_12_3ae5f763b1004aef0820.table.json` |

## Paper vs Project Method Matrix

| Component | Paper | Project | Difference type | Expected impact on performance |
| --- | --- | --- | --- | --- |
| Market and asset universe | BTC and ETH on Binance, tick data, Jan 2018 to Jun 2023 | Top-20 US equities from Hugging Face OHLCV 1-minute dataset | Non-comparable | Very high |
| Trading hours | 24/7 crypto market | NYSE-style market-hours equities pipeline | Non-comparable | Very high |
| Time split | Quarterly expanding window, five test quarters from Q2 2022 to Q2 2023 | Expanding quarterly folds from `hf_minute_data.available_datasets()`, five folds prepared from the equities dataset | Neutral or unknown | Low to medium |
| Sampling | Fixed information-driven sampling thresholds evaluated in the paper, including CUSUM thresholds such as 1%, 2%, 3% | `TunedCUSUMBarSampler(target_bars_per_day=30)` with per-ticker `h` tuned from a grid | Likely harmful for paper comparability | High |
| Labeling | Triple Barrier used as the best-performing target labeling family | Triple Barrier with `width=180` minutes and `height=1.5%`, explicit 3-class `down/exit/up` supervision | Likely harmful | High |
| Trade decision rule | Probability band: long if probability of increase exceeds 60%, short if below 40% | Now uses `p_act = p_up + p_down` and `q_up = p_up / (p_up + p_down)` with action and direction thresholds; earlier runs used argmax plus confidence floor | Neutral to mildly harmful | Medium |
| Model family | ResNet-LSTM incumbent, plus other stronger benchmark families | `Conv1DClassifier`: 2 Conv1D layers, batch norm, dropout, adaptive average pooling, linear head | Likely harmful | High |
| Hyperparameter optimization | Hyperband-based tuning | Fixed defaults in `train_experiment.py` | Likely harmful | High |
| Early stopping | Yes | No early stopping in `Trainer.fit()`; best checkpoint is restored but training always runs full epochs | Likely harmful | Medium |
| Multi-seed robustness | Three variants retained per network type | Single training run | Likely harmful | Medium |
| Ensemble on test | Top-3 equitable voting ensemble | Single model | Likely harmful | Medium |
| Loss balancing | Paper does not describe class-weighted loss | Project uses inverse-frequency class weights in `class_weights_from_dataset()` and passes them to `CrossEntropyLoss` | Neutral or unknown, possibly harmful | Medium |
| Evaluation focus | Test-set profitability, risk, and accuracy metrics after costs | The repo logs good test metrics, but many dashboards and summaries still emphasize validation accuracy | Neutral or unknown | Low |

## Evidence: What the Paper Actually Does

The paper text supports the following method choices:

- The data come from Binance and cover BTC and ETH. The paper states that the quotations “come from Binance” and uses a “quarterly expanding window” with the first test quarter in `Q2 2022` and the last one in `Q2 2023`.
- The paper uses a probability decision band for trading: long only if the probability of a price increase exceeds `60%`, short if it falls below `40%`.
- The training process includes hyperparameter optimization, early stopping, and selecting the top three model variants for an equal-vote ensemble on the test set.
- The transaction cost assumption is `0.1%` per open and close trade, which is actually close to the repo default `transaction_cost=0.001`.

Those points make the strongest paper-side reference frame for the gap analysis.

## Audit 1: Data and Label Pipeline

### Current main preparation settings

The main dataset generator in `src/kvant/ml_prepare_data/prepare_experiment.py` currently uses:

- `get_huggingface_top_20_normal_splits()`
- `TunedCUSUMBarSampler(target_bars_per_day=30, aggregate_ohlcv=True)`
- `IntradayTA10Features` wrapped in `StandardizedFeatures`
- `TripleBarrierLabeler(width_minutes=180, height=0.015, drop_time_exit_label=False)`
- lookback `L=12`

This produces the prepared family `sb_L_12_w180_h1.5_TBPD30_foldXX`.

### Per-fold class distributions

The main prepared dataset family is heavily dominated by the `EXIT` class:

| Fold | Split | Down | Exit | Up | Exit % | Tradeable % | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fold00 | train | 54340 | 151680 | 52860 | 58.59% | 41.41% | 258880 |
| fold00 | val | 1231 | 7918 | 1197 | 76.53% | 23.47% | 10346 |
| fold00 | test | 1382 | 7664 | 1260 | 74.36% | 25.64% | 10306 |
| fold01 | train | 48295 | 172275 | 45099 | 64.85% | 35.15% | 265669 |
| fold01 | val | 1385 | 8428 | 1291 | 75.90% | 24.10% | 11104 |
| fold01 | test | 2619 | 10265 | 2076 | 68.62% | 31.38% | 14960 |
| fold02 | train | 50034 | 189001 | 47580 | 65.94% | 34.06% | 286615 |
| fold02 | val | 2774 | 11459 | 2238 | 69.57% | 30.43% | 16471 |
| fold02 | test | 1279 | 10182 | 1390 | 79.23% | 20.77% | 12851 |
| fold03 | train | 42508 | 190057 | 40372 | 69.63% | 30.37% | 272937 |
| fold03 | val | 1183 | 10349 | 1351 | 80.33% | 19.67% | 12883 |
| fold03 | test | 3136 | 11652 | 2464 | 67.54% | 32.46% | 17252 |
| fold04 | train | 41224 | 188557 | 39215 | 70.10% | 29.90% | 269996 |
| fold04 | val | 3136 | 11652 | 2464 | 67.54% | 32.46% | 17252 |
| fold04 | test | 4492 | 10523 | 4106 | 55.03% | 44.97% | 19121 |

Test-set mean across folds:

- mean `EXIT` share: `68.96%`
- mean tradeable share: `31.04%`

### What that tells us

- The current supervised task is mostly asking the model to identify `EXIT`.
- Accuracy can therefore stay around `0.60` to `0.70` while directional trading quality is still weak.
- `f1_macro` is a better warning sign than accuracy here, because it punishes weak minority-class directional performance.

### Barrier comparability check

The paper’s best-reported setup is about information-driven crypto sampling and Triple Barrier labeling in a different market. The repo’s current settings are not clearly comparable:

- The repo uses `width=180` minutes and `height=1.5%` on liquid US equities.
- For many large-cap US equities, a `1.5%` barrier over a `3-hour` horizon is hard enough that time-exit becomes common.
- The prepared results confirm that: the `EXIT` class is dominant in every fold.

That suggests the current barrier regime is too conservative for directional trade generation in this equity setting.

### Sampling comparability check

The project does not use the paper’s fixed-threshold CUSUM regime. It uses per-ticker tuning to hit a target bars/day count.

Evidence from the prepared artifacts:

- `sampler_global_meta.json` shows `target_bars_per_day = 30.0`
- `sampler_per_ticker_meta.json` shows fold-specific `h` values that are mostly in `{0.0025, 0.005, 0.0075, 0.01}`
- In fold00, the tuned `h` distribution is `{0.0025: 7, 0.005: 11, 0.0075: 1, 0.01: 1}`
- Mean realized samples/day by fold are approximately `27.47`, `28.61`, `30.59`, `29.27`, `29.72`
- Per-ticker means still vary widely. In fold00 they range from `15.69` to `43.86` bars/day

This is a legitimate design, but it is not what the paper tests. It means your CUSUM process is tuned for consistency of sample density, not for the paper’s fixed sampling thresholds.

## Audit 2: Model and Training Stack

### Current repo stack

The current model in `src/kvant/ml_framework/models/conv1d.py` is a very small classifier:

- Conv1D `n_features -> 32`
- Conv1D `32 -> 64`
- batch norm and dropout after each layer
- adaptive average pooling
- linear output layer

The training loop in `src/kvant/ml_framework/train/trainer.py`:

- runs for a fixed number of epochs
- logs `val/accuracy` every epoch
- restores the best checkpoint by `val/accuracy`
- does not use early stopping
- does not run multiple seeds
- does not ensemble multiple checkpoints or runs

The training script in `src/kvant/ml_framework/scripts/train_experiment.py`:

- uses fixed defaults `epochs=10`, `lr=5e-3`, `weight_decay=5e-5`
- always constructs `Conv1DClassifier`
- uses inverse-frequency class weights through `CrossEntropyLoss(weight=...)`

### Paper stack

The paper’s incumbent configuration is much stronger:

- ResNet-LSTM architecture with residual CNN stack and LSTM
- hyperparameters optimized with Hyperband
- early stopping
- top 3 model variants retained
- equal-vote ensemble on the test set

### Impact ranking inside the training stack

| Difference | Likely impact | Why |
| --- | --- | --- |
| Conv1D instead of ResNet-LSTM | High | The current model is much less expressive for local pattern extraction plus temporal state modeling |
| No HPO | High | The paper explicitly tunes important architecture and optimization parameters |
| No early stopping | Medium | The repo restores best checkpoint but still wastes training budget and may amplify instability |
| No seed averaging | Medium | Financial time series are noisy; single-run variance matters |
| No ensemble | Medium | The paper deliberately uses top-3 ensembling to stabilize decisions |
| Class-weighted loss | Medium, uncertain | Could help directional recall, but it is another major deviation from the paper |

## Audit 3: Why Accuracy Can Look Fine While Trading Looks Bad

The stored runs make the disconnect very clear.

### Aggregated run comparison from stored W&B runs

| Run | Epochs | Mean test accuracy | Mean test F1 macro | Mean test trade signal rate | Mean annual net P/L % | Mean Sharpe | Mean max drawdown % | Mean profitable transactions % | Mean share time active % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `run-20260408_223253-lxn4p1ym` | 20 | 0.6420 | 0.4925 | n/a | -86.4659 | -4.3656 | 50.2480 | 46.1704 | 14.0341 |
| `run-20260418_170817-x71xflhg` | 10 | 0.6257 | 0.4822 | 0.0720 | -81.2533 | -4.0277 | 49.8081 | 47.0672 | 7.2851 |

### What those numbers imply

- Training longer from `10` to `20` epochs improves classification slightly.
- But that does not improve economics. Mean annual net P/L and Sharpe get worse.
- This means the problem is not “the model just needs more epochs.”

### Concrete failure example

The outer run `../wandb/run-20260415_232318-6sbt4jie` provides a strong single-fold example:

- `fold00/test/accuracy = 0.6887`
- `fold00/test/f1_macro = 0.4949`
- `fold00/test/paper/annual_net_profit_loss_pct = -99.9420`
- `fold00/test/paper/sharpe_ratio_annualized = -7.9733`
- `fold00/test/trade_signal_rate = 0.3473`

That is the clearest warning sign in the project: moderately good classification metrics can coexist with catastrophic trading metrics.

### Why that happens here

The most likely mechanism is:

1. The label space is dominated by `EXIT`.
2. The model can therefore achieve decent overall accuracy by learning `EXIT` well.
3. The directional classes remain noisy and economically weak.
4. When the system does trade, transaction costs and false directional calls dominate the edge.

This interpretation is also consistent with the normalized confusion matrix from the same run. `EXIT` is learned relatively well, but the directional classes are still confused enough to make trading unattractive.

## Primary Failure Mode

If forced to pick one primary failure mode, the best answer is:

`signal quality`, caused mainly by the current label regime and market mismatch.

Why not the decision rule as the primary explanation:

- The old rule was weaker than the new `p_act / q_up` decision rule, but the archived failures already exist before thresholding becomes the limiting factor.
- The trading decision rule can improve selectivity, but it cannot fix weak directional signal generation by itself.

Why not transaction costs alone:

- The repo uses `0.1%`, which matches the paper’s stated transaction-cost assumption quite closely.
- Costs matter, but they do not explain the entire gap.

Why not barrier economics alone:

- Barrier settings are a major contributor, but they matter mostly because they shape the label regime and therefore the quality of the learned signal.

## Biggest Differences from the Paper

### Not comparable to paper

- Crypto vs equities
- Binance 24/7 tick data vs US equities minute data
- BTC/ETH only vs top-20 liquid equities
- Fixed-threshold paper sampling vs tuned bars/day sampler

These differences are enough that direct performance comparison is not fair.

### Likely true project weaknesses

- `EXIT`-heavy label regime in the main dataset family
- Small Conv1D model relative to the paper’s ResNet-LSTM
- No HPO
- No early stopping
- No multi-seed averaging
- No top-k ensemble
- Class-weighted loss may be distorting optimization in ways that do not align with the paper

### Possible bug or implementation issue

- The prepared dataset family `sb_L_12_w120_h1.5_TBPD30_fold00` has `drop_time_exit_label=true`, while folds `01` to `04` in that family have `drop_time_exit_label=false`
- The same `w120` family also lacks `label_semantics` metadata, unlike the `w180` family

This inconsistency does not appear to affect the main `w180` experiments, but it is exactly the kind of artifact mismatch that can corrupt a controlled ablation if reused without checking.

## Other Things That Could Help Performance

These are the highest-signal levers beyond the top 3 root causes:

| Idea | Why it might help | Type |
| --- | --- | --- |
| Direction-only training with abstention at inference | Removes the dominant `EXIT` class from supervision and makes the task closer to the paper’s probability-band decision logic | Likely performance gain and paper fidelity |
| Fixed-threshold CUSUM sampler | Makes the data pipeline closer to the paper and avoids learning on a tuned density objective that the paper did not use | Paper fidelity |
| Barrier sweep for equities | The current `1.5% / 180min` setting appears to overproduce time exits on large-cap US equities | Likely performance gain |
| Remove class weights as an ablation | The paper does not describe class-balanced loss; weights may over-amplify rare labels without enough economic value | Likely performance gain |
| Probability calibration | If directional probabilities are poorly calibrated, thresholding will be noisy even with the improved `p_act / q_up` rule | Speculative |
| Per-ticker or sector-aware modeling | A single pooled model across 20 equities may be too blunt for heterogeneous intraday behavior | Speculative |
| Longer lookback and richer model | `L=12` with a tiny Conv1D may not capture the intraday structure that a ResNet-LSTM or other temporal model could use | Likely performance gain |

## Prioritized Fix List

| Priority | Fix | Label |
| --- | --- | --- |
| 1 | Run a controlled directional-only ablation with `drop_time_exit_label=True` across all 5 folds | Likely performance gain |
| 2 | Add a fixed-threshold CUSUM sampler mode to match the paper’s methodology more closely | Paper fidelity |
| 3 | Sweep Triple Barrier width and height for equities instead of assuming the paper’s crypto-friendly label behavior transfers | Likely performance gain |
| 4 | Add early stopping and 3-seed aggregation before changing architectures | Likely performance gain |
| 5 | Add a stronger baseline model, preferably ResNet-LSTM | Paper fidelity and likely performance gain |
| 6 | Ablate inverse-frequency class weighting | Likely performance gain |

## Recommended Next Step

### Fastest likely improvement

Run the directional-only ablation first.

Reason:

- It directly attacks the strongest internal weakness: the dominant `EXIT` class.
- The repo already has proof-of-concept groundwork in `sb_L_12_w180_h1.5_TBPD30_droptexit_fold04`.
- It is cheaper than implementing a new model family first.

### Closest paper replication

Implement a fixed-threshold CUSUM sampler and a ResNet-LSTM baseline, then rerun the same 5-fold protocol with early stopping and 3-seed averaging.

Reason:

- That gets the repo materially closer to the paper’s data pipeline and model stack.
- Without those changes, a “paper comparison” will stay only loosely meaningful.

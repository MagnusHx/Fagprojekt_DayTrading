# Project Performance Experiment Sheet

## Goal

Turn the gap analysis into a controlled experiment sequence that can tell us whether the main bottleneck is:

- market mismatch
- label design
- sampling design
- barrier economics
- training-policy weakness
- model-family weakness

All comparisons should use fold-aggregated **test** metrics only.

## Default Metrics for Every Experiment

For each 5-fold run, record the mean and standard deviation across folds of:

- `test/paper/annual_net_profit_loss_pct`
- `test/paper/sharpe_ratio_annualized`
- `test/paper/max_drawdown_pct`
- `test/paper/profitable_transactions_pct`
- `test/paper/share_time_active_pct`
- `test/accuracy`
- `test/f1_macro`
- `test/trade_signal_rate`

Primary ranking metric:

1. higher mean `test/paper/sharpe_ratio_annualized`
2. higher mean `test/paper/annual_net_profit_loss_pct`
3. lower mean `test/paper/max_drawdown_pct`

Guardrail:

- reject any experiment with worse Sharpe and worse annual P/L than the baseline, even if accuracy improves

## Baseline to Freeze

Use one clean reference run before changing anything else.

Recommended baseline:

- current main dataset family: `sb_L_12_w180_h1.5_TBPD30_foldXX`
- current model family: `Conv1DClassifier`
- new decision rule already implemented in the repo
- epochs: `20`
- thresholds: `trade_action_threshold=0.60`, `trade_direction_threshold=0.60`

Command:

```bash
cd /home/magnus/repositories/Fagprojekt_DayTrading/Fagprojekt_DayTrading
uv run python -m kvant.ml_framework.scripts.train_experiment \
  --epochs 20 \
  --wandb-name "cv-baseline-w180h1.5-tbpd30-act60-dir60" \
  --trade-action-threshold 0.60 \
  --trade-direction-threshold 0.60
```

Success condition for the baseline:

- all 5 folds finish
- W&B stores fold-level test paper metrics and summary metrics cleanly

## Experiment 1: Label Decision Ablation

### Hypothesis

The dominant `EXIT` class is the biggest internal bottleneck. Removing it from supervision and treating abstention purely as an inference-time choice should improve economic signal quality.

### Change

- regenerate all 5 folds with `drop_time_exit_label=True`
- keep the same lookback, features, and barrier settings initially
- keep the same decision rule at inference, but now with a directional model

### Implementation notes

- Current proof of concept exists only for `sb_L_12_w180_h1.5_TBPD30_droptexit_fold04`
- The main preparation loop in `src/kvant/ml_prepare_data/prepare_experiment.py` will need a second dataset-family path or a parameterized run mode
- Ensure all folds in the new family have consistent `drop_time_exit_label` metadata

### Suggested naming

- dataset family: `sb_L_12_w180_h1.5_TBPD30_droptexit_foldXX`
- W&B run: `cv-droptexit-w180h1.5-act60-dir60`

### Success criterion

- Sharpe improves materially versus the frozen baseline
- annual net P/L improves materially versus the frozen baseline
- max drawdown does not worsen materially

If accuracy falls but Sharpe and annual P/L improve, count that as success.

## Experiment 2: Sampling Ablation

### Hypothesis

The tuned-bars/day CUSUM sampler is not paper-faithful and may be producing a data regime that is too different from the paper.

### Change

- add a fixed-threshold CUSUM sampler mode
- test at least one paper-like fixed threshold, especially `2%`
- compare against the current tuned-bars/day sampler

### Implementation notes

- The repo currently only contains `TunedCUSUMBarSampler` in `src/kvant/ml_prepare_data/samplers/sampler_cumsum.py`
- Add a second sampler class or parameter mode for fixed `h`
- Keep the rest of the pipeline unchanged when comparing tuned vs fixed

### Suggested experiment grid

| Sampler | Parameters |
| --- | --- |
| Baseline sampler | `target_bars_per_day=30` |
| Fixed CUSUM | `h=0.02` |
| Optional extra | `h=0.01`, `h=0.03` |

### Success criterion

- fixed-threshold CUSUM should improve Sharpe and annual net P/L without catastrophic trade-frequency collapse

## Experiment 3: Barrier Ablation

### Hypothesis

The current barrier regime `width=180`, `height=1.5%` is too exit-heavy for liquid US equities.

### Change

Sweep Triple Barrier settings while keeping the sampler and model fixed.

### Minimum grid

| Width minutes | Height |
| --- | --- |
| 120 | 1.0% |
| 120 | 1.5% |
| 180 | 1.0% |
| 180 | 1.5% |

### What to record before training

- per-fold class distribution
- per-fold `EXIT` share
- per-fold tradeable share

### Success criterion

- the best setting should lower `EXIT` share meaningfully
- it should improve mean Sharpe and annual P/L versus baseline
- it should not simply spike trade count while worsening drawdown

## Experiment 4: Training-Policy Ablation

### Hypothesis

The project is under-training or overfitting because it lacks early stopping and run aggregation.

### Change

Add:

- early stopping on `val/accuracy` or a better validation trading proxy
- 3 random seeds per fold
- seed-averaged reporting

### Implementation notes

- `Trainer.fit()` currently tracks the best checkpoint but does not stop early
- add patience-based stopping first
- keep architecture fixed during this experiment

### Success criterion

- lower variance across folds and across seeds
- modest improvement in mean Sharpe and annual P/L

## Experiment 5: Class-Weight Ablation

### Hypothesis

Inverse-frequency class weighting may be amplifying labels that are rare but not economically valuable.

### Change

Compare:

- current weighted loss
- plain `CrossEntropyLoss()`

### Implementation notes

- current weighting happens in `train_experiment.py` through `class_weights_from_dataset()`
- keep every other setting fixed

### Success criterion

- improved Sharpe or annual P/L with no severe collapse in macro F1

## Experiment 6: Model-Family Ablation

### Hypothesis

The current Conv1D model is too weak relative to the paper’s ResNet-LSTM baseline.

### Change

Implement a ResNet-LSTM baseline and compare it against the current Conv1D under the same fold protocol.

### Implementation notes

- current model is defined in `src/kvant/ml_framework/models/conv1d.py`
- add a new model module and a CLI switch to select the architecture
- do not combine this with HPO in the first pass; first prove the architecture helps under matched training conditions

### Success criterion

- better mean Sharpe and annual P/L than Conv1D
- no worse max drawdown than baseline

## Recommended Order

Run the experiments in this order:

1. clean baseline rerun
2. label decision ablation
3. class-weight ablation
4. barrier ablation
5. sampling ablation
6. early stopping plus 3 seeds
7. model-family ablation

Reason for this order:

- the first three are the cheapest and attack the highest-confidence weaknesses
- the later steps require more code and more compute

## Decision Rules

### Fast fail

Stop an ablation branch early if:

- mean Sharpe drops further below baseline
- mean annual net P/L becomes more negative
- mean max drawdown increases materially

### Keep going

Promote an ablation branch if:

- mean Sharpe improves
- mean annual net P/L improves
- drawdown is stable or improved

### Report outcome as one of three labels

- `promising`: better Sharpe and better annual P/L than baseline
- `mixed`: one economic metric improves, but another worsens
- `rejected`: economics worsen despite similar or better classification

## What Success Would Look Like

The first convincing sign that the project is moving in the right direction would be:

- a directional-only or re-labeled setup
- with lower `EXIT` share
- producing higher Sharpe and less negative annual net P/L on aggregated **test** folds

The first convincing sign of paper fidelity would be:

- a fixed-threshold CUSUM pipeline
- a stronger temporal model such as ResNet-LSTM
- early stopping and seed aggregation

## Immediate Recommendation

Start with:

1. baseline rerun with the new decision rule
2. directional-only ablation
3. class-weight ablation

That is the highest-signal path per unit of implementation effort.

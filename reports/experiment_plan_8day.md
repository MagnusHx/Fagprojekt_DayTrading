# 8-Day Experiment Plan (2026-06-12 → 2026-06-19)

Goal: produce a defensible report that answers the four research questions through a
**ladder of comparisons**, starting from trivially simple baselines and adding one piece of
complexity at a time. Every experiment below exists to fill one specific table or figure in
the report.

## Where we stand (so the plan makes sense)

From the latest directional-only W&B run (epoch 33, 1 fold):

| Metric | Value | Meaning |
| --- | ---: | --- |
| test classification accuracy | 0.495 | Primary side model is at **chance** |
| test f1_macro | 0.495 | Same |
| test meta F1 | 0.413 | Meta layer cannot help a chance-level primary model |
| trade_signal_rate | 0.366 | Meta takes ~37% of signals |
| economic metrics | missing | Run used `--no-return-stats`; no `paper/*`/`portfolio/*` logged |

Consequences:

1. We will probably **not** find a profitable strategy in 8 days. That is fine. The research
   questions are *comparative* ("does X improve over Y?"), and a clean null result answers
   them. The report dies only if the comparisons are missing or unfair.
2. The single most important missing piece is the **RQ1 baseline pipeline**
   (fixed time bars + next-bar direction labels). It is not implemented — only
   `tuned_cusum`/`fixed_cusum` samplers and triple-barrier labels exist. Without it, RQ1
   cannot be answered at all. This is build item #1.
3. Final runs must drop `--no-return-stats` and run **all 5 walk-forward folds**, otherwise
   we have no CAGR/Sharpe/drawdown numbers and no error bars for the report.

## The experiment ladder

Each level adds exactly one piece of complexity. A level's result is interpreted **relative
to the level below it** — that is how we "explain our results by comparing between simple
baseline models".

```
L0  Majority class + logistic regression        (floor: is DL doing anything?)
L1  Time bars + next-bar direction + Conv1D     (RQ1 baseline arm)
L2  CUSUM bars + triple-barrier + Conv1D        (RQ1 advanced arm; needs label calibration)
L3  Same as L2 with ResNet-LSTM                 (model complexity, only if L2 > L0)
L4  + confidence thresholds                     (RQ3)
L5  + meta-selection on/off + meta features     (RQ4)
```

## Build items (code we must write first)

| # | Item | Why | Est. effort |
| --- | --- | --- | --- |
| B1 | `TimeBarSampler` (aggregate minute data to k-minute bars, e.g. `--sampler time_bar --time-bar-minutes 15`) | RQ1 baseline arm. 15-min bars ≈ 26 bars/day, comparable density to CUSUM target 30/day → fair comparison | ~half day |
| B2 | `NextBarDirectionLabeler` (`--labeler next_bar`): label = sign of next-bar return, entry at next bar open (reuse the live-safe entry convention) | RQ1 baseline arm | ~half day (with B1) |
| B3 | `scripts/simple_baselines.py`: majority-class + `LogisticRegression` on flattened prepared windows, evaluated with the **same** evaluator/metrics | L0 floor for every table | ~half day |
| B4 | `--no-meta` mode in `train_experiment.py`: act on every primary signal with fixed bet size (no TAKE/PASS, no Kelly) | RQ4 needs a true no-meta arm. `--meta-accept-threshold 0.0` approximates it but still Kelly-sizes bets | small |
| B5 | Threshold sweep at eval time: evaluate the saved meta probabilities at thresholds {0.0, 0.55, 0.65} in one run (extend evaluator or a small script on the best-checkpoint bundle via `reconcile_metrics.py` plumbing) | RQ3 without retraining 3× | small |

Notes for B1/B2: feature engineering already runs on minute data *before* sampling, so the
same `IntradayTA10Features` + scaler + selector work unchanged for time bars — that keeps
the RQ1 comparison "same features, same model, only bars+labels differ", which is exactly
what the research question specifies.

## Experiments

All runs: same seed (1337), same lookback (12), Conv1D defaults unless stated, transaction
cost 0.001 for economic metrics (plus the zero-cost `--baseline` preset where noted),
**no** `--no-return-stats` on final runs, all 5 folds for final runs (single fold OK for
screening). Log to W&B with the run names below.

### E0 — Floors (L0) — feeds Report Table 1

| Run | What | Command sketch |
| --- | --- | --- |
| `E0-majority` | Majority class per split, from class-balance tables | `simple_baselines.py --model majority` |
| `E0-logreg` | Logistic regression on flattened windows | `simple_baselines.py --model logreg` |

Record: accuracy, f1_macro per split. Interpretation rule for the report: *any DL result is
only "signal" if it beats E0-logreg out of sample.*

### E1 — RQ1 head-to-head (L1 vs L2) — feeds Report Table 2 + equity-curve Figure

Same model (Conv1D), same features, same folds. Only bars + labels differ.

| Run | Sampler | Labels |
| --- | --- | --- |
| `E1-timebar` | time_bar 15 min | next-bar direction |
| `E1-cusum` | tuned CUSUM, 30 bars/day | triple-barrier (calibrated, see E2) |

Prepare each, then:

```bash
uv run python -m kvant.ml_framework.scripts.train_experiment \
  --cv-manifest <manifest> --model conv1d --epochs 20 \
  --wandb-name E1-timebar   # / E1-cusum
```

Record: accuracy, f1_macro, and `portfolio/*`: CAGR (annual net P/L), Sharpe, cumulative
return, max drawdown. This is the direct RQ1/RQ2 answer.

### E2 — Label/sampling calibration (fixes the failed sweep) — feeds Report Table 3 (appendix)

The previous sweep failed and the existing artifact is time-exit dominated (AAPL: 18,603
exit vs 1,864 down / 1,671 up at h=1.5%, w=120). The barrier height is too large for
intraday equity vol. Re-run a **reduced** grid with smaller heights, on HPC, screening with
1 fold + Conv1D + 5 epochs:

| Parameter | Values |
| --- | --- |
| barrier height | 0.3%, 0.5%, 0.75% |
| barrier width | 60, 120 min |
| sampler | tuned CUSUM 30/day (fixed; do not also sweep bars/day) |

That is 6 preps + 6 short trainings — fits one 24h HPC job using the existing
`run_experiment_grid.py` / `hpc_label_sweep.sh` machinery (update the `HEIGHTS` array).

**Selection criterion (important):** pick the config by (a) exit-label fraction < 60%, and
(b) most positive *gross* (pre-cost) average trade return on validation — **not** by meta
F1. If gross edge is negative everywhere, pick the most balanced config and say so in the
report; the RQ comparisons remain valid.

### E3 — Model complexity (L3) — feeds Report Table 4

Only on the E2 winner, all 5 folds:

| Run | Model |
| --- | --- |
| `E3-conv1d` | conv1d, 20 epochs |
| `E3-resnet` | resnet_lstm, 30 epochs |

Gate: if `E3-conv1d` does not beat `E0-logreg` on validation f1_macro, run `E3-resnet`
once anyway (cheap, completes the table) but do **not** spend time tuning it — the earlier
ResNet run already showed overfitting (val meta F1 0.09).

### E4 — Selective trading / confidence thresholds (RQ3) — feeds Report Table 5 + frequency-vs-Sharpe Figure

On the best E3 model, **no retraining** (decision layer only, via B5):

| Arm | Threshold |
| --- | --- |
| no threshold | 0.0 (trade every signal) |
| medium | 0.55 |
| high | 0.65 |

Record per arm: n_trades, trade_signal_rate, hit rate, avg net return/trade, Sharpe, max
drawdown. The report analyzes the frequency / hit-rate / risk-adjusted-return trade-off —
even with a weak model the *shape* of this trade-off answers RQ3.

### E5 — Meta-selection ablation (RQ4) — feeds Report Table 6

On the best E3 config, all 5 folds:

| Run | Decision layer |
| --- | --- |
| `E5-nometa` | `--no-meta`: every primary signal, fixed size (B4) |
| `E5-meta-min` | meta with `--meta-features proba,embedding` |
| `E5-meta-full` | meta with full default feature set |

Incremental value of meta-selection = E5-meta-* vs E5-nometa on portfolio metrics.
Feature sensitivity = E5-meta-min vs E5-meta-full. That is RQ4 answered.

## Day-by-day schedule

| Day | Date | Work | Gate at end of day |
| --- | --- | --- | --- |
| 1 | Fri 12 Jun | B1+B2 (time-bar sampler + next-bar labeler, with tests). Kick off E2 grid on HPC tonight. | `prepare_experiment --sampler time_bar --labeler next_bar` produces a valid manifest; HPC job queued |
| 2 | Sat 13 Jun | B3 (simple baselines) → run E0. B4 (`--no-meta`). Regenerate clean prepared folds from current source (the on-disk artifact is stale / mis-nested under `Fagprojekt_DayTrading/`). | Table 1 numbers exist |
| 3 | Sun 14 Jun | Collect E2 results, pick winning label config. Run `E1-timebar` (all folds). | E2 winner chosen with stated criterion |
| 4 | Mon 15 Jun | Run `E1-cusum` = `E3-conv1d` (same run, all 5 folds, full return stats). Start `E3-resnet` on HPC. | Table 2 + equity curves exist |
| 5 | Tue 16 Jun | B5 (threshold sweep) → E4. Start E5 runs. | Table 5 + RQ3 figure exist |
| 6 | Wed 17 Jun | Finish E5. Assemble all tables/figures, per-fold mean ± std. | Tables 4 & 6 exist |
| 7 | Thu 18 Jun | Buffer for reruns/failures. Write results & discussion sections from the tables. | Draft results chapter |
| 8 | Fri 19 Jun | Polish report, sanity-check every number against W&B, archive run IDs in `reports/`. | Done |

**Buffer policy:** if a build item slips, cut from the bottom: E3-resnet first, then
E5-meta-full. Never cut E0, E1, or E4 — they carry RQ1–RQ3.

## Report mapping (so nothing is orphaned)

| Research question | Experiments | Deliverable |
| --- | --- | --- |
| RQ1: do information-driven bars + triple-barrier beat time bars + next-bar direction? | E0, E1 (+E2 appendix) | Tables 1–3, equity-curve figure |
| RQ2: predictive + economic comparison | E1, E3 | Tables 2, 4 (acc/F1 + CAGR/Sharpe/cum-return/MDD) |
| RQ3: does selective trading improve risk-adjusted returns? | E4 | Table 5, frequency-vs-Sharpe figure |
| RQ4: incremental value of meta-selection + sensitivity? | E5 | Table 6 |

## Standing rules for every final run

1. All 5 walk-forward folds; report mean ± std across folds.
2. No `--no-return-stats` — we need `portfolio/*` for every economic claim.
3. Use `portfolio/*` (strict simulator) for conclusions; `paper/*` is diagnostic only.
4. Fixed seed, identical features/lookback across compared arms; one variable changes at a time.
5. Tune thresholds on validation, report on test.
6. Every table cell in the report carries a W&B run name from this file.

# Team Method

Last updated: 2026-04-27

This file defines how the `kvant` team works together. It should be used together with:

- `README.md`
- `reports/reference_implementation_matrix.md`
- `reports/project_performance_experiment_sheet.md`
- `reports/reference_method_backtest_review.md`
- `docs/ai_dialog_log.md`
- `docs/team_change_log.md`
- `docs/experiment_log.md`

The goal is simple: every important project decision should be reproducible, traceable, and understandable by another team member without needing to reconstruct the whole conversation from memory.

## Working Principles

1. Make one methodological change at a time when possible.
2. Keep generated artifacts out of git unless they are small reports or summaries.
3. Regenerate prepared data after any change to sampling, labeling, feature timing, split construction, or backtest timing.
4. Record the exact command used for each important run.
5. Log both good and bad results. A failed experiment is still useful if the setup and result are clear.
6. Prefer small tests that protect important assumptions over large undocumented changes.
7. Do not compare project results to a reference paper unless the differences in market, data, sampling, labeling, model, and backtest assumptions are documented.

## Keeping the Team Aligned

Research projects can easily split into many half-finished directions. Use these rules to keep momentum without blocking creativity.

1. Keep one active project baseline.

The baseline is the version everyone compares against. It should specify data manifest, sampler, labeler, model, decision policy, backtest assumptions, and command. If someone changes one part, they should compare against the baseline instead of inventing a new reference point.

2. Use a small number of active workstreams.

At any time, keep at most three active workstreams:

- Pipeline correctness: leakage, timing, validation, reproducibility.
- Method experiments: sampler, labels, model, decision policy.
- Reporting: project description, figures, experiment summaries.

Everything else goes into the backlog in `reports/reference_implementation_matrix.md`.

3. Assign one owner per workstream.

An owner does not have to do all the work, but they are responsible for keeping the scope clear and updating the relevant log.

4. Require an experiment card before running long experiments.

Before a larger run, write one row in `docs/experiment_log.md` with the question, changed variable, baseline, command, and expected output. This prevents "interesting but unconnected" runs.

5. Change one variable at a time for comparison runs.

For example, do not change sampler, barrier settings, model, and decision policy in the same comparison unless the goal is explicitly to test a full new pipeline.

6. Use decision checkpoints.

After each workstream produces evidence, decide one of:

- Adopt into baseline.
- Keep as optional ablation.
- Reject for now.
- Needs another experiment.

7. Protect project contracts with tests.

If a bug would invalidate results, add a test. Priority contracts are label timing, split leakage, label mapping, feature timing, and metric meaning.

8. Keep AI assistance visible.

When AI suggests or implements something important, record the useful output and the human decision in `docs/ai_dialog_log.md`. This keeps accountability with the team, not with the tool.

9. Prefer shared vocabulary.

Use the same names everywhere: `event_outcome`, `side_label`, `meta_label`, `signal_time`, `entry_time`, `bar_close_time`, `purge`, `embargo`, `baseline`. This reduces confusion in meetings and reports.

10. End each meeting with a short written decision.

Add the decision to the relevant log or implementation matrix. A decision that only exists in conversation will be forgotten or re-litigated.

## Standard Workflow

Use this workflow for research changes, model changes, and pipeline changes.

1. Define the question.

Example: "Does fixed-threshold CUSUM improve label balance and net return compared with tuned bars/day CUSUM?"

2. Locate the relevant reference or project note.

Record whether the idea comes from Lopez de Prado, the 2025 paper, a project report, an AI discussion, or a team member.

3. Create or update an implementation row.

Update `reports/reference_implementation_matrix.md` before or during implementation. At minimum, fill in method, status, test status, evidence, priority, and next action.

4. Implement the smallest useful version.

Keep the first version narrow. For example, implement one sampler mode before building a full ablation framework.

5. Add tests for assumptions.

Focus tests on leakage, timing, label mapping, feature alignment, and metric contracts. These are the places where silent errors are most expensive.

6. Run verification.

Typical minimum commands:

```bash
uv run ruff check .
uv run pytest
```

For data or training changes, also run the relevant smoke command:

```bash
uv run python -m kvant.ml_framework.scripts.smoke_prepared_experiment --cv-manifest <path-to-manifest>
```

7. Record the change.

Add one readable team change-log entry in `docs/team_change_log.md` and, if AI was used, one AI-dialog entry in `docs/ai_dialog_log.md`.

8. Update project reports.

If the change affects methodology, update the project description, experiment sheet, or gap analysis.

## Definition of Done

A project change is done only when these items are true:

| Check | Required? | Notes |
| --- | --- | --- |
| Code or document change is committed or clearly listed as uncommitted work | Yes | Team must know what changed. |
| Tests or checks were run | Yes | If not possible, explain why. |
| New generated data requirement is documented | Yes, if relevant | Especially after sampling/labeling changes. |
| Implementation matrix is updated | Yes, for method changes | Keep status and evidence current. |
| Team change log is updated | Yes | Use readable language. |
| AI-dialog log is updated | Yes, if AI was used | Summarize decisions and caveats, not every message. |

## Required Notes for Method Changes

When changing one of these areas, add extra detail:

| Area | Must record |
| --- | --- |
| Sampling | Sampler class, threshold/grid, target bars/day, whether OHLCV is aggregated, train-only fit evidence. |
| Features | Whether features are computed before or after sampling, feature list, scaler fit data, feature selector fit data. |
| Labeling | Barrier width, barrier height, dynamic/static barriers, label distribution, time-exit rate, label metadata fields. |
| Splits | Train/val/test boundaries, purged counts, embargo length, whether any labels cross boundaries. |
| Model | Architecture, lookback, class mapping, loss, optimizer, epochs, seed, checkpoint metric. |
| Decision policy | Meta model, meta features, threshold, take rate, probability calibration if used. |
| Backtest | Entry convention, barrier search convention, transaction cost, overlap rule, capital/exposure rule. |
| Logging | W&B run name, metric namespace used for the conclusion, whether `paper/*` or `portfolio/*` is treated as the main economic evidence. |

## Current Open Method Questions

| Question | Why it matters | Suggested next evidence |
| --- | --- | --- |
| Does corrected next-bar entry materially change label balance and backtest economics? | Old prepared labels/backtests used optimistic timing for aggregated bars. | Regenerate folds and compare old vs new metrics. |
| Which Triple Barrier width/height works for US equities? | The current defaults may create too many exits and weak actionable samples. | Run barrier sweep and log tradeable share plus economics. |
| Is tuned bars/day CUSUM better than fixed-threshold CUSUM? | Current sampler is useful but not paper-faithful. | Implement fixed thresholds and run controlled ablations. |
| Does ResNet-LSTM outperform Conv1D on corrected data? | The reference model family is stronger than the current fast baseline. | Run same folds, same costs, same decision policy, multiple seeds. |
| Is meta-labeling better than a simple probability band? | Meta-labeling adds complexity and must prove value. | Compare meta policy to fixed probability band under identical backtest assumptions. |
| How different are `paper/*` and `portfolio/*` conclusions? | The stricter portfolio simulator is now available, but the team still needs experiment evidence showing whether budget constraints materially change conclusions. | Run corrected folds and compare final balance, drawdown, exposure, skipped-budget count, and diagnostic paper return. |

## File Maintenance Rules

1. Keep this file readable. Prefer concise rows over long paragraphs.
2. Put detailed experiment analysis in `reports/` and link it from this file.
3. When a table becomes too long, move old entries into `reports/archive/` and keep a summary here.
4. Never remove a failed experiment entry just because the result was bad.
5. If a team decision changes, add a new row instead of silently editing history.

# AI Dialog Log

Last updated: 2026-06-12

Purpose: record important AI-assisted project discussions in a readable way. This is not a transcript; it is a decision and traceability log.

| Date | Tool / participant | Topic | Useful output | Human/team decision | Follow-up |
| --- | --- | --- | --- | --- | --- |
| 2026-04-27 | Codex | Backtest realism and portfolio balance | Suggested replacing trade-only compounding as the final economic claim with a cash/exposure-constrained portfolio simulator while keeping paper metrics as diagnostics. | Implemented portfolio-account metrics, W&B equity-curve logging, and documentation updates. | Compare `portfolio/*` and `paper/*` on the next corrected-fold experiment. |
| 2026-04-28 | Codex | Latest experiment and next grid | Analysed the latest ResNet-LSTM fixed-CUSUM run and suggested calibrating CUSUM/barrier/meta thresholds before scaling architecture. | Set up a dry-run-first grid runner with Conv1D as the first model and ResNet-LSTM only for promising configurations. | Run the preparation grid, execute Conv1D batches, then fill `reports/promising_grid_configs.json`. |
| 2026-06-12 | Codex | Paper-default reset | Extracted the 2025 paper's main baseline parameters from the local PDF and mapped them into the repo's preparation pipeline, including support for a `24`-period vertical barrier instead of only minute-based barriers. | Adopt the paper-aligned defaults for future prepared folds, but keep the shorter `4`-quarter warmup as an explicit project choice rather than claim a literal reproduction. | Regenerate prepared folds and rerun the paper-aligned grid before comparing new economics to legacy reports. |

## Entry Template

| Date | Tool / participant | Topic | Useful output | Human/team decision | Follow-up |
| --- | --- | --- | --- | --- | --- |
| YYYY-MM-DD | Name/tool | What was discussed? | What was worth keeping? | What did the team decide? | What needs to happen next? |

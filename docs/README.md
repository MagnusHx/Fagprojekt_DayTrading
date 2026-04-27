# Documentation

The project uses MkDocs for human-facing documentation.

Key coordination documents:

- `../reports/reference_implementation_matrix.md` tracks reference methods, implementation status, test status, evidence, priority, and next actions.
- `team_method.md` defines the team workflow, alignment rules, definition of done, and open method questions.
- `ai_dialog_log.md` records AI usage summaries and human decisions.
- `team_change_log.md` keeps a readable project change history.
- `experiment_log.md` tracks experiment cards, run logs, and metric summaries.
- `../reports/Kvant_Project_Description.docx` is the readable Word-format project description for hand-in/review.

Current reporting conventions:

- Use `portfolio/*` metrics for final economic claims because they simulate one cash account with exposure limits.
- Use `paper/*` metrics as diagnostic trade-level economics for comparison with earlier experiments.
- Record important W&B runs, local commands, and conclusions in the team logs instead of relying on dashboard memory.

Build locally with:

```bash
uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build
```

Serve locally with:

```bash
uv run mkdocs serve --config-file docs/mkdocs.yaml
```

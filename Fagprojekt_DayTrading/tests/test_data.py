import json

import pytest

from kvant.labels import label_semantics_payload, validate_label_semantics
from kvant.ml_prepare_data.data_loading import PreparedExperiment


def test_validate_label_semantics_matches_current_mapping() -> None:
    """The current runtime label semantics should validate cleanly."""
    validate_label_semantics({"label_semantics": label_semantics_payload()})


def test_prepared_experiment_rejects_missing_label_semantics(tmp_path) -> None:
    """Prepared experiments without semantics metadata should fail fast."""
    exp_dir = tmp_path / "prepared_exp"
    exp_dir.mkdir()
    (exp_dir / "config.json").write_text(json.dumps({"lookback_L": 12}))

    with pytest.raises(RuntimeError, match="label semantics"):
        PreparedExperiment(exp_dir)


def test_prepared_experiment_rejects_mismatched_label_semantics(tmp_path) -> None:
    """Prepared experiments with stale semantics should fail fast."""
    exp_dir = tmp_path / "prepared_exp"
    exp_dir.mkdir()
    (exp_dir / "config.json").write_text(
        json.dumps(
            {
                "lookback_L": 12,
                "label_semantics": {
                    "version": 1,
                    "labels": {"0": "up", "1": "exit", "2": "down"},
                },
            }
        )
    )

    with pytest.raises(RuntimeError, match="Regenerate the prepared data"):
        PreparedExperiment(exp_dir)

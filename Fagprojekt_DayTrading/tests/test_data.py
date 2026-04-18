import json

import numpy as np

import pytest

from kvant.labels import (
    label_ids_from_semantics,
    label_semantics_payload,
    model_labels_to_trade_labels,
    validate_label_semantics,
)
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


def test_validate_label_semantics_accepts_binary_directional_mapping() -> None:
    """Directional-only prepared artifacts should validate cleanly."""
    semantics = validate_label_semantics({"label_semantics": label_semantics_payload(drop_time_exit_label=True)})
    assert label_ids_from_semantics(semantics) == (0, 1)


def test_model_labels_to_trade_labels_maps_binary_up_to_canonical_up() -> None:
    """Binary model labels should map into the canonical trade label space."""
    y = np.asarray([0, 1, 1, 0], dtype=np.int64)
    out = model_labels_to_trade_labels(y, label_semantics_payload(drop_time_exit_label=True))
    np.testing.assert_array_equal(out, np.asarray([0, 2, 2, 0], dtype=np.int64))

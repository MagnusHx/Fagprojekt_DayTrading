import json

import numpy as np

import pytest

from kvant.labels import (
    label_ids_from_semantics,
    label_semantics_payload,
    model_labels_to_trade_labels,
    validate_label_semantics,
)
from kvant.ml_framework.run_validation import RunValidationError, validate_cv_manifest, validate_prepared_experiment
from kvant.ml_framework.scripts.smoke_prepared_experiment import _smoke_one
from kvant.ml_prepare_data.data_loading import PreparedExperiment


def _write_ticker_fixture(exp_dir, ticker: str, *, labels: list[int], market_rows: int | None = None) -> None:
    tdir = exp_dir / "tickers" / ticker
    tdir.mkdir(parents=True)
    n = len(labels)
    features = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    timestamps = np.asarray(
        [np.datetime64("2024-01-01T00:00:00") + np.timedelta64(i, "m") for i in range(n)],
        dtype="datetime64[ns]",
    )
    label_meta = [{"label": int(y)} for y in labels]
    np.save(tdir / "features.npy", features)
    np.save(tdir / "labels.npy", np.asarray(labels, dtype=np.int64))
    np.save(tdir / "timestamps.npy", timestamps)
    if market_rows is not None:
        market = np.ones((market_rows, 5), dtype=np.float32)
        np.save(tdir / "market_data.npy", market)
    (tdir / "label_metadata.jsonl").write_text("\n".join(json.dumps(row) for row in label_meta))


def _write_prepared_fixture(tmp_path, *, binary: bool = True, with_market_data: bool = True):
    exp_dir = tmp_path / "prepared_exp"
    exp_dir.mkdir()
    (exp_dir / "tickers").mkdir()
    config = {
        "lookback_L": 2,
        "feature_engineer": {
            "feature_names_": ["f0", "f1", "f2"],
            "mean_": [0.0, 0.0, 0.0],
            "std_": [1.0, 1.0, 1.0],
        },
        "labeler": {"width_minutes": 60, "height": 0.01, "drop_time_exit_label": binary},
        "label_semantics": label_semantics_payload(drop_time_exit_label=binary),
    }
    (exp_dir / "config.json").write_text(json.dumps(config))
    (exp_dir / "tickers_all.json").write_text(json.dumps(["AAA"]))
    labels = [0, 1, 1, 0] if binary else [0, 1, 2, 0]
    _write_ticker_fixture(exp_dir, "AAA", labels=labels, market_rows=(len(labels) if with_market_data else None))
    np.save(exp_dir / "index_train.npy", np.asarray([[0, 2]], dtype=np.int64))
    np.save(exp_dir / "index_val.npy", np.asarray([[0, 3]], dtype=np.int64))
    np.save(exp_dir / "index_test.npy", np.asarray([[0, 1]], dtype=np.int64))
    return exp_dir


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


def test_validate_prepared_experiment_returns_diagnostics_for_binary_fixture(tmp_path) -> None:
    exp_dir = _write_prepared_fixture(tmp_path, binary=True, with_market_data=True)

    diagnostics = validate_prepared_experiment(exp_dir, require_market_data=True)

    assert diagnostics.label_regime == "binary"
    assert diagnostics.n_classes == 2
    assert diagnostics.has_market_data is True
    assert diagnostics.split_summaries["train"]["n"] == 1


def test_validate_prepared_experiment_fails_when_market_data_is_required(tmp_path) -> None:
    exp_dir = _write_prepared_fixture(tmp_path, binary=True, with_market_data=False)

    with pytest.raises(RunValidationError, match="market_data integrity failed"):
        validate_prepared_experiment(exp_dir, require_market_data=True)


def test_smoke_one_materializes_batch_and_forward_pass(tmp_path) -> None:
    exp_dir = _write_prepared_fixture(tmp_path, binary=False, with_market_data=True)

    report = _smoke_one(exp_dir, model_name="conv1d", batch_size=2, require_market_data=True)

    assert report["n_classes"] == 3
    assert report["batch_shape"][0] == 1
    assert report["logits_shape"] == [1, 3]


def test_validate_cv_manifest_accepts_pointer_txt(tmp_path) -> None:
    exp_dir = _write_prepared_fixture(tmp_path, binary=True, with_market_data=True)
    manifest_json = tmp_path / "manifest.json"
    manifest_json.write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "fold_idx": 0,
                        "exp_id": exp_dir.name,
                        "exp_dir": str(exp_dir),
                    }
                ]
            }
        )
    )
    manifest_ptr = tmp_path / "manifest.txt"
    manifest_ptr.write_text(str(manifest_json))

    out = validate_cv_manifest(manifest_ptr, require_market_data=True)

    assert out["n_folds"] == 1
    assert out["folds"][0]["exp_id"] == exp_dir.name

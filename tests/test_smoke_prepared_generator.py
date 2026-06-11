from __future__ import annotations

import json

from kvant.ml_framework.run_validation import validate_cv_manifest, validate_prepared_experiment
from kvant.ml_framework.scripts.create_smoke_prepared_experiment import create_smoke_prepared_experiment


def test_create_smoke_prepared_experiment_writes_valid_one_fold_artifact(tmp_path) -> None:
    """The synthetic smoke generator should produce a valid fold and CV manifest."""
    exp_dir, manifest_path = create_smoke_prepared_experiment(
        out_root=tmp_path,
        label="smoke_test",
        row_count=48,
        lookback=8,
        seed=123,
        overwrite=False,
    )

    diagnostics = validate_prepared_experiment(exp_dir, require_market_data=True)
    manifest = validate_cv_manifest(manifest_path, require_market_data=True)

    assert diagnostics.label_regime == "event_outcome"
    assert diagnostics.n_classes == 3
    assert diagnostics.has_market_data is True
    assert manifest["n_folds"] == 1
    payload = json.loads(manifest_path.read_text())
    assert payload["folds"][0]["exp_dir"] == str(exp_dir)

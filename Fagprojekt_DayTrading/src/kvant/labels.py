from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any

LABEL_DOWN = 0
LABEL_EXIT = 1
LABEL_UP = 2

ACTED_LABELS = (LABEL_DOWN, LABEL_UP)

SIDE_LABEL_DOWN = 0
SIDE_LABEL_UP = 1
SIDE_LABEL_IGNORE = -1

META_LABEL_PASS = 0
META_LABEL_TAKE = 1

LABEL_MEANINGS = {
    LABEL_DOWN: "down barrier hit",
    LABEL_EXIT: "vertical/time exit",
    LABEL_UP: "up barrier hit",
}

CLASS_NAMES = [
    "Down barrier hit (y=0)",
    "Time exit (y=1)",
    "Up barrier hit (y=2)",
]


THREE_CLASS_LABELS = {
    str(LABEL_DOWN): "down",
    str(LABEL_EXIT): "exit",
    str(LABEL_UP): "up",
}

SIDE_LABELS = {
    "0": "down",
    "1": "up",
}

BINARY_DIRECTIONAL_LABELS = SIDE_LABELS

META_LABELS = {
    str(META_LABEL_PASS): "pass",
    str(META_LABEL_TAKE): "take",
}


def pipeline_label_spaces_payload() -> dict[str, dict[str, str]]:
    """Return all label spaces used by the Lopez de Prado pipeline."""
    return {
        "event_outcome_labels": dict(THREE_CLASS_LABELS),
        "side_labels": dict(SIDE_LABELS),
        "meta_labels": dict(META_LABELS),
    }


def label_semantics_payload(*, drop_time_exit_label: bool = False) -> dict[str, Any]:
    """Return the source-of-truth label semantics stored in prepared artifacts."""
    return {
        "version": 1,
        "labels": SIDE_LABELS if drop_time_exit_label else THREE_CLASS_LABELS,
    }


def normalize_label_semantics(label_semantics: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate supported label semantics payloads."""
    if not isinstance(label_semantics, dict):
        raise RuntimeError(f"Expected label semantics to be a dict, got {type(label_semantics).__name__}.")

    version = int(label_semantics.get("version", -1))
    labels = label_semantics.get("labels")
    if version != 1 or not isinstance(labels, dict):
        raise RuntimeError(f"Unsupported label semantics payload: {label_semantics}")

    normalized = {
        "version": 1,
        "labels": {str(int(k)): str(v) for k, v in sorted(labels.items(), key=lambda kv: int(kv[0]))},
    }
    if normalized["labels"] not in (THREE_CLASS_LABELS, SIDE_LABELS):
        raise RuntimeError(f"Unsupported label semantics payload: {label_semantics}")
    return normalized


def validate_label_semantics(config: dict[str, Any], *, exp_dir: Path | None = None) -> dict[str, Any]:
    """Validate that a prepared artifact matches a supported runtime label semantics."""
    actual = config.get("label_semantics")
    try:
        return normalize_label_semantics(actual)
    except RuntimeError:
        pass

    location = f" in {exp_dir}" if exp_dir is not None else ""
    raise RuntimeError(
        "Prepared experiment label semantics do not match the current code"
        f"{location}. Supported payloads are {label_semantics_payload()} and "
        f"{label_semantics_payload(drop_time_exit_label=True)}, got {actual}. Regenerate the prepared data."
    )


def label_ids_from_semantics(label_semantics: dict[str, Any]) -> tuple[int, ...]:
    semantics = normalize_label_semantics(label_semantics)
    return tuple(int(k) for k in semantics["labels"].keys())


def is_directional_binary_semantics(label_semantics: dict[str, Any]) -> bool:
    semantics = normalize_label_semantics(label_semantics)
    return semantics["labels"] == SIDE_LABELS


def label_meanings_from_semantics(label_semantics: dict[str, Any]) -> dict[int, str]:
    semantics = normalize_label_semantics(label_semantics)
    labels = semantics["labels"]
    if labels == THREE_CLASS_LABELS:
        return dict(LABEL_MEANINGS)
    return {
        SIDE_LABEL_DOWN: "down barrier hit",
        SIDE_LABEL_UP: "up barrier hit",
    }


def class_names_from_semantics(label_semantics: dict[str, Any]) -> list[str]:
    meanings = label_meanings_from_semantics(label_semantics)
    return [f"{meaning.title()} (y={label})" for label, meaning in meanings.items()]


def model_label_to_trade_label(label: int, label_semantics: dict[str, Any]) -> int:
    """Map a model-space label into the canonical trade-label space."""
    if is_directional_binary_semantics(label_semantics):
        if int(label) == SIDE_LABEL_DOWN:
            return LABEL_DOWN
        if int(label) == SIDE_LABEL_UP:
            return LABEL_UP
        raise ValueError(f"Binary directional label must be 0 or 1, got {label}.")
    return int(label)


def model_labels_to_trade_labels(labels: Any, label_semantics: dict[str, Any]):
    """Vectorized variant of model_label_to_trade_label."""
    import numpy as np

    arr = np.asarray(labels, dtype=np.int64)
    if not is_directional_binary_semantics(label_semantics):
        return arr

    out = np.empty_like(arr)
    out[arr == SIDE_LABEL_DOWN] = LABEL_DOWN
    out[arr == SIDE_LABEL_UP] = LABEL_UP
    invalid = ~np.isin(arr, (SIDE_LABEL_DOWN, SIDE_LABEL_UP))
    if np.any(invalid):
        raise ValueError(f"Binary directional labels must only contain 0/1, got {arr[invalid][:5]}.")
    return out


def event_label_from_metadata(metadata: dict[str, Any] | None, *, fallback: int | None = None) -> int:
    """Read the canonical triple-barrier event label from persisted metadata."""
    if metadata is not None and isinstance(metadata.get("label"), numbers.Number):
        return int(metadata["label"])
    if fallback is not None:
        return int(fallback)
    raise RuntimeError("Missing canonical event label in persisted label metadata.")


def event_labels_to_side_labels(labels: Any, *, neutral_value: int = SIDE_LABEL_IGNORE):
    """Map triple-barrier event labels into side labels, keeping neutral rows ignored."""
    import numpy as np

    arr = np.asarray(labels, dtype=np.int64)
    out = np.full(arr.shape, int(neutral_value), dtype=np.int64)
    out[arr == LABEL_DOWN] = SIDE_LABEL_DOWN
    out[arr == LABEL_UP] = SIDE_LABEL_UP
    invalid = ~np.isin(arr, (LABEL_DOWN, LABEL_EXIT, LABEL_UP))
    if np.any(invalid):
        raise ValueError(f"Event labels must only contain 0/1/2, got {arr[invalid][:5]}.")
    return out


def side_labels_to_trade_labels(labels: Any):
    """Map side-model labels into canonical trade-label space."""
    import numpy as np

    arr = np.asarray(labels, dtype=np.int64)
    out = np.empty_like(arr)
    out[arr == SIDE_LABEL_DOWN] = LABEL_DOWN
    out[arr == SIDE_LABEL_UP] = LABEL_UP
    invalid = ~np.isin(arr, (SIDE_LABEL_DOWN, SIDE_LABEL_UP))
    if np.any(invalid):
        raise ValueError(f"Side labels must only contain 0/1, got {arr[invalid][:5]}.")
    return out

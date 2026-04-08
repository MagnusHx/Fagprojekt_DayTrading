from __future__ import annotations

from pathlib import Path
from typing import Any

LABEL_DOWN = 0
LABEL_EXIT = 1
LABEL_UP = 2

ACTED_LABELS = (LABEL_DOWN, LABEL_UP)

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


def label_semantics_payload() -> dict[str, Any]:
    """Return the source-of-truth label semantics stored in prepared artifacts."""
    return {
        "version": 1,
        "labels": {
            str(LABEL_DOWN): "down",
            str(LABEL_EXIT): "exit",
            str(LABEL_UP): "up",
        },
    }


def validate_label_semantics(config: dict[str, Any], *, exp_dir: Path | None = None) -> None:
    """Validate that a prepared artifact matches the current label semantics."""
    expected = label_semantics_payload()
    actual = config.get("label_semantics")
    if actual == expected:
        return

    location = f" in {exp_dir}" if exp_dir is not None else ""
    raise RuntimeError(
        "Prepared experiment label semantics do not match the current code"
        f"{location}. Expected {expected}, got {actual}. Regenerate the prepared data."
    )

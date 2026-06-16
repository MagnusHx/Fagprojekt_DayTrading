"""Shared Weights & Biases defaults for all experiment scripts."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv


load_dotenv()

DEFAULT_WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "day-trading-experiments")
DEFAULT_WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "s245509-danmarks-tekniske-universitet-dtu")


def wandb_init_kwargs(*, project: str | None = None, entity: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Return consistent wandb.init keyword arguments."""
    resolved_entity = DEFAULT_WANDB_ENTITY if entity is None else str(entity)
    init_kwargs: dict[str, Any] = {
        "project": DEFAULT_WANDB_PROJECT if project is None else str(project),
        **kwargs,
    }
    if resolved_entity:
        init_kwargs["entity"] = resolved_entity
    return init_kwargs

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
from sklearn.feature_selection import f_classif


class FeatureSelector(Protocol):
    name: str

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: Sequence[str],
    ) -> "FeatureSelector":
        ...

    def transform(
        self,
        X: np.ndarray,
        feature_names: Sequence[str],
    ) -> tuple[np.ndarray, list[str]]:
        ...

    def get_meta(self) -> dict:
        ...


@dataclass
class PrimarySideFScoreSelector:
    """Train-only supervised feature selector for the primary-side model."""

    top_k: int = 16
    min_score: float = 0.0
    name: str = "primary_side_fscore_topk"

    feature_names_: list[str] | None = field(default=None, init=False)
    selected_indices_: list[int] | None = field(default=None, init=False)
    selected_feature_names_: list[str] | None = field(default=None, init=False)
    scores_: list[float] | None = field(default=None, init=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: Sequence[str],
    ) -> "PrimarySideFScoreSelector":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        names = [str(name) for name in feature_names]

        if X.ndim != 2:
            raise ValueError(f"Feature selector expected X to be 2D, got shape {tuple(X.shape)}.")
        if len(X) != len(y):
            raise ValueError(f"Feature selector expected X and y to align, got {len(X)} and {len(y)}.")
        if X.shape[1] != len(names):
            raise ValueError(
                f"Feature selector expected {X.shape[1]} feature names, got {len(names)}."
            )
        if len(X) == 0:
            raise RuntimeError("Cannot fit feature selector without training rows.")

        self.feature_names_ = names
        scores, _ = f_classif(X, y)
        scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        ranked = np.argsort(scores)[::-1]
        positive = ranked[scores[ranked] > float(self.min_score)]
        if len(positive) == 0:
            positive = ranked[:1]

        top_k = max(1, min(int(self.top_k), X.shape[1]))
        chosen = np.sort(np.asarray(positive[:top_k], dtype=np.int64))

        self.selected_indices_ = chosen.tolist()
        self.selected_feature_names_ = [names[idx] for idx in chosen]
        self.scores_ = [float(scores[idx]) for idx in chosen]
        return self

    def transform(
        self,
        X: np.ndarray,
        feature_names: Sequence[str],
    ) -> tuple[np.ndarray, list[str]]:
        if self.selected_indices_ is None or self.feature_names_ is None:
            raise RuntimeError("Feature selector transform called before fit().")

        names = [str(name) for name in feature_names]
        if names != self.feature_names_:
            raise RuntimeError("Feature names changed between feature-selection fit and transform.")

        chosen = np.asarray(self.selected_indices_, dtype=np.int64)
        X = np.asarray(X)
        return X[:, chosen], [names[idx] for idx in chosen]

    def get_meta(self) -> dict:
        return {
            "name": self.name,
            "top_k": int(self.top_k),
            "min_score": float(self.min_score),
            "n_selected": 0 if self.selected_indices_ is None else int(len(self.selected_indices_)),
            "selected_indices": None if self.selected_indices_ is None else list(self.selected_indices_),
            "selected_feature_names": None
            if self.selected_feature_names_ is None
            else list(self.selected_feature_names_),
            "scores": None if self.scores_ is None else [float(score) for score in self.scores_],
        }

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kvant.labels import LABEL_UP, META_LABEL_TAKE, side_labels_to_trade_labels
from kvant.ml_prepare_data.data_loading import PreparedStore


DEFAULT_META_FEATURES = ("proba", "embedding")


def normalize_meta_features(features: Sequence[str] | None) -> tuple[str, ...]:
    if not features:
        return DEFAULT_META_FEATURES

    normalized: list[str] = []
    for raw_token in features:
        for token in str(raw_token).split(","):
            token = token.strip()
            if token:
                normalized.append(token)
    if not normalized:
        return DEFAULT_META_FEATURES
    return tuple(normalized)


def validate_meta_features(features: Iterable[str]) -> tuple[str, ...]:
    normalized = normalize_meta_features(tuple(features))
    for token in normalized:
        if token in {"proba", "logits", "embedding"}:
            continue
        if token.startswith("prepared_last:") and token.split(":", 1)[1].strip():
            continue
        raise ValueError(
            "meta_features entries must be one of "
            "'proba', 'logits', 'embedding', or 'prepared_last:<feature_name>'. "
            f"Got {token!r}."
        )
    return normalized


def meta_targets_from_predictions(
    *,
    pred_out: dict[str, Any],
    store: PreparedStore,
) -> np.ndarray:
    """Return 1 when the proposed side would have produced a positive realized return."""
    index = np.stack(
        [
            np.asarray(pred_out["tid"], dtype=np.int64),
            np.asarray(pred_out["tpos"], dtype=np.int64),
        ],
        axis=1,
    ).astype(np.int64, copy=False)
    metas = store.metadata_for_index(index)
    trade_labels = side_labels_to_trade_labels(pred_out["y_pred"])

    out = np.zeros(len(trade_labels), dtype=np.int64)
    for i, (meta, trade_label) in enumerate(zip(metas, trade_labels)):
        if meta is None:
            continue
        pnl_fraction = meta.get("pnl_fraction")
        if pnl_fraction is None:
            continue
        try:
            pnl_fraction = float(pnl_fraction)
        except Exception:
            continue
        signed_return = pnl_fraction if int(trade_label) == LABEL_UP else -pnl_fraction
        out[i] = int(signed_return > 0.0)
    return out


@dataclass
class LogisticMetaLabeler:
    feature_tokens: tuple[str, ...] = DEFAULT_META_FEATURES
    random_state: int = 1337
    C: float = 1.0
    max_iter: int = 1000

    def __post_init__(self) -> None:
        self.feature_tokens = validate_meta_features(self.feature_tokens)
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        C=float(self.C),
                        solver="lbfgs",
                        max_iter=int(self.max_iter),
                        class_weight="balanced",
                        random_state=int(self.random_state),
                    ),
                ),
            ]
        )
        self._constant_take_proba: float | None = None

    def build_feature_matrix(
        self,
        *,
        pred_out: dict[str, Any],
        store: PreparedStore,
    ) -> np.ndarray:
        blocks: list[np.ndarray] = []
        tids = np.asarray(pred_out["tid"], dtype=np.int64)
        tpos = np.asarray(pred_out["tpos"], dtype=np.int64)

        for token in self.feature_tokens:
            if token == "proba":
                blocks.append(np.asarray(pred_out["y_pred_proba"], dtype=np.float64))
                continue
            if token == "logits":
                blocks.append(np.asarray(pred_out["y_logits"], dtype=np.float64))
                continue
            if token == "embedding":
                blocks.append(np.asarray(pred_out["y_embedding"], dtype=np.float64))
                continue
            feature_name = token.split(":", 1)[1]
            values = store.prepared_last_feature_values(tids=tids, tpos=tpos, feature_name=feature_name)
            blocks.append(values.astype(np.float64, copy=False).reshape(-1, 1))

        if not blocks:
            raise RuntimeError("No meta-label features were assembled.")
        return np.concatenate(blocks, axis=1)

    def fit(
        self,
        *,
        pred_out: dict[str, Any],
        store: PreparedStore,
    ) -> "LogisticMetaLabeler":
        X = self.build_feature_matrix(pred_out=pred_out, store=store)
        y_meta = meta_targets_from_predictions(pred_out=pred_out, store=store)
        if len(y_meta) == 0:
            raise RuntimeError("No samples available to fit the logistic meta-label model.")

        unique = np.unique(y_meta)
        if unique.size < 2:
            self._constant_take_proba = float(unique[0]) if unique.size == 1 else 0.0
            return self

        self._constant_take_proba = None
        self.pipeline.fit(X, y_meta)
        return self

    def predict_take_proba(
        self,
        *,
        pred_out: dict[str, Any],
        store: PreparedStore,
    ) -> np.ndarray:
        X = self.build_feature_matrix(pred_out=pred_out, store=store)
        if self._constant_take_proba is not None:
            return np.full(len(X), float(self._constant_take_proba), dtype=np.float64)

        proba = self.pipeline.predict_proba(X)
        if proba.ndim != 2 or proba.shape[1] != 2:
            raise RuntimeError(f"Expected binary predict_proba output, got shape {tuple(proba.shape)}.")
        return np.asarray(proba[:, META_LABEL_TAKE], dtype=np.float64)

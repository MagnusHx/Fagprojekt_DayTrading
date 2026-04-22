from __future__ import annotations

from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _forward_with_optional_features(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "forward_features") and hasattr(model, "forward_logits_from_features"):
        features = model.forward_features(x)
        logits = model.forward_logits_from_features(features)
        return logits, features
    logits = model(x)
    return logits, logits


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    yt, yp, y_conf, y_probs, y_logits, y_embedding, tids, tposs = [], [], [], [], [], [], [], []

    for batch in loader:
        x, y, tid, tpos = batch

        x = x.to(device, non_blocking=True)
        logits, embedding = _forward_with_optional_features(model, x)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        conf = torch.gather(probs, 1, pred.unsqueeze(1)).squeeze(1).detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()

        yt.append(y.detach().cpu().numpy())
        yp.append(pred)
        y_conf.append(conf)
        y_probs.append(probs.detach().cpu().numpy())
        y_logits.append(logits.detach().cpu().numpy())
        y_embedding.append(embedding.detach().cpu().numpy())

        tids.append(tid.detach().cpu().numpy())
        tposs.append(tpos.detach().cpu().numpy())

    out: Dict[str, Any] = {
        "y_true": np.concatenate(yt) if yt else np.asarray([], dtype=np.int64),
        "y_pred": np.concatenate(yp) if yp else np.asarray([], dtype=np.int64),
        "y_pred_confidence": np.concatenate(y_conf) if y_conf else np.asarray([], dtype=np.float32),
        "y_pred_proba": np.concatenate(y_probs) if y_probs else np.asarray([], dtype=np.float32).reshape(0, 0),
        "y_logits": np.concatenate(y_logits) if y_logits else np.asarray([], dtype=np.float32).reshape(0, 0),
        "y_embedding": np.concatenate(y_embedding) if y_embedding else np.asarray([], dtype=np.float32).reshape(0, 0),
    }
    out["tid"] = np.concatenate(tids).astype(np.int64, copy=False)
    out["tpos"] = np.concatenate(tposs).astype(np.int64, copy=False)
    return out

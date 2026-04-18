from __future__ import annotations

from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    yt, yp, y_conf, y_probs, tids, tposs = [], [], [], [], [], []

    for batch in loader:
        x, y, tid, tpos = batch

        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        conf = torch.gather(probs, 1, pred.unsqueeze(1)).squeeze(1).detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()

        yt.append(y.detach().cpu().numpy())
        yp.append(pred)
        y_conf.append(conf)
        y_probs.append(probs.detach().cpu().numpy())

        tids.append(tid.detach().cpu().numpy())
        tposs.append(tpos.detach().cpu().numpy())

    out: Dict[str, Any] = {
        "y_true": np.concatenate(yt) if yt else np.asarray([], dtype=np.int64),
        "y_pred": np.concatenate(yp) if yp else np.asarray([], dtype=np.int64),
        "y_pred_confidence": np.concatenate(y_conf) if y_conf else np.asarray([], dtype=np.float32),
        "y_pred_proba": np.concatenate(y_probs) if y_probs else np.asarray([], dtype=np.float32).reshape(0, 0),
    }
    out["tid"] = np.concatenate(tids).astype(np.int64, copy=False)
    out["tpos"] = np.concatenate(tposs).astype(np.int64, copy=False)
    return out

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from kvant.ml_prepare_data.data_loading import PreparedExperiment
from kvant.ml_framework.models import create_model
from kvant.ml_framework.train import EvalConfig, ExperimentEvaluator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True, help="Best-checkpoint bundle produced by train_experiment.py")
    p.add_argument("--wandb-summary", type=Path, default=None, help="Optional wandb-summary.json to compare against.")
    p.add_argument(
        "--summary-prefix",
        type=str,
        default="best",
        help="Optional prefix used in wandb-summary keys, for example 'best' or 'fold04/best'.",
    )
    p.add_argument("--output", type=Path, default=None, help="Optional JSON output path for the reconciliation report.")
    return p.parse_args()


def _scalar_metrics(metrics: Dict[str, Any]) -> Dict[str, float | int]:
    out: Dict[str, float | int] = {}
    for key, value in metrics.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            out[str(key)] = float(value) if isinstance(value, (float, np.floating)) else int(value)
    return out


def _compare_metrics(
    computed: Dict[str, float | int],
    summary: Dict[str, Any],
    *,
    summary_prefix: str,
) -> Dict[str, Any]:
    prefix = summary_prefix.strip("/")
    if prefix:
        prefix = prefix + "/"

    matched = []
    missing_in_summary = []
    mismatches = []

    for key, computed_value in sorted(computed.items()):
        summary_key = f"{prefix}{key}"
        if summary_key not in summary:
            missing_in_summary.append(summary_key)
            continue

        summary_value = summary[summary_key]
        if isinstance(computed_value, int) and isinstance(summary_value, int):
            equal = computed_value == summary_value
            abs_diff = 0
        else:
            computed_float = float(computed_value)
            summary_float = float(summary_value)
            abs_diff = abs(computed_float - summary_float)
            equal = abs_diff <= 1e-9

        row = {
            "metric": key,
            "summary_key": summary_key,
            "computed": computed_value,
            "summary": summary_value,
            "abs_diff": abs_diff,
        }
        matched.append(row)
        if not equal:
            mismatches.append(row)

    return {
        "matched_count": int(len(matched)),
        "missing_in_summary": missing_in_summary,
        "mismatches": mismatches,
    }


def main() -> None:
    args = parse_args()
    bundle = torch.load(args.bundle, map_location="cpu")
    exp_dir = Path(bundle["exp_dir"])
    exp = PreparedExperiment(exp_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_name=str(bundle["model_name"]),
        n_features=exp.store.n_features,
        n_classes=exp.n_classes,
        **dict(bundle.get("model_kwargs", {})),
    ).to(device)
    model.load_state_dict(bundle["model_state"])

    dl_train, dl_val, dl_test = exp.get_loaders(
        train_batch_size=256,
        eval_batch_size=512,
        num_workers=0,
        pin_memory=True,
    )
    dl_train_eval = DataLoader(
        dl_train.dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    eval_cfg_payload = dict(bundle["eval_config"])
    eval_cfg_payload["labels"] = exp.label_ids
    eval_cfg_payload["label_semantics"] = exp.label_semantics
    evaluator = ExperimentEvaluator(
        store=exp.store,
        device=device,
        logger=None,
        cfg=EvalConfig(**eval_cfg_payload),
    )
    metrics = evaluator.evaluate_all(
        model,
        {"train": dl_train_eval, "val": dl_val, "test": dl_test},
        step=None,
    )
    scalars = _scalar_metrics(metrics)

    report: Dict[str, Any] = {
        "bundle": str(args.bundle),
        "exp_dir": str(exp_dir),
        "n_scalar_metrics": int(len(scalars)),
        "metrics": scalars,
    }

    if args.wandb_summary is not None:
        summary = json.loads(args.wandb_summary.read_text())
        report["comparison"] = _compare_metrics(scalars, summary, summary_prefix=args.summary_prefix)

    payload = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"Wrote reconciliation report to {args.output}")
    else:
        print(payload)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kvant.ml_framework.models import create_model
from kvant.ml_framework.run_validation import validate_cv_manifest, validate_prepared_experiment
from kvant.ml_prepare_data.data_loading import PreparedExperiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", type=Path, default=None)
    p.add_argument("--cv-manifest", type=Path, default=None)
    p.add_argument("--model", choices=("conv1d", "resnet_lstm"), default="conv1d")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--require-market-data", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def _smoke_one(exp_dir: Path, *, model_name: str, batch_size: int, require_market_data: bool) -> dict:
    diagnostics = validate_prepared_experiment(exp_dir, require_market_data=require_market_data)
    exp = PreparedExperiment(exp_dir)
    dl_train, dl_val, dl_test = exp.get_loaders(
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    x, y, tid, tpos = next(iter(dl_train))
    model = create_model(model_name=model_name, n_features=exp.store.n_features, n_classes=exp.n_classes)
    with torch.no_grad():
        logits = model(x)

    return {
        "exp_dir": str(exp_dir),
        "label_regime": diagnostics.label_regime,
        "n_classes": int(exp.n_classes),
        "n_features": int(exp.store.n_features),
        "batch_shape": list(x.shape),
        "labels_shape": list(y.shape),
        "logits_shape": list(logits.shape),
        "first_tid": int(tid[0].item()),
        "first_tpos": int(tpos[0].item()),
        "preflight": diagnostics.to_jsonable(),
        "train_rows": int(len(dl_train.dataset)),
        "val_rows": int(len(dl_val.dataset)),
        "test_rows": int(len(dl_test.dataset)),
    }


def main() -> None:
    args = parse_args()
    reports = []
    if args.cv_manifest is not None:
        manifest = validate_cv_manifest(args.cv_manifest, require_market_data=args.require_market_data)
        for fold in manifest["folds"]:
            reports.append(
                _smoke_one(
                    Path(fold["exp_dir"]),
                    model_name=args.model,
                    batch_size=args.batch_size,
                    require_market_data=args.require_market_data,
                )
            )
    elif args.exp_dir is not None:
        reports.append(
            _smoke_one(
                args.exp_dir,
                model_name=args.model,
                batch_size=args.batch_size,
                require_market_data=args.require_market_data,
            )
        )
    else:
        raise SystemExit("Provide either --exp-dir or --cv-manifest.")

    payload = json.dumps({"reports": reports}, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"Wrote smoke report to {args.output}")
    else:
        print(payload)


if __name__ == "__main__":
    main()

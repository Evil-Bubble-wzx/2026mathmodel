#!/usr/bin/env python3
"""demo7 — mask_probability sweep + regression-threshold neutral attack.

Two questions:
  1. Does the train-time missing-augmentation mask ratio (``mask_probability``)
     move the neutral-class recall?  Sweep 0.0 -> 0.75 on the frozen
     ``label_sm015`` config (label_smoothing=0.15, no focal, neutral_scale=1.0).
  2. Neutral is "confidently misclassified" and boundary tweaks already failed.
     Does classifying from the regression head (``|intensity| < eps -> neutral``)
     recover more neutral samples than softmax argmax?

Reuses ``demo1/code`` modules unchanged; writes ONLY into ``demo7/``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]   # demo7/code -> demo7 -> math_model
DEMO7_ROOT = Path(__file__).resolve().parents[1]  # demo7
sys.path.insert(0, str(REPO_ROOT / "demo1" / "code"))

from common import (  # noqa: E402
    CACHE_ROOT,
    FeatureScaler,
    infer_temporal_masks,
    load_aligned,
    set_seed,
)
from experiment_loss_sweep import FocalMultiTaskLoss  # noqa: E402
from problem2_train import (  # noqa: E402
    build_dataset,
    predict,
    train_epoch,
)
from robust_model import ModelConfig, RobustFusionModel  # noqa: E402

CLASS_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def per_class_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for label in (0, 1, 2):
        name = CLASS_NAMES[label]
        tp = int(((pred == label) & (true == label)).sum())
        fn = int(((pred != label) & (true == label)).sum())
        fp = int(((pred == label) & (true != label)).sum())
        out[f"{name}_recall"] = tp / (tp + fn) if (tp + fn) else 0.0
        out[f"{name}_precision"] = tp / (tp + fp) if (tp + fp) else 0.0
        out[f"{name}_f1"] = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return out


def reg_threshold_class(raw: np.ndarray, eps: float) -> np.ndarray:
    raw = np.asarray(raw)
    return np.where(raw < -eps, 0, np.where(raw > eps, 2, 1)).astype(np.int64)


def run_config(args: argparse.Namespace, mask_probability: float, tag: str) -> dict:
    set_seed(args.seed)
    device = resolve_device(args.device)
    cache_dir = CACHE_ROOT / "bert_features"

    aligned = load_aligned()
    train_masks = infer_temporal_masks(
        aligned["train"]["text_bert"],
        aligned["train"]["audio"],
        aligned["train"]["vision"],
    )
    scaler = FeatureScaler.fit(
        aligned["train"]["audio"], aligned["train"]["vision"], train_masks
    )

    datasets = {}
    for split in ("train", "valid", "test"):
        text_cache = np.load(cache_dir / f"{split}.npy", mmap_mode="r")
        datasets[split] = build_dataset(
            aligned[split]["id"], aligned[split], text_cache, scaler, labelled=True
        )
    loaders = {
        split: DataLoader(
            ds, batch_size=args.batch_size, shuffle=(split == "train"), num_workers=0
        )
        for split, ds in datasets.items()
    }

    model = RobustFusionModel(ModelConfig(hidden_dim=args.hidden_dim)).to(device)
    counts = np.bincount(
        np.asarray(aligned["train"]["classification_labels"], dtype=np.int64),
        minlength=3,
    )
    class_weights = np.sqrt(len(datasets["train"]) / (3.0 * counts))
    class_weights = class_weights / class_weights.mean()
    criterion = FocalMultiTaskLoss(
        torch.tensor(class_weights, dtype=torch.float32, device=device),
        focal_gamma=0.0,
        label_smoothing=0.15,
        consistency_weight=0.10,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=2e-6
    )
    amp_scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_score = -math.inf
    best_state = None
    best_epoch = -1
    wait = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        training_loss = train_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            amp_scaler,
            use_augmentation=True,
            mask_probability=mask_probability,
        )
        if not math.isfinite(training_loss):
            raise RuntimeError(f"Non-finite loss at mask_probability={mask_probability}")
        valid = predict(model, loaders["valid"], device)
        score = valid["metrics"]["selection_score"]
        scheduler.step(score)
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)

    valid = predict(model, loaders["valid"], device)
    test = predict(model, loaders["test"], device)
    missing = predict(
        model, loaders["valid"], device, scenario=(("text",), 0.30, "middle")
    )

    clean_valid = valid["metrics"]
    missing_valid = missing["metrics"]
    rel_drop = (
        (clean_valid["macro_f1"] - missing_valid["macro_f1"]) / clean_valid["macro_f1"]
        if clean_valid["macro_f1"] > 0
        else None
    )

    # Regression-threshold neutral attack on the test split (free — no retrain).
    raw = test["predicted_regression_raw"]
    true = test["true_class"]
    reg_threshold = {}
    for eps in (0.1, 0.2, 0.3, 0.5):
        pc = reg_threshold_class(raw, eps)
        reg_threshold[f"eps_{eps}"] = {
            "accuracy": float(accuracy_score(true, pc)),
            "macro_f1": float(f1_score(true, pc, average="macro", zero_division=0)),
            "neutral_recall": float(np.mean(pc[true == 1] == 1)),
            "neutral_precision": float(
                (pc[true == 1] == 1).sum() / max(1, int((pc == 1).sum()))
            ),
        }

    return {
        "tag": tag,
        "mask_probability": mask_probability,
        "elapsed_seconds": time.time() - start,
        "best_epoch": best_epoch,
        "device": str(device),
        "seed": args.seed,
        "valid": clean_valid,
        "valid_per_class": per_class_metrics(valid["true_class"], valid["predicted_class"]),
        "test": test["metrics"],
        "test_per_class": per_class_metrics(test["true_class"], test["predicted_class"]),
        "robustness_text_middle_30": {
            "valid_macro_f1": clean_valid["macro_f1"],
            "missing_macro_f1": missing_valid["macro_f1"],
            "macro_f1_rel_drop": rel_drop,
        },
        "reg_threshold": reg_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mask-probabilities",
        type=str,
        default="0.0,0.15,0.30,0.45,0.60,0.75",
        help="comma-separated mask_probability values to sweep",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEMO7_ROOT / "results" / "mask_sweep"),
    )
    args = parser.parse_args()

    values = [float(x) for x in args.mask_probabilities.split(",") if x.strip() != ""]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for mp in values:
        tag = f"mp_{mp:.2f}"
        print(f"\n===== mask_probability={mp:.2f} =====", flush=True)
        result = run_config(args, mp, tag)
        (out_dir / f"{tag}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tpc = result["test_per_class"]
        rt = result["reg_threshold"]
        summary_rows.append(
            {
                "mask_probability": mp,
                "best_epoch": result["best_epoch"],
                "elapsed_seconds": round(result["elapsed_seconds"], 1),
                "valid_macro_f1": round(result["valid"]["macro_f1"], 4),
                "valid_neutral_recall": round(result["valid_per_class"]["neutral_recall"], 4),
                "test_acc": round(result["test"]["accuracy"], 4),
                "test_macro_f1": round(result["test"]["macro_f1"], 4),
                "test_neutral_recall": round(tpc["neutral_recall"], 4),
                "test_neutral_f1": round(tpc["neutral_f1"], 4),
                "test_mae": round(result["test"]["mae"], 4),
                "test_pearson": round(result["test"]["pearson"], 4),
                "robust_rel_drop": (
                    round(result["robustness_text_middle_30"]["macro_f1_rel_drop"], 4)
                    if result["robustness_text_middle_30"]["macro_f1_rel_drop"] is not None
                    else None
                ),
                "reg_eps0.3_neutral_recall": round(rt["eps_0.3"]["neutral_recall"], 4),
                "reg_eps0.3_neutral_prec": round(rt["eps_0.3"]["neutral_precision"], 4),
                "reg_eps0.3_macro_f1": round(rt["eps_0.3"]["macro_f1"], 4),
            }
        )
        print(json.dumps(summary_rows[-1], ensure_ascii=False), flush=True)

    # Write a compact CSV summary.
    import csv

    summary_path = out_dir / "summary.csv"
    if summary_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()

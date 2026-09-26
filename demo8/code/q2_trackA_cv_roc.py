#!/usr/bin/env python3
"""Q2 Track A (Transformer) — 5-fold CV + OvR ROC + confusion matrix.

Reproduces the paper-chosen Track A configuration (label_smoothing=0.15 from
the loss sweep), runs stratified 5-fold CV on the train split ONLY (the frozen
test is never used for model selection), then trains one final model on full
train (valid as early-stop monitor) and predicts the frozen test split.

Emits: per-fold metrics, out-of-fold (OOF) probabilities, test probabilities,
per-class one-vs-rest ROC curves/AUC, and confusion matrices.
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "demo1" / "code"))

from common import (
    CACHE_ROOT,
    RESULTS_ROOT,
    FeatureScaler,
    infer_temporal_masks,
    load_aligned,
    set_seed,
    write_json,
)
from problem2_train import (
    build_dataset,
    metric_dict,
    predict,
    to_device,
    train_epoch,
)
from robust_model import ModelConfig, RobustFusionModel
from experiment_loss_sweep import FocalMultiTaskLoss

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

OUT = RESULTS_ROOT / "Q2" / "cv_roc" / "trackA"


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def slice_split(data: dict, idx: np.ndarray) -> dict:
    ids = np.asarray(data["id"], dtype=object)[idx]
    return {
        "id": [str(x) for x in ids],
        "text_bert": np.asarray(data["text_bert"])[idx],
        "audio": np.asarray(data["audio"])[idx],
        "vision": np.asarray(data["vision"])[idx],
        "classification_labels": np.asarray(data["classification_labels"]).reshape(-1)[idx],
        "regression_labels": np.asarray(data["regression_labels"]).reshape(-1)[idx],
    }


def class_weights_tensor(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=3)
    weights = np.sqrt(len(y) / (3.0 * counts))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_model(
    model: RobustFusionModel,
    criterion: FocalMultiTaskLoss,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[int, float]:
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
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model, train_loader, optimizer, criterion, device, amp_scaler,
            use_augmentation=True, mask_probability=args.mask_probability,
        )
        if not math.isfinite(loss):
            raise RuntimeError(f"Non-finite loss ({loss}) on {device.type}")
        validation = predict(model, valid_loader, device)
        score = validation["metrics"]["selection_score"]
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
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    return best_epoch, best_score


def full_metrics(true_class: np.ndarray, pred_class: np.ndarray,
                 true_reg: np.ndarray, pred_reg: np.ndarray) -> dict:
    metrics = metric_dict(true_class, pred_class, true_reg, pred_reg)
    metrics["weighted_f1"] = float(
        f1_score(true_class, pred_class, average="weighted")
    )
    return metrics


def ovr_roc(y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    n_classes = probs.shape[1]
    fpr_grid = np.linspace(0.0, 1.0, 201)
    tprs = []
    per_class = {}
    for c in range(n_classes):
        yb = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(yb, probs[:, c])
        auc = float(roc_auc_score(yb, probs[:, c]))
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        per_class[f"class_{c}"] = {
            "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc,
        }
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    return {
        "per_class": per_class,
        "per_class_auc": {str(c): per_class[f"class_{c}"]["auc"] for c in range(n_classes)},
        "macro_auc": macro_auc,
        "macro": {"fpr": fpr_grid.tolist(), "tpr": mean_tpr.tolist(), "auc": macro_auc},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--label-smoothing", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-probability", type=float, default=0.45)
    parser.add_argument("--hidden-dim", type=int, default=96)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device={device}", flush=True)

    aligned = load_aligned()
    train = aligned["train"]
    cache_dir = CACHE_ROOT / "bert_features"
    text_cache = np.load(cache_dir / "train.npy", mmap_mode="r")
    valid_cache = np.load(cache_dir / "valid.npy", mmap_mode="r")
    test_cache = np.load(cache_dir / "test.npy", mmap_mode="r")

    y = np.asarray(train["classification_labels"]).reshape(-1).astype(np.int64)
    yr = np.asarray(train["regression_labels"]).reshape(-1).astype(np.float32)
    n = len(y)

    OUT.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    oof_probs = np.full((n, 3), np.nan, dtype=np.float32)
    oof_class = np.full(n, -1, dtype=np.int64)
    oof_reg = np.full(n, np.nan, dtype=np.float32)
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(y, y)):
        t0 = time.time()
        set_seed(args.seed + fold)
        tr = slice_split(train, tr_idx)
        va = slice_split(train, va_idx)

        tr_masks = infer_temporal_masks(tr["text_bert"], tr["audio"], tr["vision"])
        scaler = FeatureScaler.fit(tr["audio"], tr["vision"], tr_masks)
        ds_tr = build_dataset(tr["id"], tr, text_cache[tr_idx], scaler, labelled=True)
        ds_va = build_dataset(va["id"], va, text_cache[va_idx], scaler, labelled=True)
        loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
        loader_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = RobustFusionModel(ModelConfig(hidden_dim=args.hidden_dim)).to(device)
        criterion = FocalMultiTaskLoss(
            class_weights_tensor(y[tr_idx], device),
            focal_gamma=0.0,
            label_smoothing=args.label_smoothing,
            consistency_weight=0.10,
        )
        best_epoch, best_score = train_model(
            model, criterion, loader_tr, loader_va, device, args
        )
        pred = predict(model, loader_va, device)
        oof_probs[va_idx] = pred["probabilities"]
        oof_class[va_idx] = pred["predicted_class"]
        oof_reg[va_idx] = pred["predicted_regression"]
        metrics = full_metrics(
            pred["true_class"], pred["predicted_class"],
            pred["true_regression"], pred["predicted_regression"],
        )
        fold_rows.append({
            "fold": fold, "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "train_rows": len(tr_idx), "valid_rows": len(va_idx),
            "elapsed_seconds": time.time() - t0,
            **metrics,
        })
        print(f"fold={fold} best_epoch={best_epoch} "
              f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
              f"weighted_f1={metrics['weighted_f1']:.4f} "
              f"mae={metrics['mae']:.4f} pearson={metrics['pearson']:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    oof_metrics = full_metrics(y, oof_class, yr, oof_reg)
    oof_roc = ovr_roc(y, oof_probs)
    oof_cm = confusion_matrix(y, oof_class, labels=[0, 1, 2]).tolist()

    # --- Final model: full train + valid monitor, predict frozen test ---
    t0 = time.time()
    set_seed(args.seed)
    full_masks = infer_temporal_masks(
        np.asarray(train["text_bert"]), np.asarray(train["audio"]), np.asarray(train["vision"])
    )
    full_scaler = FeatureScaler.fit(
        np.asarray(train["audio"]), np.asarray(train["vision"]), full_masks
    )
    ds_train = build_dataset(
        [str(x) for x in np.asarray(train["id"], dtype=object)],
        train, text_cache, full_scaler, labelled=True,
    )
    ds_valid = build_dataset(
        [str(x) for x in np.asarray(aligned["valid"]["id"], dtype=object)],
        aligned["valid"], valid_cache, full_scaler, labelled=True,
    )
    ds_test = build_dataset(
        [str(x) for x in np.asarray(aligned["test"]["id"], dtype=object)],
        aligned["test"], test_cache, full_scaler, labelled=True,
    )
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    loader_valid = DataLoader(ds_valid, batch_size=args.batch_size, shuffle=False, num_workers=0)
    loader_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = RobustFusionModel(ModelConfig(hidden_dim=args.hidden_dim)).to(device)
    criterion = FocalMultiTaskLoss(
        class_weights_tensor(y, device),
        focal_gamma=0.0,
        label_smoothing=args.label_smoothing,
        consistency_weight=0.10,
    )
    best_epoch, best_score = train_model(
        model, criterion, loader_train, loader_valid, device, args
    )
    test_pred = predict(model, loader_test, device)
    test_metrics = full_metrics(
        test_pred["true_class"], test_pred["predicted_class"],
        test_pred["true_regression"], test_pred["predicted_regression"],
    )
    test_roc = ovr_roc(test_pred["true_class"], test_pred["probabilities"])
    test_cm = confusion_matrix(
        test_pred["true_class"], test_pred["predicted_class"], labels=[0, 1, 2]
    ).tolist()
    final_elapsed = time.time() - t0

    # --- Persist ---
    summary = {
        "track": "A",
        "config": {
            "label_smoothing": args.label_smoothing,
            "focal_gamma": 0.0,
            "consistency_weight": 0.10,
            "hidden_dim": args.hidden_dim,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "patience": args.patience,
            "mask_probability": args.mask_probability,
        },
        "seed": args.seed,
        "device": str(device),
        "n_folds": args.n_folds,
        "fold_metrics": fold_rows,
        "oof_metrics": oof_metrics,
        "oof_roc": oof_roc,
        "oof_confusion_matrix": oof_cm,
        "test_metrics": test_metrics,
        "test_roc": test_roc,
        "test_confusion_matrix": test_cm,
        "final_best_epoch": best_epoch,
        "final_elapsed_seconds": final_elapsed,
    }
    write_json(OUT / "summary.json", summary)

    np.save(OUT / "oof_probs.npy", oof_probs)
    np.save(OUT / "oof_class.npy", oof_class)
    np.save(OUT / "oof_reg.npy", oof_reg)
    np.save(OUT / "oof_true_class.npy", y)
    np.save(OUT / "oof_true_reg.npy", yr)
    np.save(OUT / "test_probs.npy", test_pred["probabilities"])
    np.save(OUT / "test_class.npy", test_pred["predicted_class"])
    np.save(OUT / "test_reg.npy", test_pred["predicted_regression"])
    np.save(OUT / "test_true_class.npy", test_pred["true_class"])
    np.save(OUT / "test_true_reg.npy", test_pred["true_regression"])

    print(json.dumps({
        "oof_metrics": oof_metrics,
        "oof_macro_auc": oof_roc["macro_auc"],
        "test_metrics": test_metrics,
        "test_macro_auc": test_roc["macro_auc"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

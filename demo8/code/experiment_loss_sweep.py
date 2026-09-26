"""Loss-level experiments targeting the neutral class.

Evidence so far: neutral is confidently misclassified (73/85 confident-wrong),
so boundary tweaks fail.  This script sweeps loss-level levers, each a cheap
~100s CPU retrain, reusing the existing model and training loop.

Levers:
  --focal-gamma        0 = plain cross-entropy; >0 = focal loss (focus hard cases)
  --consistency-weight  weight of the classification/regression sign-consistency loss
  --neutral-scale      multiplier applied to the neutral class weight
  --label-smoothing    soft-label smoothing for the classification target
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from common import (
    ARTIFACTS_ROOT,
    CACHE_ROOT,
    RESULTS_ROOT,
    FeatureScaler,
    infer_temporal_masks,
    load_aligned,
    set_seed,
    write_json,
)
from sklearn.metrics import f1_score

from problem2_train import (
    AlignedDataset,
    build_dataset,
    metric_dict,
    model_kwargs,
    predict,
    to_device,
    train_epoch,
)
from robust_model import ModelConfig, RobustFusionModel


def per_class_f1(prediction: dict) -> dict[str, float]:
    true = prediction["true_class"]
    pred = prediction["predicted_class"]
    return {
        "neutral_f1": float(f1_score(true, pred, labels=[1], average=None)[0]),
        "negative_f1": float(f1_score(true, pred, labels=[0], average=None)[0]),
        "positive_f1": float(f1_score(true, pred, labels=[2], average=None)[0]),
    }


class FocalMultiTaskLoss(nn.Module):
    """Classification (optionally focal + label smoothing) + the same aux losses."""

    def __init__(
        self,
        class_weights: torch.Tensor,
        focal_gamma: float = 0.0,
        label_smoothing: float = 0.0,
        regression_weight: float = 0.60,
        correlation_weight: float = 0.20,
        consistency_weight: float = 0.10,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.regression_weight = regression_weight
        self.correlation_weight = correlation_weight
        self.consistency_weight = consistency_weight

    @staticmethod
    def correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction - prediction.mean()
        target = target - target.mean()
        denominator = torch.sqrt(
            torch.sum(prediction.square()) * torch.sum(target.square()) + 1e-8
        )
        return 1.0 - torch.sum(prediction * target) / denominator

    def forward(self, outputs, classification, regression):
        logits = outputs["logits"]
        ce = F.cross_entropy(
            logits, classification, weight=self.class_weights, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if self.focal_gamma > 0:
            pt = torch.exp(-ce)  # softmax probability of the target class
            classification_loss = ((1.0 - pt) ** self.focal_gamma * ce).mean()
        else:
            classification_loss = ce.mean()
        regression_loss = F.smooth_l1_loss(outputs["regression"], regression, beta=0.5)
        correlation_loss = self.correlation_loss(outputs["regression"], regression)
        probabilities = torch.softmax(logits, dim=-1)
        polarity_axis = torch.tensor([-1.0, 0.0, 1.0], device=probabilities.device, dtype=probabilities.dtype)
        expected_polarity = probabilities @ polarity_axis
        consistency_loss = F.mse_loss(expected_polarity, outputs["regression"] / 3.0)
        total = (
            classification_loss
            + self.regression_weight * regression_loss
            + self.correlation_weight * correlation_loss
            + self.consistency_weight * consistency_loss
        )
        parts = {
            "classification": float(classification_loss.detach()),
            "regression": float(regression_loss.detach()),
            "correlation": float(correlation_loss.detach()),
            "consistency": float(consistency_loss.detach()),
        }
        return total, parts


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device != "auto":
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_experiment(args: argparse.Namespace, tag: str) -> dict:
    set_seed(args.seed)
    device = resolve_device(args)
    cache_dir = CACHE_ROOT / "bert_features"

    aligned = load_aligned()
    train_masks = infer_temporal_masks(
        aligned["train"]["text_bert"], aligned["train"]["audio"], aligned["train"]["vision"]
    )
    scaler = FeatureScaler.fit(aligned["train"]["audio"], aligned["train"]["vision"], train_masks)

    datasets = {}
    for split in ("train", "valid", "test"):
        text_cache = np.load(cache_dir / f"{split}.npy", mmap_mode="r")
        datasets[split] = build_dataset(
            aligned[split]["id"], aligned[split], text_cache, scaler, labelled=True
        )
    loaders = {
        split: DataLoader(ds, batch_size=args.batch_size, shuffle=split == "train", num_workers=0)
        for split, ds in datasets.items()
    }

    model = RobustFusionModel(ModelConfig(hidden_dim=args.hidden_dim)).to(device)
    counts = np.bincount(
        np.asarray(aligned["train"]["classification_labels"], dtype=np.int64), minlength=3
    )
    class_weights = np.sqrt(len(datasets["train"]) / (3.0 * counts))
    class_weights = class_weights / class_weights.mean()
    class_weights[1] *= args.neutral_scale  # 1 = neutral
    criterion = FocalMultiTaskLoss(
        torch.tensor(class_weights, dtype=torch.float32, device=device),
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        consistency_weight=args.consistency_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
            model, loaders["train"], optimizer, criterion, device, amp_scaler,
            use_augmentation=args.variant == "robust", mask_probability=args.mask_probability,
        )
        if not math.isfinite(training_loss):
            raise RuntimeError(f"Non-finite training loss ({training_loss}) on {device.type}")
        validation = predict(model, loaders["valid"], device)
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
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)
    validation = predict(model, loaders["valid"], device)
    test = predict(model, loaders["test"], device)

    return {
        "tag": tag,
        "elapsed_seconds": time.time() - start,
        "best_epoch": best_epoch,
        "validation": validation["metrics"],
        "validation_per_class": per_class_f1(validation),
        "test": test["metrics"],
        "test_per_class": per_class_f1(test),
        "class_weights": class_weights.tolist(),
        "device": str(device),
        "seed": args.seed,
        "config": {
            "focal_gamma": args.focal_gamma,
            "consistency_weight": args.consistency_weight,
            "neutral_scale": args.neutral_scale,
            "label_smoothing": args.label_smoothing,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("robust", "no_aug"), default="robust")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-probability", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    parser.add_argument("--neutral-scale", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--tag", type=str, default="exp")
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    result = None
    for attempt in range(1, args.max_attempts + 1):
        attempt_args = copy.copy(args)
        attempt_args.seed = args.seed + (attempt - 1)
        try:
            result = run_experiment(attempt_args, args.tag)
            break
        except RuntimeError as exc:
            print(
                f"[attempt {attempt}/{args.max_attempts}] failed: {exc}; "
                f"retrying with seed={attempt_args.seed + 1}",
                flush=True,
            )
    if result is None:
        raise RuntimeError(f"{args.tag} failed after {args.max_attempts} attempts")

    out_dir = RESULTS_ROOT / "loss_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.tag}.json"
    write_json(path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Train and validate the missing-aware multimodal sentiment model.

Each run trains exactly one variant (robust augmentation or no augmentation),
selects its checkpoint on attachment 2 validation data only, evaluates the
untouched attachment 2 test split, and predicts the 30 aligned attachment 3
files by filename.  Attachment 3 is never used for model selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import (
    ARTIFACTS_ROOT,
    CACHE_ROOT,
    CLASS_NAMES_EN,
    CLASS_NAMES_ZH,
    FeatureScaler,
    RESULTS_ROOT,
    attachment3_aligned_root,
    infer_temporal_masks,
    load_aligned,
    load_special_collection,
    project_intensity_to_class,
    set_seed,
    write_json,
)
from robust_model import (
    ModelConfig,
    MultiTaskLoss,
    RobustFusionModel,
    augment_local_blocks,
)


class AlignedDataset(Dataset):
    def __init__(
        self,
        identifiers: Iterable[str],
        text: np.ndarray,
        audio: np.ndarray,
        vision: np.ndarray,
        masks: dict[str, np.ndarray],
        classification: np.ndarray | None = None,
        regression: np.ndarray | None = None,
    ) -> None:
        self.identifiers = np.asarray(list(map(str, identifiers)), dtype=object)
        self.text = text
        self.audio = audio
        self.vision = vision
        self.masks = masks
        self.classification = classification
        self.regression = regression

    def __len__(self) -> int:
        return len(self.identifiers)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "id": self.identifiers[index],
            # The BERT cache is a read-only memmap.  Copy one short sequence so
            # PyTorch never receives a non-writable NumPy view.
            "text": torch.as_tensor(np.array(self.text[index], copy=True), dtype=torch.float32),
            "audio": torch.as_tensor(self.audio[index], dtype=torch.float32),
            "vision": torch.as_tensor(self.vision[index], dtype=torch.float32),
            "timeline": torch.as_tensor(self.masks["timeline"][index], dtype=torch.bool),
            "text_observed": torch.as_tensor(
                self.masks["text_observed"][index], dtype=torch.bool
            ),
            "audio_observed": torch.as_tensor(
                self.masks["audio_observed"][index], dtype=torch.bool
            ),
            "vision_observed": torch.as_tensor(
                self.masks["vision_observed"][index], dtype=torch.bool
            ),
        }
        if self.classification is not None:
            item["classification"] = torch.as_tensor(
                self.classification[index], dtype=torch.long
            )
            item["regression"] = torch.as_tensor(
                self.regression[index], dtype=torch.float32
            )
        return item


def build_dataset(
    identifiers: Iterable[str],
    data: dict[str, Any],
    text_cache: np.ndarray,
    scaler: FeatureScaler,
    labelled: bool,
) -> AlignedDataset:
    text_bert = np.asarray(data["text_bert"])
    audio_raw = np.asarray(data["audio"])
    vision_raw = np.asarray(data["vision"])
    masks = infer_temporal_masks(text_bert, audio_raw, vision_raw)
    audio = scaler.transform_audio(audio_raw)
    vision = scaler.transform_vision(vision_raw)
    classification = (
        np.asarray(data["classification_labels"]).reshape(-1).astype(np.int64)
        if labelled
        else None
    )
    regression = (
        np.asarray(data["regression_labels"]).reshape(-1).astype(np.float32)
        if labelled
        else None
    )
    return AlignedDataset(
        identifiers=identifiers,
        text=text_cache,
        audio=audio,
        vision=vision,
        masks=masks,
        classification=classification,
        regression=regression,
    )


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def model_kwargs(batch: dict[str, Any], observed: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
    observed = observed or {
        "text": batch["text_observed"],
        "audio": batch["audio_observed"],
        "vision": batch["vision_observed"],
    }
    return {
        "text": batch["text"],
        "audio": batch["audio"],
        "vision": batch["vision"],
        "timeline": batch["timeline"],
        "text_observed": observed["text"],
        "audio_observed": observed["audio"],
        "vision_observed": observed["vision"],
    }


def metric_dict(
    true_class: np.ndarray,
    predicted_class: np.ndarray,
    true_regression: np.ndarray,
    predicted_regression: np.ndarray,
) -> dict[str, float]:
    if np.std(true_regression) < 1e-12 or np.std(predicted_regression) < 1e-12:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(true_regression, predicted_regression)[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(true_class, predicted_class)),
        "macro_f1": float(f1_score(true_class, predicted_class, average="macro")),
        "mae": float(mean_absolute_error(true_regression, predicted_regression)),
        "pearson": pearson,
    }
    metrics["selection_score"] = (
        0.50 * metrics["macro_f1"]
        + 0.30 * ((metrics["pearson"] + 1.0) / 2.0)
        + 0.20 * max(0.0, 1.0 - metrics["mae"] / 3.0)
    )
    return metrics


def overlay_missing_block(
    observed: dict[str, torch.Tensor],
    timeline: torch.Tensor,
    modalities: tuple[str, ...],
    fraction: float,
    location: str,
) -> dict[str, torch.Tensor]:
    result = {name: mask.clone() for name, mask in observed.items()}
    for batch_index in range(timeline.shape[0]):
        positions = torch.nonzero(timeline[batch_index], as_tuple=False).flatten()
        length = int(positions.numel())
        if length == 0:
            continue
        block_length = min(length, max(1, int(round(length * fraction))))
        if location == "early":
            start = 0
        elif location == "middle":
            start = max(0, (length - block_length) // 2)
        elif location == "late":
            start = length - block_length
        elif location == "all":
            start = 0
            block_length = length
        else:
            raise ValueError(location)
        selected = positions[start : start + block_length]
        for name in modalities:
            result[name][batch_index, selected] = False
    return result


@torch.inference_mode()
def predict(
    model: RobustFusionModel,
    loader: DataLoader,
    device: torch.device,
    scenario: tuple[tuple[str, ...], float, str] | None = None,
    gate_mode: str = "dynamic",
    include_attention: bool = False,
) -> dict[str, Any]:
    model.eval()
    storage: dict[str, list[Any]] = {
        "id": [],
        "probabilities": [],
        "predicted_class": [],
        "predicted_regression": [],
        "gates": [],
        "reliability": [],
    }
    if include_attention:
        storage.update({f"{name}_attention": [] for name in RobustFusionModel.MODALITIES})
    labelled = False
    true_class: list[np.ndarray] = []
    true_regression: list[np.ndarray] = []
    for batch in loader:
        batch = to_device(batch, device)
        observed = {
            "text": batch["text_observed"],
            "audio": batch["audio_observed"],
            "vision": batch["vision_observed"],
        }
        if scenario is not None:
            observed = overlay_missing_block(
                observed, batch["timeline"], scenario[0], scenario[1], scenario[2]
            )
        outputs = model(**model_kwargs(batch, observed), gate_mode=gate_mode)
        probabilities = torch.softmax(outputs["logits"], dim=-1)
        storage["id"].extend(batch["id"])
        storage["probabilities"].append(probabilities.cpu().numpy())
        storage["predicted_class"].append(probabilities.argmax(dim=-1).cpu().numpy())
        storage["predicted_regression"].append(outputs["regression"].cpu().numpy())
        storage["gates"].append(outputs["gates"].cpu().numpy())
        storage["reliability"].append(outputs["reliability"].cpu().numpy())
        if include_attention:
            for name in RobustFusionModel.MODALITIES:
                storage[f"{name}_attention"].append(outputs[f"{name}_attention"].cpu().numpy())
        if "classification" in batch:
            labelled = True
            true_class.append(batch["classification"].cpu().numpy())
            true_regression.append(batch["regression"].cpu().numpy())

    result: dict[str, Any] = {"id": np.asarray(storage["id"], dtype=object)}
    for key, chunks in storage.items():
        if key == "id":
            continue
        result[key] = np.concatenate(chunks, axis=0)
    result["predicted_regression_raw"] = result["predicted_regression"].copy()
    result["predicted_regression"] = project_intensity_to_class(
        result["predicted_regression_raw"], result["predicted_class"]
    )
    if labelled:
        result["true_class"] = np.concatenate(true_class)
        result["true_regression"] = np.concatenate(true_regression)
        result["metrics"] = metric_dict(
            result["true_class"],
            result["predicted_class"],
            result["true_regression"],
            result["predicted_regression"],
        )
    return result


def predictions_frame(prediction: dict[str, Any], include_truth: bool) -> pd.DataFrame:
    probabilities = prediction["probabilities"]
    gates = prediction["gates"]
    reliability = prediction["reliability"]
    classes = prediction["predicted_class"].astype(int)
    frame = pd.DataFrame(
        {
            "sample_id": prediction["id"],
            "predicted_class": classes,
            "polarity": [CLASS_NAMES_ZH[value] for value in classes],
            "polarity_en": [CLASS_NAMES_EN[value] for value in classes],
            "predicted_intensity": prediction["predicted_regression"],
            "raw_predicted_intensity": prediction["predicted_regression_raw"],
            "confidence": probabilities.max(axis=1),
            "prob_negative": probabilities[:, 0],
            "prob_neutral": probabilities[:, 1],
            "prob_positive": probabilities[:, 2],
            "weight_text": gates[:, 0],
            "weight_audio": gates[:, 1],
            "weight_vision": gates[:, 2],
            "observed_text": reliability[:, 0],
            "observed_audio": reliability[:, 1],
            "observed_vision": reliability[:, 2],
        }
    )
    if include_truth:
        frame["true_class"] = prediction["true_class"].astype(int)
        frame["true_polarity"] = [
            CLASS_NAMES_ZH[value] for value in prediction["true_class"].astype(int)
        ]
        frame["true_intensity"] = prediction["true_regression"]
        frame["absolute_error"] = np.abs(
            prediction["predicted_regression"] - prediction["true_regression"]
        )
    return frame


def train_epoch(
    model: RobustFusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MultiTaskLoss,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_augmentation: bool,
    mask_probability: float,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0
    use_amp = device.type == "cuda"
    for batch in loader:
        batch = to_device(batch, device)
        observed = {
            "text": batch["text_observed"],
            "audio": batch["audio_observed"],
            "vision": batch["vision_observed"],
        }
        if use_augmentation:
            observed = augment_local_blocks(
                observed, batch["timeline"], probability=mask_probability
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(**model_kwargs(batch, observed))
            loss, _ = criterion(
                outputs, batch["classification"], batch["regression"]
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        batch_size = len(batch["id"])
        total_loss += float(loss.detach()) * batch_size
        sample_count += batch_size
    return total_loss / sample_count


def robustness_grid(
    model: RobustFusionModel,
    loader: DataLoader,
    device: torch.device,
    variant: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    clean = predict(model, loader, device)
    rows.append(
        {
            "variant": variant,
            "missing_type": "none",
            "missing_rate": 0.0,
            "location": "none",
            **clean["metrics"],
        }
    )
    modality_sets = {
        "text": ("text",),
        "audio": ("audio",),
        "vision": ("vision",),
        "audio+vision": ("audio", "vision"),
    }
    for missing_type, modalities in modality_sets.items():
        for rate in (0.10, 0.20, 0.30, 0.40, 0.50):
            for location in ("early", "middle", "late"):
                result = predict(
                    model,
                    loader,
                    device,
                    scenario=(modalities, rate, location),
                )
                rows.append(
                    {
                        "variant": variant,
                        "missing_type": missing_type,
                        "missing_rate": rate,
                        "location": location,
                        **result["metrics"],
                    }
                )
    return pd.DataFrame(rows)


def ablation_table(
    model: RobustFusionModel,
    loader: DataLoader,
    device: torch.device,
    variant: str,
) -> pd.DataFrame:
    scenarios: dict[str, tuple[tuple[str, ...], float, str] | None] = {
        "完整动态门控": None,
        "仅文本": (("audio", "vision"), 1.0, "all"),
        "仅语音": (("text", "vision"), 1.0, "all"),
        "仅视觉": (("text", "audio"), 1.0, "all"),
        "去除文本": (("text",), 1.0, "all"),
        "去除语音": (("audio",), 1.0, "all"),
        "去除视觉": (("vision",), 1.0, "all"),
    }
    rows = []
    for name, scenario in scenarios.items():
        result = predict(model, loader, device, scenario=scenario)
        rows.append({"variant": variant, "setting": name, **result["metrics"]})
    uniform = predict(model, loader, device, gate_mode="uniform")
    rows.append({"variant": variant, "setting": "完整等权融合", **uniform["metrics"]})
    return pd.DataFrame(rows)


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
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = CACHE_ROOT / "bert_features"
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Run code/cache_bert_features.py first")

    aligned = load_aligned()
    train_masks = infer_temporal_masks(
        aligned["train"]["text_bert"],
        aligned["train"]["audio"],
        aligned["train"]["vision"],
    )
    feature_scaler = FeatureScaler.fit(
        aligned["train"]["audio"], aligned["train"]["vision"], train_masks
    )

    datasets: dict[str, AlignedDataset] = {}
    for split in ("train", "valid", "test"):
        text_cache = np.load(cache_dir / f"{split}.npy", mmap_mode="r")
        datasets[split] = build_dataset(
            aligned[split]["id"], aligned[split], text_cache, feature_scaler, labelled=True
        )
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }

    config = ModelConfig(hidden_dim=args.hidden_dim)
    model = RobustFusionModel(config).to(device)
    counts = np.bincount(
        np.asarray(aligned["train"]["classification_labels"], dtype=np.int64), minlength=3
    )
    class_weights = np.sqrt(len(datasets["train"]) / (3.0 * counts))
    class_weights = class_weights / class_weights.mean()
    criterion = MultiTaskLoss(torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=2e-6
    )
    amp_scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    artifact_dir = ARTIFACTS_ROOT / f"problem2_{args.variant}"
    result_dir = RESULTS_ROOT / f"problem2_{args.variant}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    feature_scaler.save(artifact_dir / "feature_scaler.json")

    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    wait = 0
    history = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        training_loss = train_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            amp_scaler,
            use_augmentation=args.variant == "robust",
            mask_probability=args.mask_probability,
        )
        validation = predict(model, loaders["valid"], device)
        metrics = validation["metrics"]
        scheduler.step(metrics["selection_score"])
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **metrics,
            }
        )
        print(
            f"epoch={epoch:02d} loss={training_loss:.4f} "
            f"f1={metrics['macro_f1']:.4f} mae={metrics['mae']:.4f} "
            f"r={metrics['pearson']:.4f} score={metrics['selection_score']:.4f}"
        )
        if metrics["selection_score"] > best_score + 1e-4:
            best_score = metrics["selection_score"]
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
    predictions_frame(validation, include_truth=True).to_csv(
        result_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig"
    )
    predictions_frame(test, include_truth=True).to_csv(
        result_dir / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(history).to_csv(result_dir / "training_history.csv", index=False)

    robustness = robustness_grid(model, loaders["valid"], device, args.variant)
    robustness.to_csv(result_dir / "missing_robustness.csv", index=False, encoding="utf-8-sig")
    ablations = ablation_table(model, loaders["valid"], device, args.variant)
    ablations.to_csv(result_dir / "ablation.csv", index=False, encoding="utf-8-sig")

    names3, data3 = load_special_collection(attachment3_aligned_root())
    text3 = np.load(cache_dir / "attachment3.npy", mmap_mode="r")
    dataset3 = build_dataset(names3, data3, text3, feature_scaler, labelled=False)
    loader3 = DataLoader(
        dataset3,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    prediction3 = predict(model, loader3, device)
    frame3 = predictions_frame(prediction3, include_truth=False)
    if frame3["sample_id"].duplicated().any() or set(frame3["sample_id"]) != set(names3):
        raise RuntimeError("Attachment 3 filename-to-prediction mapping is not one-to-one")
    frame3.to_csv(
        result_dir / "附件3预测结果.csv", index=False, encoding="utf-8-sig"
    )

    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": config.to_dict(),
        "variant": args.variant,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "class_weights": class_weights.tolist(),
        "bert_model": json.loads(metadata_path.read_text(encoding="utf-8"))["model_name"],
    }
    torch.save(checkpoint, artifact_dir / "fusion_head.pt")
    summary = {
        "variant": args.variant,
        "device": str(device),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "elapsed_seconds": time.time() - start_time,
        "train_samples": len(datasets["train"]),
        "validation_samples": len(datasets["valid"]),
        "test_samples": len(datasets["test"]),
        "attachment3_samples": len(dataset3),
        "validation": validation["metrics"],
        "test": test["metrics"],
        "model_config": config.to_dict(),
        "training_arguments": vars(args),
    }
    write_json(result_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

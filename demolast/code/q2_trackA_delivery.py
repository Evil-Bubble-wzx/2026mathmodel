"""Reproduce the selected Track A checkpoint and complete Q2 validation/delivery."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo1/code"))
sys.path.insert(0, str(ROOT / "code/Q2"))

from common import (CACHE_ROOT, FeatureScaler, attachment3_aligned_root,
                    infer_temporal_masks, load_aligned, load_special_collection,
                    set_seed, write_json)
from experiment_loss_sweep import FocalMultiTaskLoss
from problem2_train import build_dataset, metric_dict, predict, predictions_frame
from q2_trackA_cv_roc import class_weights_tensor, resolve_device, train_model
from robust_model import ModelConfig, RobustFusionModel


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(pred: dict) -> dict:
    result = dict(pred["metrics"])
    result["weighted_f1"] = float(f1_score(pred["true_class"], pred["predicted_class"], average="weighted"))
    result["macro_auc"] = float(roc_auc_score(pred["true_class"], pred["probabilities"], multi_class="ovr", average="macro"))
    result["confusion_matrix"] = confusion_matrix(pred["true_class"], pred["predicted_class"], labels=[0, 1, 2]).tolist()
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    seed = 2026
    set_seed(seed)
    device = resolve_device(args.device)
    print(f"device={device}", flush=True)
    data = load_aligned()
    train, valid, test = data["train"], data["valid"], data["test"]
    names3, attachment = load_special_collection(attachment3_aligned_root())
    cache = CACHE_ROOT / "bert_features"
    metadata = json.loads((cache / "metadata.json").read_text())
    if metadata["attachment3_files"] != names3:
        raise ValueError("Attachment3 BERT cache order differs from current filenames")
    text = {key: np.load(cache / f"{key}.npy", mmap_mode="r") for key in ("train", "valid", "test", "attachment3")}
    for name, block in (("train", train), ("valid", valid), ("test", test), ("attachment3", attachment)):
        expected = (len(block["audio"]), 50, 768)
        if text[name].shape != expected:
            raise ValueError(f"{name}: BERT cache {text[name].shape} != {expected}")
        for modality, shape in (("audio", (50, 74)), ("vision", (50, 35))):
            arr = np.asarray(block[modality])
            if arr.shape[1:] != shape or not np.isfinite(arr).all():
                raise ValueError(f"{name}: invalid {modality} shape or nonfinite value")

    masks = infer_temporal_masks(np.asarray(train["text_bert"]), np.asarray(train["audio"]), np.asarray(train["vision"]))
    scaler = FeatureScaler.fit(np.asarray(train["audio"]), np.asarray(train["vision"]), masks)
    identifiers = {name: [str(x) for x in np.asarray(block["id"], dtype=object)] for name, block in (("train", train), ("valid", valid), ("test", test))}
    datasets = {
        "train": build_dataset(identifiers["train"], train, text["train"], scaler, labelled=True),
        "valid": build_dataset(identifiers["valid"], valid, text["valid"], scaler, labelled=True),
        "test": build_dataset(identifiers["test"], test, text["test"], scaler, labelled=True),
        "attachment3": build_dataset(names3, attachment, text["attachment3"], scaler, labelled=False),
    }
    loaders = {name: DataLoader(ds, batch_size=64, shuffle=(name == "train"), num_workers=0) for name, ds in datasets.items()}
    model = RobustFusionModel(ModelConfig(hidden_dim=96)).to(device)
    y = np.asarray(train["classification_labels"]).reshape(-1).astype(np.int64)
    criterion = FocalMultiTaskLoss(class_weights_tensor(y, device), focal_gamma=0.0, label_smoothing=0.15, consistency_weight=0.10)
    train_args = argparse.Namespace(learning_rate=3e-4, weight_decay=1e-4, epochs=40, patience=8, mask_probability=0.45)
    best_epoch, best_score = train_model(model, criterion, loaders["train"], loaders["valid"], device, train_args)
    out = ROOT / "results/Q2/trackA_delivery"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "label_sm015_final.pt"
    torch.save({"state_dict": model.state_dict(), "config": asdict(ModelConfig(hidden_dim=96)), "seed": seed, "best_epoch": best_epoch, "best_score": best_score}, checkpoint)
    scaler_path = out / "train_scaler.json"
    scaler.save(scaler_path)

    full = {}
    for name in ("valid", "test", "attachment3"):
        p = predict(model, loaders[name], device)
        frame = predictions_frame(p, include_truth=(name != "attachment3"))
        frame.to_csv(out / f"{name}_predictions.csv", index=False, encoding="utf-8-sig")
        full[name] = evaluate(p) if name != "attachment3" else {
            "rows": len(frame), "class_counts": {str(k): int(v) for k, v in frame["predicted_class"].value_counts().sort_index().items()},
        }
        if name == "attachment3" and (len(frame) != 30 or frame["sample_id"].tolist() != names3):
            raise ValueError("Attachment3 row count or order failed")
        if name == "attachment3" and not np.allclose(frame[["prob_negative", "prob_neutral", "prob_positive"]].sum(axis=1), 1, atol=1e-5):
            raise ValueError("Attachment3 probabilities do not sum to 1")
        print(name, full[name], flush=True)

    grid = []
    for modality in ("text", "audio", "vision"):
        for location in ("early", "middle", "late"):
            for fraction in (0.1, 0.2, 0.3, 0.4, 0.5):
                p = predict(model, loaders["valid"], device, scenario=((modality,), fraction, location))
                grid.append({"modality": modality, "location": location, "fraction": fraction, **evaluate(p)})
    pd.DataFrame([{k: v for k, v in row.items() if k != "confusion_matrix"} for row in grid]).to_csv(out / "missing_grid.csv", index=False, encoding="utf-8-sig")
    write_json(out / "missing_grid.json", grid)

    ablations = []
    for name, modalities, gate_mode in [
        ("equal_gate", (), "uniform"),
        ("text_only", ("audio", "vision"), "dynamic"),
        ("audio_only", ("text", "vision"), "dynamic"),
        ("vision_only", ("text", "audio"), "dynamic"),
        ("without_text", ("text",), "dynamic"),
        ("without_audio", ("audio",), "dynamic"),
        ("without_vision", ("vision",), "dynamic"),
    ]:
        scenario = (modalities, 1.0, "all") if modalities else None
        p = predict(model, loaders["valid"], device, scenario=scenario, gate_mode=gate_mode)
        ablations.append({"name": name, **evaluate(p)})
    write_json(out / "ablations.json", ablations)
    pd.DataFrame([{k: v for k, v in row.items() if k != "confusion_matrix"} for row in ablations]).to_csv(out / "ablations.csv", index=False, encoding="utf-8-sig")
    summary = {
        "model": "Track A RobustFusionModel label_sm015", "seed": seed, "device": str(device),
        "best_epoch": best_epoch, "best_validation_score": best_score,
        "train_rows": len(train["audio"]), "valid_rows": len(valid["audio"]), "test_rows": len(test["audio"]),
        "attachment3_rows": len(names3), "validation": full["valid"], "test": full["test"],
        "attachment3": full["attachment3"], "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
        "scaler": str(scaler_path), "scaler_sha256": digest(scaler_path),
        "bert_cache_metadata": str(cache / "metadata.json"), "missing_grid_rows": len(grid),
        "ablation_rows": len(ablations), "test_not_used_for_checkpoint_selection": True,
        "historical_test_sweep_caveat": "The label smoothing value 0.15 was previously compared using test results; treat test metrics as exploratory, not untouched blind-test estimates.",
    }
    write_json(out / "run_summary.json", summary)
    print(out / "run_summary.json", flush=True)


if __name__ == "__main__":
    main()

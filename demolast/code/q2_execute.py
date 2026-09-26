#!/usr/bin/env python3
"""Documented Q2 commands. Never runs training merely on import or --help."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse

import run_q2_round1 as core

ROOT = core.ROOT
DEFAULT_RUN = ROOT / "results/Q2/experiments/manual_run01"
ATTACHMENT = ROOT / "E题数据/附件3-模态缺失特征样本/未对齐版本"
SOURCES = [Path(__file__).resolve(), Path(core.__file__).resolve()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def fresh_dir(path):
    Path(path).mkdir(parents=True, exist_ok=False)


def fingerprint(run):
    return {
        "bundle": digest(run / "models/q2_bundle.joblib"),
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in SOURCES},
        "sklearn": sklearn.__version__,
        "data_sha256": digest(core.DATA_PATH),
    }


def load_bundle(run):
    bundle = joblib.load(run / "models/q2_bundle.joblib")
    if bundle["sklearn_version"] != sklearn.__version__:
        raise ValueError("scikit-learn版本与训练时不同，请使用原mathModel环境。")
    return bundle


def inferred_lengths(x):
    """Last observed nonzero position, NOT a ground-truth missingness mask."""
    present = np.any(x != 0, axis=2)
    return np.max(np.where(present, np.arange(x.shape[1])[None, :] + 1, 0), axis=1)


def check_split(split, labeled=True):
    n = len(split["raw_text"])
    if n == 0:
        raise ValueError("Empty input")
    for key, width in (("audio", 74), ("vision", 35)):
        x = np.asarray(split[key])
        if x.shape != (n, 500, width) or not np.isfinite(x).all():
            raise ValueError(f"{key}: expected ({n},500,{width}) finite values, got {x.shape}")
        lengths = split.get(key + "_lengths")
        if lengths is not None:
            lengths = np.asarray(lengths)
            if lengths.shape != (n,) or not np.isfinite(lengths).all() or np.any(lengths != np.floor(lengths)) or np.any((lengths < 0) | (lengths > 500)):
                raise ValueError(f"Invalid {key}_lengths")
    if labeled:
        yc = np.asarray(split["classification_labels"]).reshape(-1)
        yr = np.asarray(split["regression_labels"]).reshape(-1)
        if yc.shape != (n,) or not np.isin(yc, core.LABELS).all():
            raise ValueError("Invalid classification labels")
        if yr.shape != (n,) or not np.isfinite(yr).all() or np.any(np.abs(yr) > 3):
            raise ValueError("Invalid regression labels")


def features(bundle, split, infer=False):
    xt = bundle["vectorizer"].transform(map(str, split["raw_text"]))
    av = []
    for key in ("audio", "vision"):
        x = np.asarray(split[key])
        lengths = inferred_lengths(x) if infer else np.asarray(split[key + "_lengths"])
        av.append(core.sequence_summary(x, lengths))
    return xt, *av


def predict(bundle, split, infer=False):
    xt, xa, xv = features(bundle, split, infer)
    matrix = sparse.hstack([xt, sparse.csr_matrix(bundle["scaler"].transform(np.hstack([xa, xv])))], format="csr")
    result = {"M0": (bundle["baseline_cls"].predict_proba(matrix), bundle["baseline_reg"].predict(matrix))}
    probs = np.stack([bundle[k + "_cls"].predict_proba(x) for k, x in zip(("text", "audio", "vision"), (xt, xa, xv))])
    values = np.stack([bundle[k + "_reg"].predict(x) for k, x in zip(("text", "audio", "vision"), (xt, xa, xv))])
    # Preserve the approved round-1 rule: text is always on, not learned gating.
    available = np.stack([np.ones(xt.shape[0]), xa[:, -1] > 0, xv[:, -1] > 0])
    weights = np.asarray(bundle["weights"])[:, None] * available
    weights /= weights.sum(axis=0, keepdims=True)
    result["M1"] = ((probs * weights[:, :, None]).sum(0), (values * weights).sum(0))
    for name, (p, y) in result.items():
        if not np.isfinite(p).all() or not np.isfinite(y).all() or not np.allclose(p.sum(1), 1) or np.any(p < 0):
            raise ValueError(f"Invalid predictions: {name}")
    return result


def altered(split, scenario):
    changed = dict(split)
    for modality in scenario.modalities:
        if modality == "text":
            changed["raw_text"] = core.mask_text(split["raw_text"], scenario.fraction, scenario.position)
        else:
            changed[modality] = core.mask_sequence(split[modality], split[modality + "_lengths"], scenario.fraction, scenario.position)
    return changed


def evaluate(bundle, split, output):
    """Same observations and masks, compare native lengths vs deployment proxy."""
    check_split(split)
    rows = []
    complete = {}
    for scenario in core.build_scenarios():
        changed = altered(split, scenario)
        for mode in ("provided_lengths", "inferred_lengths"):
            predictions = predict(bundle, changed, infer=(mode == "inferred_lengths"))
            for method, (p, y) in predictions.items():
                cls = core.classification_metrics(np.asarray(split["classification_labels"]).reshape(-1), p)
                reg = core.regression_metrics(np.asarray(split["regression_labels"]).reshape(-1), y)
                if scenario.scenario_id == "complete":
                    complete[(mode, method)] = cls["weighted_f1"]
                    frame = pd.DataFrame({"id": list(map(str, split["id"])), "true_class": np.asarray(split["classification_labels"]).reshape(-1), "true_intensity": np.asarray(split["regression_labels"]).reshape(-1), "predicted_class": np.argmax(p, axis=1), "intensity": np.clip(y, -3, 3)})
                    for i, name in enumerate(core.LABEL_NAMES):
                        frame["prob_" + name.lower()] = p[:, i]
                    frame.to_csv(output / f"{mode}_{method}_predictions.csv", index=False)
                rows.append({"mode": mode, "method": method, "scenario": scenario.scenario_id, **cls, **reg, "relative_weighted_f1_drop": core.relative_drop(complete[(mode, method)], cls["weighted_f1"])})
    pd.DataFrame(rows).to_csv(output / "scenario_metrics.csv", index=False)
    write_json(output / "scenario_metrics.json", rows)
    lengths_report = {}
    for key in ("audio", "vision"):
        original = np.asarray(split[key + "_lengths"])
        inferred = inferred_lengths(split[key])
        lengths_report[key] = {"exact_match_fraction": float(np.mean(original == inferred)), "mean_absolute_position_error": float(np.mean(np.abs(original - inferred)))}
    report = {"rows": len(split["raw_text"]), "metric_rows": len(rows), "length_comparison": lengths_report,
              "complete": [x for x in rows if x["scenario"] == "complete"],
              "text_middle_30": [x for x in rows if x["scenario"] == "text_middle_30"],
              "warnings": ["Inferred lengths can confuse trailing missingness with padding.", "Text remains enabled even for an empty or out-of-vocabulary string.", "No performance acceptance is implied by command success."]}
    write_json(output / "report.json", report)
    return report


def load_official(split):
    # Pickle loads the container in full; only the named split is used downstream.
    with core.DATA_PATH.open("rb") as f:
        data = pickle.load(f)
    return data[split]


def verify_locked(run):
    receipt = read_json(run / "locked.json")
    if receipt["fingerprint"] != fingerprint(run):
        raise ValueError("模型、代码、输入或环境已变化，请回到验证阶段；不能继续使用旧锁定记录。")
    if receipt["validation_report_sha256"] != digest(run / "deployment_validation/report.json"):
        raise ValueError("验证报告已变化。")
    return receipt


def check_environment():
    paths = {"project": ROOT.is_dir(), "features": core.DATA_PATH.is_file(), "attachment3_directory": ATTACHMENT.is_dir()}
    print(json.dumps({"python": sys.version.split()[0], "executable": sys.executable, "sklearn": sklearn.__version__, "platform": platform.platform(), "paths": paths, "attachment3_file_count": len(list(ATTACHMENT.glob("*.pkl")))}, ensure_ascii=False, indent=2))
    if not all(paths.values()):
        raise ValueError("Required paths missing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("check", help="Check imports and paths without loading data")
    for name in ("train", "validate", "lock", "test", "predict"):
        sub = subs.add_parser(name)
        sub.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
        if name == "lock":
            sub.add_argument("--note", required=True, help="Your judgment after reading validation results")
    args = parser.parse_args()
    if args.command == "check":
        check_environment()
        return
    run = args.run_dir.expanduser().resolve()
    if not run.is_relative_to(ROOT / "results/Q2/experiments"):
        raise ValueError("run-dir must be below results/Q2/experiments")
    if args.command == "train":
        fresh_dir(run)
        core.OUTPUT_ROOT = run
        core.TABLE_DIR, core.METRIC_DIR, core.FIGURE_DIR = [run / n for n in ("tables", "metrics", "figures")]
        train, valid = core.load_data()
        check_split(train)
        check_split(valid)
        # Avoid loading a second copy of the 2.9 GB container.
        core.load_data = lambda: (train, valid)
        core.main()
        write_json(run / "training_fingerprint.json", fingerprint(run))
    elif args.command == "validate":
        if read_json(run / "training_fingerprint.json") != fingerprint(run):
            raise ValueError("训练后代码、模型或数据已变化，使用新的run-dir重新训练。")
        output = run / "deployment_validation"
        fresh_dir(output)
        evaluate(load_bundle(run), load_official("valid"), output)
        write_json(output / "run_summary.json", {"question": "Q2", "status": "success", "used_split": "valid", "methods": ["M0", "M1"], "seed": core.SEED, "created_at": timestamp(), "training_fingerprint": read_json(run / "training_fingerprint.json"), "report": str(output / "report.json")})
    elif args.command == "lock":
        if not args.note.strip():
            raise ValueError("请填写你对验证结果的判断。")
        if (run / "locked.json").exists():
            raise FileExistsError("此运行已锁定。")
        validation = run / "deployment_validation/report.json"
        read_json(run / "deployment_validation/run_summary.json")
        report = read_json(validation)
        for row in report["text_middle_30"]:
            if row["method"] == "M1" and row["relative_weighted_f1_drop"] > 0.10:
                raise ValueError("M1文本中段缺失30%的F1下降超过约定10%红线，先回到验证阶段。")
        for row in report["complete"]:
            if row["method"] == "M1" and (row["unique_classes"] < 3 or row["top_class_mass"] >= 0.8 or row["prediction_std"] < 0.05 or row["unique_rounded_predictions"] < 20):
                raise ValueError("M1触发已有输出退化警报，先检查验证结果。")
        current = fingerprint(run)
        if current != read_json(run / "training_fingerprint.json"):
            raise ValueError("训练来源已变化。")
        write_json(run / "locked.json", {"locked_at": timestamp(), "operator_note": args.note, "fingerprint": current, "validation_report_sha256": digest(validation), "note": "Execution lock only; not paper-number freeze or audit approval."})
    elif args.command == "test":
        verify_locked(run)
        output = run / "final_test"
        fresh_dir(output)
        evaluate(load_bundle(run), load_official("test"), output)
        write_json(output / "run_summary.json", {"question": "Q2", "status": "success", "used_split": "test", "created_at": timestamp(), "methods": ["M0", "M1"], "seed": core.SEED, "locked_receipt": str(run / "locked.json")})
    elif args.command == "predict":
        verify_locked(run)
        read_json(run / "final_test/run_summary.json")
        output = run / "attachment3"
        fresh_dir(output)
        bundle = load_bundle(run)
        files = sorted(ATTACHMENT.glob("*.pkl"))
        if len(files) != 30:
            raise ValueError(f"Expected 30 samples, found {len(files)}")
        rows = []
        for path in files:
            with path.open("rb") as f:
                sample = pickle.load(f)
            text = np.asarray(sample["raw_text"], dtype=object).reshape(-1)
            if len(text) != 1:
                raise ValueError(f"Expected one text sample: {path.name}")
            split = {"raw_text": text}
            for key, width in (("audio", 74), ("vision", 35)):
                x = np.asarray(sample[key])
                if x.shape == (500, width):
                    x = x[None, :, :]
                split[key] = x
            check_split(split, labeled=False)
            p, y = predict(bundle, split, infer=True)["M1"]
            code = int(np.argmax(p[0]))
            rows.append({"sample_file": path.name, "class_id": code, "class_name": core.LABEL_NAMES[code], "intensity": float(np.clip(y[0], -3, 3)), **{f"prob_{name.lower()}": float(p[0, i]) for i, name in enumerate(core.LABEL_NAMES)}, "audio_inferred_length": int(inferred_lengths(split["audio"])[0]), "vision_inferred_length": int(inferred_lengths(split["vision"])[0]), "text_has_vocabulary_features": bool(bundle["vectorizer"].transform(map(str, text)).nnz)})
        pd.DataFrame(rows).to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
        write_json(output / "run_summary.json", {"question": "Q2", "status": "success", "created_at": timestamp(), "method": "M1", "seed": core.SEED, "rows": len(rows), "input_sha256": {p.name: digest(p) for p in files}, "model_sha256": digest(run / "models/q2_bundle.joblib"), "metrics": None, "warnings": ["Unlabeled samples: accuracy cannot be calculated.", "Lengths inferred from last nonzero position; not ground truth.", "Review availability fields before interpreting predictions."]})
    print(f"完成 {args.command}: {run}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)

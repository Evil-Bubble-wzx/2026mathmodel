#!/usr/bin/env python3
"""Run the approved Q2 round-1 main method (M1) and baseline (M0).

The pickle container includes all splits; only train and valid participate in
fitting/evaluation. No Attachment 3 file is loaded here.
"""

from __future__ import annotations

import json
import joblib
import pickle
import platform
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "E题数据" / "附件2-数据集特征文件" / "unaligned_50.pkl"
OUTPUT_ROOT = ROOT / "results" / "Q2" / "experiments" / "round1"
TABLE_DIR = OUTPUT_ROOT / "tables"
METRIC_DIR = OUTPUT_ROOT / "metrics"
FIGURE_DIR = OUTPUT_ROOT / "figures"
SEED = 20260924
LABELS = [0, 1, 2]
LABEL_NAMES = ["Negative", "Neutral", "Positive"]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    modalities: tuple[str, ...]
    position: str
    fraction: float


def build_scenarios() -> list[Scenario]:
    scenarios = [Scenario("complete", tuple(), "none", 0.0)]
    for modality in ("text", "audio", "vision"):
        for position in ("head", "middle", "tail"):
            for fraction in (0.10, 0.30, 0.50):
                scenarios.append(
                    Scenario(f"{modality}_{position}_{int(fraction * 100)}", (modality,), position, fraction)
                )
    for modalities in (("text", "audio"), ("text", "vision"), ("audio", "vision")):
        scenarios.append(Scenario("_".join(modalities) + "_middle_30", modalities, "middle", 0.30))
    return scenarios


def load_data() -> tuple[dict, dict]:
    with DATA_PATH.open("rb") as handle:
        data = pickle.load(handle)
    return data["train"], data["valid"]


def sequence_summary(x: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    rows = []
    for arr, raw_length in zip(x, lengths):
        length = max(0, min(int(raw_length), arr.shape[0]))
        if length == 0:
            rows.append(np.zeros(arr.shape[1] * 8 + 3, dtype=np.float32))
            continue
        valid = np.asarray(arr[:length], dtype=np.float32)
        cuts = np.linspace(0, length, 5, dtype=int)
        segments = [valid[left:max(left + 1, right)].mean(0) for left, right in zip(cuts[:-1], cuts[1:])]
        row_nonzero = np.any(valid != 0, axis=1)
        quality = np.asarray(
            [length / arr.shape[0], row_nonzero.mean(), float(row_nonzero.any())], dtype=np.float32
        )
        rows.append(np.concatenate([valid.mean(0), valid.std(0), valid.max(0), valid.min(0), *segments, quality]))
    return np.asarray(rows, dtype=np.float32)


def block_bounds(length: int, fraction: float, position: str) -> tuple[int, int]:
    width = max(1, int(round(length * fraction)))
    if position == "head":
        left = 0
    elif position == "middle":
        left = max(0, (length - width) // 2)
    elif position == "tail":
        left = max(0, length - width)
    else:
        raise ValueError(f"Unsupported position: {position}")
    return left, min(length, left + width)


def mask_text(texts: np.ndarray, fraction: float, position: str) -> np.ndarray:
    output = []
    for raw in map(str, texts):
        words = raw.split()
        if not words:
            output.append(raw)
            continue
        left, right = block_bounds(len(words), fraction, position)
        output.append(" ".join(words[:left] + words[right:]))
    return np.asarray(output)


def mask_sequence(x: np.ndarray, lengths: np.ndarray, fraction: float, position: str) -> np.ndarray:
    masked = x.copy()
    for i, raw_length in enumerate(lengths):
        length = max(1, min(int(raw_length), x.shape[1]))
        left, right = block_bounds(length, fraction, position)
        masked[i, left:right] = 0
    return masked


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = np.argmax(probabilities, axis=1)
    counts = np.bincount(predictions, minlength=3)
    safe = np.clip(probabilities, 1e-12, 1.0)
    recalls = recall_score(y_true, predictions, labels=LABELS, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted")),
        "negative_recall": float(recalls[0]),
        "neutral_recall": float(recalls[1]),
        "positive_recall": float(recalls[2]),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=LABELS).tolist(),
        "unique_classes": int(np.unique(predictions).size),
        "top_class_mass": float(counts.max() / counts.sum()),
        "mean_probability_entropy": float(np.mean(-np.sum(safe * np.log(safe), axis=1))),
    }


def regression_metrics(y_true: np.ndarray, raw_predictions: np.ndarray) -> dict:
    predictions = np.clip(raw_predictions, -3, 3)
    corr = (float(pearsonr(y_true, predictions).statistic)
            if len(predictions) > 1 and np.std(predictions) > 0 and np.std(y_true) > 0 else None)
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "pearson": corr,
        "prediction_std": float(np.std(predictions)),
        "unique_rounded_predictions": int(np.unique(np.round(predictions, 3)).size),
        "boundary_mass": float(np.mean((predictions <= -2.999) | (predictions >= 2.999))),
    }


def relative_drop(reference: float, value: float) -> float:
    return float((reference - value) / max(abs(reference), 1e-12))


def fit_extra_trees(x_train: np.ndarray, yc_train: np.ndarray, yr_train: np.ndarray, seed: int):
    classifier = ExtraTreesClassifier(
        n_estimators=180,
        min_samples_leaf=4,
        class_weight="balanced",
        n_jobs=1,
        random_state=seed,
    ).fit(x_train, yc_train)
    regressor = ExtraTreesRegressor(
        n_estimators=180,
        min_samples_leaf=4,
        n_jobs=1,
        random_state=seed,
    ).fit(x_train, yr_train)
    return classifier, regressor


def save_confusion_figure(complete_results: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, method_id in zip(axes, ("M0", "M1")):
        matrix = np.asarray(complete_results[method_id]["classification"]["confusion_matrix"])
        image = ax.imshow(matrix, cmap="Blues")
        for row in range(3):
            for col in range(3):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
        ax.set_title(f"{method_id} complete input")
        ax.set_xticks(range(3), LABEL_NAMES, rotation=25)
        ax.set_yticks(range(3), LABEL_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(image, ax=ax, fraction=0.046)
    path = FIGURE_DIR / "complete_confusion_matrices.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_missingness_figure(rows: list[dict]) -> Path:
    frame = pd.DataFrame(rows)
    frame = frame[frame["scenario_id"] != "complete"].copy()
    frame["relative_drop_percent"] = frame["relative_weighted_f1_drop"] * 100
    frame["label"] = frame["scenario_id"].str.replace("_", " ")
    frame = frame.sort_values(["method_id", "relative_drop_percent"], ascending=[True, False])
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharex=True, constrained_layout=True)
    for ax, method_id in zip(axes, ("M0", "M1")):
        part = frame[frame["method_id"] == method_id]
        colors = np.where(part["relative_drop_percent"] > 10, "#c0392b", "#2874a6")
        ax.barh(part["label"], part["relative_drop_percent"], color=colors)
        ax.axvline(10, color="#c0392b", linestyle="--", linewidth=1, label="10% threshold")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{method_id}: weighted-F1 relative drop")
        ax.set_xlabel("Relative drop (%)")
        ax.grid(axis="x", alpha=0.25)
    axes[1].legend(loc="lower right")
    path = FIGURE_DIR / "missingness_weighted_f1.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    overall_started = time.perf_counter()
    np.random.seed(SEED)
    for directory in (TABLE_DIR, METRIC_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    train, valid = load_data()
    yc_train = np.asarray(train["classification_labels"], dtype=int)
    yr_train = np.asarray(train["regression_labels"], dtype=float)
    yc_valid = np.asarray(valid["classification_labels"], dtype=int)
    yr_valid = np.asarray(valid["regression_labels"], dtype=float)
    audio_train_lengths = np.asarray(train["audio_lengths"], dtype=int)
    audio_valid_lengths = np.asarray(valid["audio_lengths"], dtype=int)
    vision_train_lengths = np.asarray(train["vision_lengths"], dtype=int)
    vision_valid_lengths = np.asarray(valid["vision_lengths"], dtype=int)

    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, max_features=7000, sublinear_tf=True
    )
    xt_train = vectorizer.fit_transform(map(str, train["raw_text"]))
    xt_valid_complete = vectorizer.transform(map(str, valid["raw_text"]))
    xa_train = sequence_summary(train["audio"], audio_train_lengths)
    xa_valid_complete = sequence_summary(valid["audio"], audio_valid_lengths)
    xv_train = sequence_summary(train["vision"], vision_train_lengths)
    xv_valid_complete = sequence_summary(valid["vision"], vision_valid_lengths)

    baseline_started = time.perf_counter()
    scaler = StandardScaler().fit(np.hstack([xa_train, xv_train]))
    av_train = sparse.csr_matrix(scaler.transform(np.hstack([xa_train, xv_train])))
    early_train = sparse.hstack([xt_train, av_train], format="csr")
    baseline_cls = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=SEED)
    ).fit(early_train, yc_train)
    baseline_reg = Ridge(alpha=12.0).fit(early_train, yr_train)
    baseline_fit_seconds = time.perf_counter() - baseline_started

    main_started = time.perf_counter()
    text_cls = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=SEED)
    ).fit(xt_train, yc_train)
    text_reg = Ridge(alpha=8.0).fit(xt_train, yr_train)
    audio_cls, audio_reg = fit_extra_trees(xa_train, yc_train, yr_train, SEED + 1)
    vision_cls, vision_reg = fit_extra_trees(xv_train, yc_train, yr_train, SEED + 2)
    main_fit_seconds = time.perf_counter() - main_started

    # Store estimators, not local functions, so a new process can load the bundle.
    model_path = OUTPUT_ROOT / "models" / "q2_bundle.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "schema_version": 1, "seed": SEED, "sklearn_version": sklearn.__version__,
        "vectorizer": vectorizer, "scaler": scaler,
        "baseline_cls": baseline_cls, "baseline_reg": baseline_reg,
        "text_cls": text_cls, "text_reg": text_reg,
        "audio_cls": audio_cls, "audio_reg": audio_reg,
        "vision_cls": vision_cls, "vision_reg": vision_reg,
        "weights": [0.50, 0.25, 0.25],
        "length_rule_without_metadata": "last_nonzero_row_plus_one; all_zero=0",
        "text_availability_rule": "always_on_in_approved_M1",
    }, model_path, compress=3)

    def baseline_predict(text_features, audio_features, vision_features):
        av = sparse.csr_matrix(scaler.transform(np.hstack([audio_features, vision_features])))
        matrix = sparse.hstack([text_features, av], format="csr")
        return baseline_cls.predict_proba(matrix), baseline_reg.predict(matrix)

    def main_predict(text_features, audio_features, vision_features):
        probabilities = np.stack(
            [
                text_cls.predict_proba(text_features),
                audio_cls.predict_proba(audio_features),
                vision_cls.predict_proba(vision_features),
            ]
        )
        regression = np.stack(
            [
                text_reg.predict(text_features),
                audio_reg.predict(audio_features),
                vision_reg.predict(vision_features),
            ]
        )
        available = np.ones((3, text_features.shape[0]), dtype=float)
        available[1] = audio_features[:, -1] > 0
        available[2] = vision_features[:, -1] > 0
        weights = np.asarray([0.50, 0.25, 0.25])[:, None] * available
        weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)
        return np.sum(probabilities * weights[:, :, None], axis=0), np.sum(regression * weights, axis=0)

    scenarios = build_scenarios()
    detailed_results: dict[str, dict] = {"M0": {}, "M1": {}}
    flat_rows: list[dict] = []
    complete_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for scenario in scenarios:
        text_features = xt_valid_complete
        audio_features = xa_valid_complete
        vision_features = xv_valid_complete
        if "text" in scenario.modalities:
            text_features = vectorizer.transform(
                mask_text(valid["raw_text"], scenario.fraction, scenario.position)
            )
        if "audio" in scenario.modalities:
            audio_masked = mask_sequence(
                valid["audio"], audio_valid_lengths, scenario.fraction, scenario.position
            )
            audio_features = sequence_summary(audio_masked, audio_valid_lengths)
        if "vision" in scenario.modalities:
            vision_masked = mask_sequence(
                valid["vision"], vision_valid_lengths, scenario.fraction, scenario.position
            )
            vision_features = sequence_summary(vision_masked, vision_valid_lengths)

        predictions = {
            "M0": baseline_predict(text_features, audio_features, vision_features),
            "M1": main_predict(text_features, audio_features, vision_features),
        }
        for method_id, (probabilities, regression) in predictions.items():
            classification = classification_metrics(yc_valid, probabilities)
            regression_result = regression_metrics(yr_valid, regression)
            detailed_results[method_id][scenario.scenario_id] = {
                "modalities": list(scenario.modalities),
                "position": scenario.position,
                "fraction": scenario.fraction,
                "classification": classification,
                "regression": regression_result,
            }
            if scenario.scenario_id == "complete":
                complete_predictions[method_id] = (probabilities, np.clip(regression, -3, 3))

    for method_id, method_results in detailed_results.items():
        complete = method_results["complete"]
        for scenario_id, result in method_results.items():
            cls = result["classification"]
            reg = result["regression"]
            row = {
                "method_id": method_id,
                "scenario_id": scenario_id,
                "modalities": "+".join(result["modalities"]) or "none",
                "position": result["position"],
                "fraction": result["fraction"],
                **{key: value for key, value in cls.items() if key != "confusion_matrix"},
                **reg,
                "confusion_matrix": json.dumps(cls["confusion_matrix"]),
                "absolute_weighted_f1_change": cls["weighted_f1"] - complete["classification"]["weighted_f1"],
                "relative_weighted_f1_drop": relative_drop(
                    complete["classification"]["weighted_f1"], cls["weighted_f1"]
                ),
                "relative_mae_increase": float(
                    (reg["mae"] - complete["regression"]["mae"])
                    / max(complete["regression"]["mae"], 1e-12)
                ),
            }
            flat_rows.append(row)
            result["absolute_weighted_f1_change"] = row["absolute_weighted_f1_change"]
            result["relative_weighted_f1_drop"] = row["relative_weighted_f1_drop"]
            result["relative_mae_increase"] = row["relative_mae_increase"]

    metrics_table = pd.DataFrame(flat_rows)
    metrics_path = TABLE_DIR / "scenario_metrics.csv"
    metrics_table.to_csv(metrics_path, index=False)

    prediction_frame = pd.DataFrame(
        {
            "id": [str(x) for x in valid["id"]],
            "true_class": yc_valid,
            "true_regression": yr_valid,
        }
    )
    for method_id, (probabilities, regression) in complete_predictions.items():
        prediction_frame[f"{method_id}_predicted_class"] = np.argmax(probabilities, axis=1)
        for index, name in enumerate(LABEL_NAMES):
            prediction_frame[f"{method_id}_prob_{name.lower()}"] = probabilities[:, index]
        prediction_frame[f"{method_id}_predicted_regression"] = regression
    prediction_path = TABLE_DIR / "complete_validation_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)

    metrics_json_path = METRIC_DIR / "scenario_metrics.json"
    metrics_json_path.write_text(json.dumps(detailed_results, ensure_ascii=False, indent=2), encoding="utf-8")

    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "seed": SEED,
        "extra_trees_n_jobs": 1,
    }
    environment_path = METRIC_DIR / "environment.json"
    environment_path.write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")

    confusion_path = save_confusion_figure(
        {method_id: detailed_results[method_id]["complete"] for method_id in ("M0", "M1")}
    )
    missingness_path = save_missingness_figure(flat_rows)

    complete_m0 = detailed_results["M0"]["complete"]
    complete_m1 = detailed_results["M1"]["complete"]
    text_middle_30 = detailed_results["M1"]["text_middle_30"]
    m1_trigger_checks = {
        "text_middle_30_relative_weighted_f1_drop_gt_10pct": text_middle_30["relative_weighted_f1_drop"] > 0.10,
        "fewer_than_three_predicted_classes": complete_m1["classification"]["unique_classes"] < 3,
        "top_class_mass_ge_80pct": complete_m1["classification"]["top_class_mass"] >= 0.80,
        "regression_std_lt_0_05": complete_m1["regression"]["prediction_std"] < 0.05,
        "unique_rounded_regression_lt_20": complete_m1["regression"]["unique_rounded_predictions"] < 20,
        "attachment3_length_rule": "not_evaluated_in_round1",
        "neutral_recall": complete_m1["classification"]["neutral_recall"],
    }
    observed_trigger = any(value is True for value in m1_trigger_checks.values())
    masking_improvement_scenarios = metrics_table[
        (metrics_table["method_id"] == "M1")
        & (metrics_table["scenario_id"] != "complete")
        & (metrics_table["relative_weighted_f1_drop"] < 0)
    ]["scenario_id"].tolist()

    overall_seconds = time.perf_counter() - overall_started
    run_summary = {
        "schema_version": 1,
        "question": "Q2",
        "round": OUTPUT_ROOT.name,
        "implementation_target": "python",
        "random_seed": SEED,
        "approved_decision_id": "q2_method_choice",
        "data_policy": {
            "input_files": [str(DATA_PATH.relative_to(ROOT))],
            "used_splits": ["train", "valid"],
            "container_includes_test": True,
            "test_used_for_fit_or_evaluation": False,
            "attachment3_opened": False,
        },
        "methods": [
            {
                "method_id": "M0",
                "role": "usable_baseline",
                "script": "code/Q2/run_q2_round1.py",
                "status": "success",
                "execution_time_seconds": baseline_fit_seconds,
                "input_files": [str(DATA_PATH.relative_to(ROOT))],
                "output_files": [str(metrics_path.relative_to(ROOT)), str(prediction_path.relative_to(ROOT))],
                "figure_files": [str(confusion_path.relative_to(ROOT)), str(missingness_path.relative_to(ROOT))],
                "metrics_summary": complete_m0,
                "warnings": ["Local audio/vision masking can strongly change standardized early-fusion summaries."],
                "errors": [],
            },
            {
                "method_id": "M1",
                "role": "main_candidate",
                "script": "code/Q2/run_q2_round1.py",
                "status": "success",
                "execution_time_seconds": main_fit_seconds,
                "input_files": [str(DATA_PATH.relative_to(ROOT))],
                "output_files": [str(metrics_path.relative_to(ROOT)), str(prediction_path.relative_to(ROOT))],
                "figure_files": [str(confusion_path.relative_to(ROOT)), str(missingness_path.relative_to(ROOT))],
                "metrics_summary": complete_m1,
                "warnings": [
                    "Fixed fusion weights are a reliability proxy, not a learned gate.",
                    "Neutral recall requires human judgment.",
                    "Negative relative drops under masking are instability/noise signals, not beneficial-missingness evidence.",
                ],
                "errors": [],
            },
        ],
        "comparison": {
            "complete_weighted_f1_difference_M1_minus_M0": complete_m1["classification"]["weighted_f1"] - complete_m0["classification"]["weighted_f1"],
            "complete_macro_f1_difference_M1_minus_M0": complete_m1["classification"]["macro_f1"] - complete_m0["classification"]["macro_f1"],
            "complete_mae_difference_M1_minus_M0": complete_m1["regression"]["mae"] - complete_m0["regression"]["mae"],
            "complete_pearson_difference_M1_minus_M0": (
                complete_m1["regression"]["pearson"] - complete_m0["regression"]["pearson"]
                if all(x["regression"]["pearson"] is not None for x in (complete_m0, complete_m1)) else None),
            "scenario_count_per_method": len(scenarios),
        },
        "fallback_trigger": {
            "fallback_id": "M2",
            "condition": "Recorded M1 interface, class-collapse, text-drop, or output-degeneracy trigger.",
            "observed": observed_trigger,
            "evidence": m1_trigger_checks,
        },
        "masking_improvement_scenarios_M1": masking_improvement_scenarios,
        "environment": environment,
        "total_execution_time_seconds": overall_seconds,
        "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2),
        "model_bundle": str(model_path),
        "warnings": ["This run does not establish repeated-run reproducibility."],
        "errors": [],
    }
    summary_path = OUTPUT_ROOT / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_summary": str(summary_path), "runtime_seconds": overall_seconds}, ensure_ascii=False))


if __name__ == "__main__":
    if OUTPUT_ROOT.exists():
        raise SystemExit("Output exists. Use q2_execute.py train with a new --run-dir; existing results are preserved.")
    main()

#!/usr/bin/env python3
"""Bounded Neutral-score adjustment; reuse trained M0/M1, never touch test to tune."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import q2_execute as cli
import run_q2_round1 as core

BASE_PREDICT = cli.predict
FACTORS = (1.0, 1.025, 1.05, 1.075, 1.10, 1.125, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 1.75, 2.0)


def scale_neutral(probabilities, factor):
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Neutral factor must be positive and finite")
    result = np.array(probabilities, dtype=float, copy=True)
    result[:, 1] *= factor
    result /= result.sum(axis=1, keepdims=True)
    return result


def adjusted_predict(bundle, split, infer=False):
    result = BASE_PREDICT(bundle, split, infer=infer)
    p, y = result["M1"]
    result["M1"] = (scale_neutral(p, bundle.get("neutral_factor", 1.0)), y)
    return result


def install_adjustment():
    cli.predict = adjusted_predict
    source = Path(__file__).resolve()
    if source not in cli.SOURCES:
        cli.SOURCES.append(source)


def select_factor(y, probabilities):
    baseline = core.classification_metrics(y, probabilities)
    rows = []
    eligible = []
    for factor in FACTORS:
        metrics = core.classification_metrics(y, scale_neutral(probabilities, factor))
        allowed = (factor > 1 and metrics["neutral_recall"] > baseline["neutral_recall"]
                   and metrics["weighted_f1"] >= baseline["weighted_f1"]
                   and metrics["macro_f1"] >= baseline["macro_f1"])
        row = {"factor": factor, "eligible_on_tuning": allowed, **metrics}
        rows.append(row)
        if allowed:
            eligible.append(row)
    selected = max(eligible, key=lambda r: (r["weighted_f1"], r["macro_f1"], -r["factor"]))["factor"] if eligible else 1.0
    return selected, rows


def adjust(source, output):
    started = time.perf_counter()
    source, output = source.resolve(), output.resolve()
    if not output.is_relative_to(cli.ROOT / "results/Q2/experiments"):
        raise ValueError("Output must be under results/Q2/experiments")
    source_fingerprint = cli.read_json(source / "training_fingerprint.json")
    if cli.fingerprint(source) != source_fingerprint:
        raise ValueError("Parent model/code/data no longer match the parent run")
    cli.fresh_dir(output)
    for name in ("metrics", "tables", "figures", "models", "deployment_validation"):
        (output / name).mkdir()
    bundle = cli.load_bundle(source)
    split = cli.load_official("valid")
    cli.check_split(split)
    y = np.asarray(split["classification_labels"]).reshape(-1).astype(int)
    ids = np.asarray(list(map(str, split["id"])))
    groups = np.array([x.split("$_$")[0] for x in ids])
    tuner, confirmation = next(GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=core.SEED).split(ids, y, groups))
    if set(groups[tuner]) & set(groups[confirmation]):
        raise AssertionError("Group overlap")
    if any(len(np.unique(y[index])) != 3 for index in (tuner, confirmation)):
        raise ValueError("Both validation portions must contain all three classes")

    # Verify cached evidence before using it. No parameter is selected on confirmation rows.
    original = {}
    modality_rows = []
    for mode in ("provided_lengths", "inferred_lengths"):
        infer = mode == "inferred_lengths"
        original[mode] = BASE_PREDICT(bundle, split, infer=infer)
        cached = pd.read_csv(source / "deployment_validation" / f"{mode}_M1_predictions.csv")
        if not np.array_equal(cached["id"].astype(str).to_numpy(), ids):
            raise ValueError("Parent IDs do not match current valid")
        np.testing.assert_array_equal(cached["true_class"].to_numpy(), y)
        np.testing.assert_allclose(original[mode]["M1"][0], cached[["prob_negative", "prob_neutral", "prob_positive"]].to_numpy(), atol=1e-12, rtol=0)
        xt, xa, xv = cli.features(bundle, split, infer=infer)
        for modality, x in zip(("text", "audio", "vision"), (xt, xa, xv)):
            p = bundle[modality + "_cls"].predict_proba(x)
            modality_rows.append({"mode": mode, "modality": modality, **core.classification_metrics(y, p), "mean_scores_on_true_neutral": p[y == 1].mean(axis=0).tolist()})
    factor, search_rows = select_factor(y[tuner], original["inferred_lengths"]["M1"][0][tuner])
    cli.write_json(output / "metrics/tuning_candidates.json", search_rows)
    pd.DataFrame(search_rows).to_csv(output / "tables/tuning_candidates.csv", index=False)
    cli.write_json(output / "metrics/modality_diagnostics.json", modality_rows)
    membership = np.full(len(y), "confirmation", dtype=object)
    membership[tuner] = "tuning"
    pd.DataFrame({"id": ids, "video_group": groups, "partition": membership, "true_class": y}).to_csv(output / "tables/validation_membership.csv", index=False)

    comparisons = []
    for mode, predictions in original.items():
        for partition, index in (("tuning", tuner), ("confirmation", confirmation), ("full_valid", np.arange(len(y)))):
            for variant, p in (("before", predictions["M1"][0]), ("adjusted", scale_neutral(predictions["M1"][0], factor))):
                comparisons.append({"mode": mode, "partition": partition, "variant": variant, "n": len(index), **core.classification_metrics(y[index], p[index])})
    cli.write_json(output / "metrics/partition_comparison.json", comparisons)
    pd.DataFrame(comparisons).to_csv(output / "tables/partition_comparison.csv", index=False)

    bundle["neutral_factor"] = factor
    bundle["parent_bundle_sha256"] = source_fingerprint["bundle"]
    bundle["entrypoint"] = "code/Q2/q2_adjust_neutral.py"
    joblib.dump(bundle, output / "models/q2_bundle.joblib", compress=3)
    restored = cli.load_bundle(output)
    for mode in original:
        p, regression = adjusted_predict(restored, split, infer=(mode == "inferred_lengths"))["M1"]
        np.testing.assert_allclose(p, scale_neutral(original[mode]["M1"][0], factor), atol=1e-12, rtol=0)
        np.testing.assert_array_equal(regression, original[mode]["M1"][1])

    install_adjustment()
    deployment = output / "deployment_validation"
    report = cli.evaluate(restored, split, deployment)
    rows = cli.read_json(deployment / "scenario_metrics.json")
    old_rows = {(r["mode"], r["method"], r["scenario"]): r for r in cli.read_json(source / "deployment_validation/scenario_metrics.json")}
    for row in rows:
        old = old_rows[(row["mode"], row["method"], row["scenario"])]
        for key in ("mae", "pearson", "prediction_std", "unique_rounded_predictions", "boundary_mass"):
            if row[key] != old[key]:
                raise AssertionError(f"Regression changed: {key}")
        if row["method"] == "M0":
            for key in ("weighted_f1", "macro_f1", "accuracy", "confusion_matrix"):
                if row[key] != old[key]:
                    raise AssertionError(f"Baseline changed: {key}")

    confirmation_rows = {r["variant"]: r for r in comparisons if r["partition"] == "confirmation" and r["mode"] == "inferred_lengths"}
    before, after = confirmation_rows["before"], confirmation_rows["adjusted"]
    checks = {
        "selected_nonidentity": factor != 1.0,
        "confirmation_neutral_improved": after["neutral_recall"] > before["neutral_recall"],
        "confirmation_weighted_f1_not_lower": after["weighted_f1"] >= before["weighted_f1"],
        "confirmation_macro_f1_not_lower": after["macro_f1"] >= before["macro_f1"],
        "regression_identical_all_124_rows": True,
        "baseline_identical_all_62_rows": True,
    }
    triggers = {r["mode"]: r["relative_weighted_f1_drop"] > 0.10 for r in report["text_middle_30"] if r["method"] == "M1"}
    cli.write_json(output / "training_fingerprint.json", cli.fingerprint(output))
    cli.write_json(deployment / "run_summary.json", {"question": "Q2", "status": "success", "used_split": "valid", "methods": ["M0", "M1"], "seed": core.SEED, "created_at": cli.timestamp(), "report": str(deployment / "report.json")})
    summary = {
        "schema_version": 1, "question": "Q2", "round": output.name, "status": "success",
        "approved_decision_id": "q2_manual_run01_adjust_neutral", "seed": core.SEED,
        "methods": [{"id": "M0", "role": "usable_baseline", "change": "none"}, {"id": "M1", "role": "main_candidate", "change": "Neutral-score multiplier only"}],
        "parent_run": str(source), "parent_fingerprint": source_fingerprint,
        "neutral_factor": factor, "candidate_factors": list(FACTORS),
        "selection_rule": "On tuning subset, improve Neutral recall without reducing weighted-F1/macro-F1; maximize weighted-F1 then macro-F1, tie-break smaller factor.",
        "tuning_rows": len(tuner), "confirmation_rows": len(confirmation),
        "tuning_video_groups": len(set(groups[tuner])), "confirmation_video_groups": len(set(groups[confirmation])),
        "complete_metrics": report["complete"], "partition_checks": checks,
        "text_middle_30_trigger_by_length_mode": triggers,
        "model_bundle": str(output / "models/q2_bundle.joblib"),
        "outputs": ["tables/tuning_candidates.csv", "tables/partition_comparison.csv", "tables/validation_membership.csv", "metrics/modality_diagnostics.json", "deployment_validation/report.json", "deployment_validation/scenario_metrics.json"],
        "elapsed_seconds": time.perf_counter() - started,
        "environment": cli.read_json(source / "metrics/environment.json"),
        "data_policy": {"used_split": "valid", "test_used": False, "attachment3_loaded": False, "estimators_refit": False},
        "warnings": ["Confirmation is a grouped subset of previously inspected valid, not an untouched independent test.", "Normalized adjusted scores are not proven calibrated probabilities.", "Fixed score scaling can trade Neutral precision against recall.", "Result acceptance and final test remain pending."], "errors": [],
    }
    cli.write_json(output / "run_summary.json", summary)
    print(__import__("json").dumps({"run": str(output), "neutral_factor": factor, "partition_checks": checks, "seconds": summary["elapsed_seconds"]}, ensure_ascii=False))


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("lock", "test", "predict"):
        install_adjustment()
        cli.main()
        return
    parser = argparse.ArgumentParser(description=__doc__, epilog="After result acceptance, use this same script with lock/test/predict and --run-dir.")
    parser.add_argument("action", choices=["adjust"])
    parser.add_argument("--source-run", type=Path, default=cli.DEFAULT_RUN)
    parser.add_argument("--run-dir", type=Path, default=cli.ROOT / "results/Q2/experiments/neutral_run01")
    args = parser.parse_args()
    adjust(args.source_run, args.run_dir)


if __name__ == "__main__":
    main()

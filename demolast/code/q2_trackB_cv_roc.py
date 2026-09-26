#!/usr/bin/env python3
"""Q2 Track B (traditional ML) — 5-fold CV + OvR ROC + confusion matrix.

Reproduces the paper-chosen Track B main model M1 (late fusion with
neutral_factor=1.10) and the M0 baseline, runs stratified 5-fold CV on the
train split ONLY, and reuses the frozen test predictions for the ROC /
confusion-matrix headline. The TF-IDF vectorizer and the audio/vision scaler
are re-fit inside each fold (no leakage).
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import run_q2_round1 as core
from q2_adjust_neutral import scale_neutral

OUT = core.OUTPUT_ROOT.parent.parent / "cv_roc" / "trackB"
FROZEN_TEST_DIR = (
    core.ROOT / "results" / "Q2" / "experiments" / "neutral_robust_run01" / "final_test"
)

WEIGHTS = [0.50, 0.25, 0.25]
NEUTRAL_FACTOR = 1.10


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def fit_fold_models(xt_tr, xa_tr, xv_tr, yc_tr, yr_tr, seed):
    scaler = StandardScaler().fit(np.hstack([xa_tr, xv_tr]))
    av_tr = sparse.csr_matrix(scaler.transform(np.hstack([xa_tr, xv_tr])))
    early_tr = sparse.hstack([xt_tr, av_tr], format="csr")
    baseline_cls = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=seed)
    ).fit(early_tr, yc_tr)
    baseline_reg = Ridge(alpha=12.0).fit(early_tr, yr_tr)

    text_cls = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=seed)
    ).fit(xt_tr, yc_tr)
    text_reg = Ridge(alpha=8.0).fit(xt_tr, yr_tr)
    audio_cls, audio_reg = core.fit_extra_trees(xa_tr, yc_tr, yr_tr, seed + 1)
    vision_cls, vision_reg = core.fit_extra_trees(xv_tr, yc_tr, yr_tr, seed + 2)
    return {
        "scaler": scaler,
        "baseline_cls": baseline_cls, "baseline_reg": baseline_reg,
        "text_cls": text_cls, "text_reg": text_reg,
        "audio_cls": audio_cls, "audio_reg": audio_reg,
        "vision_cls": vision_cls, "vision_reg": vision_reg,
    }


def m0_predict(comp, xt, xa, xv):
    av = sparse.csr_matrix(comp["scaler"].transform(np.hstack([xa, xv])))
    early = sparse.hstack([xt, av], format="csr")
    return comp["baseline_cls"].predict_proba(early), comp["baseline_reg"].predict(early)


def m1_predict(comp, xt, xa, xv, neutral_factor=NEUTRAL_FACTOR):
    probs = np.stack([
        comp["text_cls"].predict_proba(xt),
        comp["audio_cls"].predict_proba(xa),
        comp["vision_cls"].predict_proba(xv),
    ])
    reg = np.stack([
        comp["text_reg"].predict(xt),
        comp["audio_reg"].predict(xa),
        comp["vision_reg"].predict(xv),
    ])
    available = np.ones((3, xt.shape[0]), dtype=float)
    available[1] = xa[:, -1] > 0
    available[2] = xv[:, -1] > 0
    weights = np.asarray(WEIGHTS)[:, None] * available
    weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)
    p = np.sum(probs * weights[:, :, None], axis=0)
    y = np.sum(reg * weights, axis=0)
    return scale_neutral(p, neutral_factor), y


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
        per_class[f"class_{c}"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc}
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    return {
        "per_class": per_class,
        "per_class_auc": {str(c): per_class[f"class_{c}"]["auc"] for c in range(n_classes)},
        "macro_auc": macro_auc,
        "macro": {"fpr": fpr_grid.tolist(), "tpr": mean_tpr.tolist(), "auc": macro_auc},
    }


def fold_metrics(yc, yr, probs, reg):
    cls = core.classification_metrics(yc, probs)
    r = core.regression_metrics(yr, reg)
    out = {
        "accuracy": cls["accuracy"], "macro_f1": cls["macro_f1"],
        "weighted_f1": cls["weighted_f1"],
        "negative_recall": cls["negative_recall"],
        "neutral_recall": cls["neutral_recall"],
        "positive_recall": cls["positive_recall"],
        "mae": r["mae"], "pearson": r["pearson"],
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=core.SEED)
    parser.add_argument("--neutral-factor", type=float, default=NEUTRAL_FACTOR)
    args = parser.parse_args()

    with core.DATA_PATH.open("rb") as handle:
        data = pickle.load(handle)
    train = data["train"]
    raw_text = np.asarray(train["raw_text"], dtype=object)
    audio = np.asarray(train["audio"])
    vision = np.asarray(train["vision"])
    audio_lengths = np.asarray(train["audio_lengths"], dtype=int)
    vision_lengths = np.asarray(train["vision_lengths"], dtype=int)
    yc = np.asarray(train["classification_labels"], dtype=int).reshape(-1)
    yr = np.asarray(train["regression_labels"], dtype=float).reshape(-1)
    n = len(yc)

    OUT.mkdir(parents=True, exist_ok=True)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    oof = {
        "M0_probs": np.full((n, 3), np.nan), "M0_reg": np.full(n, np.nan),
        "M1_probs": np.full((n, 3), np.nan), "M1_reg": np.full(n, np.nan),
    }
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(yc, yc)):
        t0 = time.time()
        vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2, max_features=7000, sublinear_tf=True
        )
        xt_tr = vectorizer.fit_transform(map(str, raw_text[tr_idx]))
        xt_va = vectorizer.transform(map(str, raw_text[va_idx]))
        xa_tr = core.sequence_summary(audio[tr_idx], audio_lengths[tr_idx])
        xa_va = core.sequence_summary(audio[va_idx], audio_lengths[va_idx])
        xv_tr = core.sequence_summary(vision[tr_idx], vision_lengths[tr_idx])
        xv_va = core.sequence_summary(vision[va_idx], vision_lengths[va_idx])

        comp = fit_fold_models(
            xt_tr, xa_tr, xv_tr, yc[tr_idx], yr[tr_idx], args.seed
        )
        p0, y0 = m0_predict(comp, xt_va, xa_va, xv_va)
        p1, y1 = m1_predict(comp, xt_va, xa_va, xv_va, args.neutral_factor)

        oof["M0_probs"][va_idx] = p0
        oof["M0_reg"][va_idx] = np.clip(y0, -3, 3)
        oof["M1_probs"][va_idx] = p1
        oof["M1_reg"][va_idx] = np.clip(y1, -3, 3)

        row = {"fold": fold, "elapsed_seconds": time.time() - t0}
        for method, (p, yreg) in (("M0", (p0, y0)), ("M1", (p1, y1))):
            m = fold_metrics(yc[va_idx], yr[va_idx], p, np.clip(yreg, -3, 3))
            row[method] = m
        fold_rows.append(row)
        print(f"fold={fold} M0_acc={row['M0']['accuracy']:.4f} "
              f"M1_acc={row['M1']['accuracy']:.4f} "
              f"M1_wf1={row['M1']['weighted_f1']:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    # Aggregate OOF metrics + ROC + confusion
    summary = {"track": "B", "seed": args.seed, "n_folds": args.n_folds,
               "neutral_factor": args.neutral_factor, "fold_metrics": fold_rows}
    for method in ("M0", "M1"):
        probs = oof[f"{method}_probs"]
        reg = oof[f"{method}_reg"]
        pred = np.argmax(probs, axis=1)
        m = fold_metrics(yc, yr, probs, reg)
        summary[f"{method}_oof_metrics"] = m
        summary[f"{method}_oof_roc"] = ovr_roc(yc, probs)
        summary[f"{method}_oof_confusion_matrix"] = core.classification_metrics(yc, probs)["confusion_matrix"]
        np.save(OUT / f"{method}_oof_probs.npy", probs)
        np.save(OUT / f"{method}_oof_class.npy", pred)
        np.save(OUT / f"{method}_oof_reg.npy", reg)

    # Frozen test predictions (native provided lengths, consistent with CV)
    for method in ("M0", "M1"):
        frame = pd.read_csv(FROZEN_TEST_DIR / f"provided_lengths_{method}_predictions.csv")
        probs = frame[["prob_negative", "prob_neutral", "prob_positive"]].to_numpy()
        y_true = frame["true_class"].to_numpy().astype(int)
        y_true_reg = frame["true_intensity"].to_numpy().astype(float)
        y_pred = frame["predicted_class"].to_numpy().astype(int)
        y_reg = frame["intensity"].to_numpy().astype(float)
        summary[f"{method}_test_metrics"] = fold_metrics(y_true, y_true_reg, probs, y_reg)
        summary[f"{method}_test_roc"] = ovr_roc(y_true, probs)
        summary[f"{method}_test_confusion_matrix"] = core.classification_metrics(y_true, probs)["confusion_matrix"]

    write_json(OUT / "summary.json", summary)
    np.save(OUT / "oof_true_class.npy", yc)
    np.save(OUT / "oof_true_reg.npy", yr)
    print(json.dumps({
        "M1_oof_metrics": summary["M1_oof_metrics"],
        "M1_oof_macro_auc": summary["M1_oof_roc"]["macro_auc"],
        "M1_test_metrics": summary["M1_test_metrics"],
        "M1_test_macro_auc": summary["M1_test_roc"]["macro_auc"],
        "M0_test_metrics": summary["M0_test_metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

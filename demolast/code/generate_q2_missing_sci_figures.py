#!/usr/bin/env python3
"""Generate publication figures for the Q2 missing-modality stress test."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CSV = ROOT / "demolast/results/trackA_delivery/missing_grid.csv"
RUN_SUMMARY = ROOT / "demolast/results/trackA_delivery/run_summary.json"
OUTPUT_DIR = ROOT / "paper/figures"

MODALITIES = ["text", "audio", "vision"]
LOCATIONS = ["early", "middle", "late"]
FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]

MODALITY_LABELS = {"text": "文本", "audio": "语音", "vision": "视觉"}
LOCATION_LABELS = {"early": "前段", "middle": "中段", "late": "后段"}
MODALITY_COLORS = {"text": "#1A6FC4", "audio": "#E28E2C", "vision": "#7B5FD6"}
MODALITY_LINESTYLES = {"text": "-", "audio": "--", "vision": ":"}
LOCATION_MARKERS = {"early": "o", "middle": "s", "late": "^"}
LOCATION_LINESTYLES = {"early": "-", "middle": "--", "late": ":"}

INK = "#2B2B2B"
MUTED = "#666666"
GRID = "#D9D9D9"
BASELINE = "#767676"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Hiragino Sans GB",
                "Heiti SC",
                "Arial Unicode MS",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 9.2,
            "axes.titlesize": 10.8,
            "axes.labelsize": 9.6,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 8.2,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
        }
    )


def load_data() -> tuple[list[dict[str, float | str]], dict[str, float]]:
    with SOURCE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows: list[dict[str, float | str]] = []
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "modality": raw["modality"],
                    "location": raw["location"],
                    "fraction": float(raw["fraction"]),
                    "accuracy": float(raw["accuracy"]),
                    "macro_f1": float(raw["macro_f1"]),
                    "mae": float(raw["mae"]),
                    "pearson": float(raw["pearson"]),
                }
            )

    summary = json.loads(RUN_SUMMARY.read_text(encoding="utf-8"))
    baseline = {
        "macro_f1": float(summary["validation"]["macro_f1"]),
        "mae": float(summary["validation"]["mae"]),
        "accuracy": float(summary["validation"]["accuracy"]),
        "pearson": float(summary["validation"]["pearson"]),
    }

    expected = {(m, l, f) for m in MODALITIES for l in LOCATIONS for f in FRACTIONS}
    observed = {(str(r["modality"]), str(r["location"]), float(r["fraction"])) for r in rows}
    if len(rows) != 45 or observed != expected:
        raise ValueError("Expected exactly 45 unique modality-location-fraction scenarios")
    return rows, baseline


def row_index(rows: list[dict[str, float | str]]) -> dict[tuple[str, str, float], dict[str, float | str]]:
    return {
        (str(r["modality"]), str(r["location"]), float(r["fraction"])): r
        for r in rows
    }


def finish_axes(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.65, alpha=0.7)
        ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.06,
    )
    plt.close(fig)


def plot_heatmap(rows: list[dict[str, float | str]], baseline: dict[str, float]) -> None:
    indexed = row_index(rows)
    row_pairs = [(m, l) for m in MODALITIES for l in LOCATIONS]
    absolute = np.array(
        [[float(indexed[(m, l, f)]["macro_f1"]) for f in FRACTIONS] for m, l in row_pairs]
    )
    delta_pp = (absolute - baseline["macro_f1"]) * 100.0

    cmap = LinearSegmentedColormap.from_list(
        "deviation",
        ["#B64949", "#F1CBC4", "#FAFAFA", "#C9DDF0", "#1A6FC4"],
        N=256,
    )
    limit = float(np.ceil(np.max(np.abs(delta_pp)) * 2) / 2)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, ax = plt.subplots(figsize=(7.25, 4.65), constrained_layout=True)
    image = ax.imshow(delta_pp, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(FRACTIONS)))
    ax.set_xticklabels([f"{f:.0%}" for f in FRACTIONS])
    ax.set_yticks(range(len(row_pairs)))
    ax.set_yticklabels(
        [f"{MODALITY_LABELS[m]} · {LOCATION_LABELS[l]}" for m, l in row_pairs]
    )
    ax.set_xlabel("缺失率")
    ax.set_ylabel("缺失模态与时间位置")
    ax.set_title("连续模态缺失场景下的预测鲁棒性", pad=10, weight="bold")

    for tick, (modality, _) in zip(ax.get_yticklabels(), row_pairs):
        tick.set_color(MODALITY_COLORS[modality])
        tick.set_fontweight("semibold")

    for i in range(absolute.shape[0]):
        for j in range(absolute.shape[1]):
            text_color = "white" if abs(delta_pp[i, j]) >= 1.75 else INK
            ax.text(j, i, f"{absolute[i, j]:.4f}", ha="center", va="center", fontsize=7.7, color=text_color)

    for boundary in (2.5, 5.5):
        ax.axhline(boundary, color="white", linewidth=2.4)
        ax.axhline(boundary, color="#B8B8B8", linewidth=0.65)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("相对完整输入的 ΔMacro-F1（百分点）")
    colorbar.outline.set_linewidth(0.6)
    ax.text(
        0.0,
        -0.16,
        f"单元格为绝对 Macro-F1；完整输入基准 = {baseline['macro_f1']:.4f}。",
        transform=ax.transAxes,
        fontsize=7.7,
        color=MUTED,
        ha="left",
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_figure(fig, "q2_missing_scenario_heatmap")


def blend_with_white(hex_color: str, amount: float) -> tuple[float, float, float]:
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(rgb * (1.0 - amount) + np.ones(3) * amount)


def plot_trajectories(rows: list[dict[str, float | str]], baseline: dict[str, float]) -> None:
    indexed = row_index(rows)
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.85), sharex=True, sharey=True, constrained_layout=True)
    lightening = {"early": 0.05, "middle": 0.30, "late": 0.55}

    for ax, modality, panel in zip(axes, MODALITIES, "abc"):
        for location in LOCATIONS:
            values = [float(indexed[(modality, location, f)]["macro_f1"]) for f in FRACTIONS]
            ax.plot(
                np.array(FRACTIONS) * 100,
                values,
                color=blend_with_white(MODALITY_COLORS[modality], lightening[location]),
                linewidth=1.9,
                linestyle=LOCATION_LINESTYLES[location],
                marker=LOCATION_MARKERS[location],
                markersize=4.7,
                markeredgecolor="white",
                markeredgewidth=0.55,
                label=LOCATION_LABELS[location],
            )
        ax.axhline(
            baseline["macro_f1"],
            color=BASELINE,
            linewidth=1.05,
            linestyle=(0, (4, 3)),
            zorder=0,
        )
        ax.set_title(MODALITY_LABELS[modality], color=MODALITY_COLORS[modality], weight="bold")
        ax.set_xticks([10, 20, 30, 40, 50])
        ax.set_xlabel("缺失率（%）")
        ax.text(-0.14, 1.03, panel, transform=ax.transAxes, fontsize=10.5, fontweight="bold")
        finish_axes(ax, "y")

    axes[0].set_ylabel("Macro-F1")
    axes[0].set_ylim(0.576, 0.6245)
    axes[0].set_yticks([0.58, 0.59, 0.60, 0.61, 0.62])

    location_handles = [
        Line2D(
            [0],
            [0],
            color=INK,
            linewidth=1.6,
            linestyle=LOCATION_LINESTYLES[l],
            marker=LOCATION_MARKERS[l],
            markersize=4.5,
            label=LOCATION_LABELS[l],
        )
        for l in LOCATIONS
    ]
    location_handles.append(
        Line2D([0], [0], color=BASELINE, linewidth=1.05, linestyle=(0, (4, 3)), label="完整输入")
    )
    fig.legend(
        handles=location_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=4,
        frameon=False,
        handlelength=2.4,
    )
    fig.suptitle("不同缺失率下的性能响应轨迹", y=1.17, fontsize=11.3, weight="bold")
    axes[2].text(
        0.98,
        0.035,
        "纵轴局部放大",
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=MUTED,
    )
    save_figure(fig, "q2_missing_ratio_trajectories")


def plot_joint_tradeoff(rows: list[dict[str, float | str]], baseline: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 4.75), constrained_layout=True)

    for modality in MODALITIES:
        for location in LOCATIONS:
            subset = [
                r
                for r in rows
                if r["modality"] == modality and r["location"] == location
            ]
            subset.sort(key=lambda r: float(r["fraction"]))
            x = np.array([(float(r["macro_f1"]) - baseline["macro_f1"]) * 100 for r in subset])
            y = np.array([(float(r["mae"]) - baseline["mae"]) * 100 for r in subset])
            sizes = np.array([42 + 1.7 * float(r["fraction"]) * 100 for r in subset])
            ax.plot(
                x,
                y,
                color=MODALITY_COLORS[modality],
                alpha=0.30,
                linewidth=0.9,
                linestyle=MODALITY_LINESTYLES[modality],
                zorder=1,
            )
            ax.scatter(
                x,
                y,
                s=sizes,
                color=MODALITY_COLORS[modality],
                marker=LOCATION_MARKERS[location],
                alpha=0.82,
                edgecolor="white",
                linewidth=0.65,
                zorder=3,
            )

    ax.axvline(0, color=BASELINE, linestyle=(0, (4, 3)), linewidth=1.0, zorder=0)
    ax.axhline(0, color=BASELINE, linestyle=(0, (4, 3)), linewidth=1.0, zorder=0)
    ax.set_xlabel("相对完整输入的 ΔMacro-F1（百分点）")
    ax.set_ylabel("相对完整输入的 ΔMAE（百分点）")
    ax.set_title("模态缺失下分类与回归性能的联合响应", pad=10, weight="bold")
    finish_axes(ax, None)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)

    worst = min(rows, key=lambda r: float(r["macro_f1"]))
    worst_x = (float(worst["macro_f1"]) - baseline["macro_f1"]) * 100
    worst_y = (float(worst["mae"]) - baseline["mae"]) * 100
    ax.annotate(
        "文本 · 后段 · 50%",
        xy=(worst_x, worst_y),
        xytext=(18, -18),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=7.8,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.7},
    )

    modality_handles = [
        Line2D(
            [0], [0], marker="o", linestyle=MODALITY_LINESTYLES[m], linewidth=1.4, markersize=6,
            markerfacecolor=MODALITY_COLORS[m], markeredgecolor="white",
            label=MODALITY_LABELS[m],
        )
        for m in MODALITIES
    ]
    location_handles = [
        Line2D(
            [0], [0], marker=LOCATION_MARKERS[l], linestyle="none", markersize=6,
            markerfacecolor="#777777", markeredgecolor="white",
            label=LOCATION_LABELS[l],
        )
        for l in LOCATIONS
    ]
    size_handles = [
        plt.scatter([], [], s=42 + 1.7 * rate, color="#B0B0B0", edgecolor="white", label=f"{rate}%")
        for rate in (10, 30, 50)
    ]
    legend1 = ax.legend(
        handles=modality_handles,
        title="模态",
        loc="upper right",
        frameon=False,
        bbox_to_anchor=(1.0, 0.98),
    )
    ax.add_artist(legend1)
    legend2 = ax.legend(
        handles=location_handles,
        title="缺失位置",
        loc="upper right",
        frameon=False,
        bbox_to_anchor=(1.0, 0.70),
    )
    ax.add_artist(legend2)
    ax.legend(
        handles=size_handles,
        title="缺失率",
        loc="upper right",
        frameon=False,
        bbox_to_anchor=(1.0, 0.42),
        labelspacing=0.8,
    )
    save_figure(fig, "q2_missing_joint_response")


def main() -> None:
    configure_style()
    rows, baseline = load_data()
    plot_heatmap(rows, baseline)
    plot_trajectories(rows, baseline)
    plot_joint_tradeoff(rows, baseline)
    source_hash = hashlib.sha256(SOURCE_CSV.read_bytes()).hexdigest()
    print(f"Generated 9 outputs from 45 scenarios; source_sha256={source_hash}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

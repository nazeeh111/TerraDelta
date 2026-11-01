"""Render report figures from immutable comparison JSON, without recomputing metrics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "pairwise": "Pairwise",
    "spatial_tracking": "Calibrated spatial tracking",
    "smoothing": "Causal smoothing",
    "hmm": "Causal HMM",
    "ted": "Causal TED adaptation",
    "sn7_retrospective": "SN7 retrospective adaptation",
    "observation": "Observation GRU",
    "no_quality": "GRU quality ablation",
    "watch_lstm": "WATCH LSTM adaptation",
    "proposal_tracking": "Shared raw proposals",
    "observation_gate": "Observation event gate",
    "gate_no_quality": "Event gate quality ablation",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--reliability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve earlier figures; use a new output directory")
    args.output.mkdir(parents=True)
    data = json.loads(args.comparison.read_text())["methods"]
    methods = list(LABELS)
    colors = [
        "#226c9d" if k == "observation_gate" else "#b36d12" if k == "gate_no_quality" else "#919aa1"
        for k in methods
    ]
    y = np.arange(len(methods))
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12.8, 7.2), sharey=True, gridspec_kw={"width_ratios": [1, 1]}
    )
    left.barh(y - 0.18, [data[k]["event"]["f1"] for k in methods], height=0.32, color=colors)
    left.barh(
        y + 0.18,
        [data[k]["within_62_days"]["f1"] for k in methods],
        height=0.32,
        color=colors,
        alpha=0.42,
    )
    left.set_yticks(y, [LABELS[k] for k in methods])
    left.invert_yaxis()
    left.set_xlabel("Event F1 (solid: all matched; pale: within 62 days)")
    left.set_xlim(0, 0.032)
    left.grid(axis="x", alpha=0.2)
    left.set_axisbelow(True)
    right.barh(
        y,
        [data[k]["false_confirmed_alerts_per_source_valid_km2_year"] for k in methods],
        color=colors,
    )
    right.set_xlabel("False confirmed alerts / source-valid km²-year")
    right.set_xscale("log")
    right.grid(axis="x", alpha=0.2)
    right.set_axisbelow(True)
    fig.suptitle(
        "First frozen study: geographic transfer failed",
        x=0.025,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.025,
        0.925,
        "Three complete confirmation regions · 1,506 label-supported events · globally fixed development thresholds",
        fontsize=10,
    )
    fig.text(
        0.025,
        0.025,
        "Low scores are retained. The quality inputs do not establish a benefit. Monthly visibility intervals are not physical construction dates.",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.255, right=0.98, top=0.86, bottom=0.13, wspace=0.2)
    for ext in ["png", "svg"]:
        fig.savefig(args.output / f"confirmation-v2.{ext}", dpi=180)
    plt.close(fig)
    reliability = json.loads(args.reliability.read_text())["regions"]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8), sharex=True, sharey=True)
    for ax, (region, entries) in zip(axes, reliability.items()):
        ax.plot([0, 1], [0, 1], color="#aaa", linestyle="--", linewidth=1, label="Ideal reference")
        for kind, color in [("observation_gate", "#226c9d"), ("gate_no_quality", "#b36d12")]:
            r = entries[kind]
            bins = r["bins"]
            ax.plot(
                [b["mean_score"] for b in bins],
                [b["positive_fraction"] for b in bins],
                marker="o",
                markersize=4,
                color=color,
                label=f"{LABELS[kind]} (Brier {r['brier']:.4f})",
            )
        ax.set_title(region.split("_")[0] + "\n" + region.split("_")[1], fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Mean frozen event-compatibility score")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("Weighted observed compatible-proposal fraction")
    fig.suptitle(
        "Event-score reliability varies across regions",
        x=0.035,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.035,
        0.025,
        "Every raw proposal totals one across at most four available prefixes. Rare high-score bins can be unstable; this is not event-date coverage.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.88))
    for ext in ["png", "svg"]:
        fig.savefig(args.output / f"event-calibration-v2.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

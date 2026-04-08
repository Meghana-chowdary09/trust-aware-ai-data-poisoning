#!/usr/bin/env python3
"""
Generate publication-style figures for Results section.
Default: cached metrics (seed=42, synthetic data) for instant output.
Use --live to re-run the full training pipeline (slower).

Outputs PNG + PDF under paper_figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "paper_figures"

# Reproducible snapshot: synthetic CERT-like, seed 42 (matches run_experiment defaults)
CACHE = {
    "report": {
        "n_train": 600,
        "n_test": 200,
        "n_features": 12,
        "label_poisoned_samples": 77,
        "autoencoder_test_f1_vs_gt": 0.75,
        "ae_threshold": 1.0261931419372559,
        "xgboost_clean": {
            "f1": 0.8148148148148148,
            "roc_auc": 0.9847146739130435,
            "precision": 1.0,
            "recall": 0.6875,
            "accuracy": 0.975,
        },
        "xgboost_poisoned_no_trust": {
            "f1": 0.7777777777777778,
            "roc_auc": 0.9541440217391305,
            "precision": 0.7,
            "recall": 0.875,
            "accuracy": 0.96,
        },
        "xgboost_poisoned_trust_weighted": {
            "f1": 0.8,
            "roc_auc": 0.9677309782608695,
            "precision": 0.7368421052631579,
            "recall": 0.875,
            "accuracy": 0.965,
        },
    },
    "trust_summary": {
        "mean_trust_label_flipped_users": 0.4666143013330584,
        "mean_trust_other_users": 0.9671593289664685,
    },
    "shap_top": [
        {"feature": "f0", "mean_abs_shap": 0.50097},
        {"feature": "f3", "mean_abs_shap": 0.43436},
        {"feature": "f1", "mean_abs_shap": 0.42808},
        {"feature": "f7", "mean_abs_shap": 0.42094},
        {"feature": "f2", "mean_abs_shap": 0.38573},
        {"feature": "f11", "mean_abs_shap": 0.37918},
        {"feature": "f4", "mean_abs_shap": 0.34377},
        {"feature": "f10", "mean_abs_shap": 0.32762},
        {"feature": "f6", "mean_abs_shap": 0.32018},
        {"feature": "f5", "mean_abs_shap": 0.31616},
    ],
    "drift_per_feature": [
        {"feature": "f0", "p_value": 0.4447, "drift_flag": False},
        {"feature": "f1", "p_value": 0.6765, "drift_flag": False},
        {"feature": "f2", "p_value": 0.5737, "drift_flag": False},
        {"feature": "f3", "p_value": 0.4755, "drift_flag": False},
        {"feature": "f4", "p_value": 0.2628, "drift_flag": False},
        {"feature": "f5", "p_value": 0.8088, "drift_flag": False},
        {"feature": "f6", "p_value": 0.6421, "drift_flag": False},
        {"feature": "f7", "p_value": 0.6765, "drift_flag": False},
        {"feature": "f8", "p_value": 0.1421, "drift_flag": False},
        {"feature": "f9", "p_value": 2.385e-05, "drift_flag": True},
        {"feature": "f10", "p_value": 0.3084, "drift_flag": False},
        {"feature": "f11", "p_value": 0.0456, "drift_flag": True},
    ],
    "inference_counts": {"critical": 2, "elevated": 2, "normal": 16},
}


def _style_axes(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)


def fig_model_comparison(report: dict, out: Path) -> None:
    models = ["Clean", "Poisoned\n(no trust)", "Poisoned +\ntrust-weighted"]
    keys = (
        "xgboost_clean",
        "xgboost_poisoned_no_trust",
        "xgboost_poisoned_trust_weighted",
    )
    f1 = [report[k]["f1"] for k in keys]
    auc = [report[k]["roc_auc"] for k in keys]

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.bar(x - w / 2, f1, w, label="F1 score", color="#2c7fb8")
    ax.bar(x + w / 2, auc, w, label="ROC-AUC", color="#7fcdbb")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Supervised performance on held-out test (synthetic user-level data)")
    _style_axes(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig1_model_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_trust_summary(trust: dict, n_poison_labels: int, out: Path) -> None:
    labels = ["Label-poisoned\ntraining users", "Other training\nusers"]
    means = [
        trust["mean_trust_label_flipped_users"],
        trust["mean_trust_other_users"],
    ]
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    colors = ["#e34a33", "#31a354"]
    bars = ax.bar(labels, means, color=colors, width=0.55)
    ax.set_ylabel("Mean dynamic trust")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Trust after iterative training (n flipped labels = {n_poison_labels})")
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    _style_axes(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig2_mean_trust_poisoning.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_shap(shap_top: list, out: Path) -> None:
    names = [r["feature"] for r in shap_top]
    vals = [r["mean_abs_shap"] for r in shap_top]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    ax.barh(y, vals, color="#8856a7")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| (trust-weighted XGBoost)")
    ax.set_title("Global feature attribution (subset of test users)")
    _style_axes(ax, grid=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig3_shap_feature_importance.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_drift(drift_rows: list, alpha: float, out: Path) -> None:
    names = [r["feature"] for r in drift_rows]
    pvals = [max(r["p_value"], 1e-8) for r in drift_rows]
    colors = ["#cb181d" if r["drift_flag"] else "#2171b5" for r in drift_rows]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    ax.barh(y, [-np.log10(p) for p in pvals], color=colors)
    ax.axvline(-np.log10(alpha), color="black", linestyle="--", linewidth=1, label=f"α = {alpha}")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(r"$-\log_{10}(p)$  (KS test, train vs test)")
    ax.set_title("Per-feature distribution shift proxy")
    ax.legend(frameon=False, loc="lower right")
    _style_axes(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig4_drift_ks_train_vs_test.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_inference_mix(counts: dict, out: Path) -> None:
    """Optional pie / bar of threat levels on a scored sample."""
    order = ["critical", "elevated", "watch", "normal"]
    vals = [counts.get(k, 0) for k in order]
    if sum(vals) == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    ax.bar(order, vals, color=["#cb181d", "#fd8d3c", "#fecc5c", "#74c476"])
    ax.set_ylabel("Count (first N test users scored)")
    ax.set_title("Illustrative inference outcomes (response layer)")
    _style_axes(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig5_inference_threat_mix.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="Re-run full architecture pipeline (slow) instead of cached metrics.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.live:
        from architecture_pipeline import run_full_architecture_pipeline

        o = run_full_architecture_pipeline(force_synthetic=True, seed=42, inference_limit=25)
        report = o["report"]
        trust = o["trust_summary"]
        arch = o["architecture"]
        shap_top = arch["shap"]["top_features"][:10]
        drift_rows = arch["drift"]["per_feature"]
        inf = arch.get("inference_sample") or []
        counts = {"critical": 0, "elevated": 0, "watch": 0, "normal": 0}
        for row in inf:
            counts[row["threat_level"]] = counts.get(row["threat_level"], 0) + 1
        payload = {
            "report": report,
            "trust_summary": trust,
            "shap_top": shap_top,
            "drift_per_feature": drift_rows,
            "inference_counts": counts,
        }
        (OUT_DIR / "metrics_snapshot.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
    else:
        payload = CACHE
        report = CACHE["report"]
        trust = CACHE["trust_summary"]
        shap_top = CACHE["shap_top"]
        drift_rows = CACHE["drift_per_feature"]
        counts = CACHE["inference_counts"]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "figure.facecolor": "white",
        }
    )

    fig_model_comparison(report, OUT_DIR)
    fig_trust_summary(trust, report["label_poisoned_samples"], OUT_DIR)
    fig_shap(shap_top, OUT_DIR)
    fig_drift(drift_rows, 0.05, OUT_DIR)
    fig_inference_mix(counts, OUT_DIR)

    print(f"Figures written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

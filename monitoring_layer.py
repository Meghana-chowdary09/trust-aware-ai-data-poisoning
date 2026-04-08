"""SHAP explainability, statistical drift detection, and JSONL audit logging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


class AuditLogger:
    """Append-only audit trail (aligns with architecture audit logs)."""

    def __init__(self, log_path: Path | None = None):
        self.log_path = log_path
        self.entries: list[dict[str, Any]] = []

    def log(self, stage: str, payload: dict[str, Any]) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            **payload,
        }
        self.entries.append(entry)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    def tail_json(self, n: int = 30) -> list[dict]:
        return self.entries[-n:]


def shap_summary_for_xgboost(
    clf,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    max_background: int = 120,
    max_explain: int = 60,
) -> dict[str, Any]:
    """Mean |SHAP| per feature for a subsample (TreeExplainer)."""
    try:
        import shap
    except ImportError:
        return {
            "available": False,
            "reason": "install shap package for explainability",
            "top_features": [],
        }

    X_bg = X_background[:max_background]
    X_ex = X_explain[:max_explain]
    explainer = shap.TreeExplainer(clf, X_bg)
    sv = explainer.shap_values(X_ex)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    mean_abs = np.mean(np.abs(sv), axis=0)
    order = np.argsort(-mean_abs)
    top = [
        {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
        for i in order[: min(12, len(order))]
    ]
    return {"available": True, "n_explained": int(len(X_ex)), "top_features": top}


def drift_detection_ks(
    X_reference: np.ndarray,
    X_current: np.ndarray,
    feature_names: list[str],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Kolmogorov–Smirnov two-sample test per feature (train vs holdout as proxy for drift).
    """
    flags = []
    for j, name in enumerate(feature_names):
        a, b = X_reference[:, j], X_current[:, j]
        if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
            stat, p = 0.0, 1.0
        else:
            stat, p = stats.ks_2samp(a, b)
        flags.append(
            {
                "feature": name,
                "ks_statistic": float(stat),
                "p_value": float(p),
                "drift_flag": bool(p < alpha),
            }
        )
    n_drift = sum(f["drift_flag"] for f in flags)
    return {
        "method": "ks_2samp",
        "alpha": alpha,
        "n_features_drifted": int(n_drift),
        "per_feature": flags,
    }


def json_safe_architecture(arch: dict) -> dict:
    """Strip non-serializable entries for Streamlit download."""
    out = {}
    for k, v in arch.items():
        if k.startswith("_"):
            continue
        try:
            json.dumps(v, default=str)
            out[k] = v
        except TypeError:
            continue
    return out

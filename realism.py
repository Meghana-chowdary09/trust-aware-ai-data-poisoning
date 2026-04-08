"""Controlled label noise, feature noise, and overfitting diagnostics for evaluation realism."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def apply_label_noise(
    y: np.ndarray,
    noise_fraction: float = 0.08,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """
    Randomly flip a fraction of labels (symmetric noise). Returns (y_new, n_flipped).
    """
    y = np.asarray(y).astype(np.int32).copy()
    n = len(y)
    if n == 0 or noise_fraction <= 0:
        return y, 0
    rng = np.random.default_rng(seed)
    k = max(1, int(round(n * noise_fraction)))
    k = min(k, n)
    flip_idx = rng.choice(n, size=k, replace=False)
    y[flip_idx] = 1 - y[flip_idx]
    return y, int(k)


def apply_numeric_feature_noise_dataframe(
    df: pd.DataFrame,
    user_col: str = "user_id",
    seed: int = 0,
    volume_std: float = 50.0,
) -> pd.DataFrame:
    """
    Add Gaussian noise to numeric columns; special handling for data_volume_mb and failed_attempts.
    user_id is never modified.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(out)
    for c in out.columns:
        if c == user_col:
            continue
        if not pd.api.types.is_numeric_dtype(out[c]):
            continue
        col = out[c].astype(np.float64)
        if c == "data_volume_mb":
            out[c] = col + rng.normal(0, volume_std, n)
        elif c == "failed_attempts":
            out[c] = (col + rng.integers(-1, 2, size=n)).clip(lower=0)
        else:
            med = float(np.nanmedian(np.abs(col.to_numpy())) + 1.0)
            scale = max(0.03 * med, 0.08)
            out[c] = col + rng.normal(0, scale, n)
    return out


def overfitting_warnings(metrics: dict[str, float], threshold: float = 0.98) -> list[str]:
    """Flag suspiciously perfect metrics on clean test evaluation."""
    msgs: list[str] = []
    for key in ("accuracy", "f1", "roc_auc"):
        v = metrics.get(key)
        if v is None or v != v:  # nan
            continue
        if float(v) > threshold:
            msgs.append(
                f"Clean model {key}={float(v):.3f} exceeds {threshold:.2f}: "
                "possible overfitting or an overly separable dataset."
            )
    if msgs:
        msgs.append("Model may be overfitting or dataset too simple")
    return msgs


def summarize_realism(
    label_noise_fraction: float,
    n_label_flips: int,
    feature_noise_applied: bool,
    overfit_msgs: list[str],
) -> dict[str, Any]:
    return {
        "label_noise_fraction": float(label_noise_fraction),
        "labels_randomly_flipped_after_load": int(n_label_flips),
        "gaussian_feature_noise_applied": bool(feature_noise_applied),
        "overfitting_warnings": list(overfit_msgs),
    }

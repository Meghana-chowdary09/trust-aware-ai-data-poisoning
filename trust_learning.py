"""
Dynamic user trust scores and trust-weighted XGBoost training.

Trust is updated from (1) consistency of gradient proxy: high loss users downweighted,
(2) alignment with autoencoder anomaly score vs label (optional).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class TrustState:
    user_ids: np.ndarray
    trust: np.ndarray  # same length as users in training set

    def weights(self, power: float = 1.0) -> np.ndarray:
        w = np.clip(self.trust, 1e-3, 1.0) ** power
        return w / (w.mean() + 1e-8)


def init_trust(user_ids: np.ndarray, value: float = 1.0) -> TrustState:
    u = np.asarray(user_ids)
    return TrustState(user_ids=u, trust=np.full(len(u), value, dtype=np.float64))


def update_trust_from_loss(
    state: TrustState,
    per_sample_loss: np.ndarray,
    lr: float = 0.15,
    min_trust: float = 0.15,
    max_trust: float = 1.0,
) -> TrustState:
    """
    Downweight users whose samples incur high loss (possible poison / mislabeled).
    per_sample_loss aligned with state.user_ids order.
    """
    loss = np.asarray(per_sample_loss, dtype=np.float64)
    if len(loss) != len(state.trust):
        raise ValueError("per_sample_loss length must match trust length")
    # Normalize loss to [0,1]-ish
    hi = np.percentile(loss, 90) + 1e-8
    norm = np.clip(loss / hi, 0, 1)
    # High loss -> decrease trust
    delta = lr * (norm - norm.mean())
    new_t = state.trust - delta
    new_t = np.clip(new_t, min_trust, max_trust)
    return TrustState(user_ids=state.user_ids, trust=new_t)


def update_trust_with_autoencoder(
    state: TrustState,
    y: np.ndarray,
    ae_scores: np.ndarray,
    ae_threshold: float,
    step: float = 0.08,
    min_trust: float = 0.15,
    max_trust: float = 1.0,
) -> TrustState:
    """
    If benign user has very high AE error, slightly reduce trust (noisy / poisoned features).
    If malicious user has very low AE error, reduce trust (may be camouflaged poison).
    """
    y = np.asarray(y)
    high_ae = ae_scores > ae_threshold
    t = state.trust.copy()
    # Benign but looks anomalous -> could be bad data
    mask_b = (y == 0) & high_ae
    t[mask_b] -= step
    # Malicious but not anomalous under AE -> suspicious cleanliness
    mask_m = (y == 1) & (~high_ae)
    t[mask_m] -= step * 0.5
    t = np.clip(t, min_trust, max_trust)
    return TrustState(user_ids=state.user_ids, trust=t)


def train_xgb(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    seed: int = 42,
) -> xgb.XGBClassifier:
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=seed,
        eval_metric="logloss",
        tree_method="hist",
    )
    clf.fit(X, y, sample_weight=sample_weight)
    return clf


def per_sample_log_loss_proxy(
    clf: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Binary log loss per row using predicted positive class probability."""
    p = clf.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-7, 1 - 1e-7)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def evaluate_classifier(clf: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    y = np.asarray(y)
    pred = clf.predict(X)
    proba = clf.predict_proba(X)[:, 1]
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, proba))
    else:
        out["roc_auc"] = float("nan")
    return out


def iterative_trust_weighted_train(
    X: np.ndarray,
    y: np.ndarray,
    user_ids: np.ndarray,
    ae_scores: np.ndarray | None = None,
    ae_threshold: float | None = None,
    rounds: int = 3,
    trust_power: float = 1.5,
    seed: int = 42,
) -> tuple[xgb.XGBClassifier, TrustState]:
    """
    Alternate: fit XGB -> compute per-sample loss -> update trust -> refit with weights.
    Optional AE-guided trust nudge each round.
    """
    state = init_trust(user_ids, 1.0)
    clf = None
    for r in range(rounds):
        w = state.weights(power=trust_power)
        clf = train_xgb(X, y, sample_weight=w, seed=seed + r)
        loss = per_sample_log_loss_proxy(clf, X, y)
        state = update_trust_from_loss(state, loss, lr=0.12 + 0.03 * r)
        if ae_scores is not None and ae_threshold is not None:
            state = update_trust_with_autoencoder(
                state, y, ae_scores, ae_threshold, step=0.06
            )
    assert clf is not None
    return clf, state

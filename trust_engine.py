"""
Dynamic Trust Scoring Engine — named facade over trust_learning helpers.

Feeds trust-weighted signals into downstream AE-assisted updates and XGBoost training
(as in the reference architecture).
"""

from __future__ import annotations

import numpy as np

from trust_learning import (
    TrustState,
    init_trust,
    iterative_trust_weighted_train,
    per_sample_log_loss_proxy,
    train_xgb,
    update_trust_from_loss,
    update_trust_with_autoencoder,
)


class DynamicTrustScoringEngine:
    """Coordinates per-user trust initialization and multi-round refinement."""

    def __init__(self, initial_trust: float = 1.0):
        self.initial_trust = initial_trust

    def initial_state(self, user_ids: np.ndarray) -> TrustState:
        return init_trust(user_ids, value=self.initial_trust)

    def refine_round(
        self,
        state: TrustState,
        clf,
        X: np.ndarray,
        y: np.ndarray,
        ae_scores: np.ndarray | None = None,
        ae_threshold: float | None = None,
        loss_lr: float = 0.12,
    ) -> TrustState:
        loss = per_sample_log_loss_proxy(clf, X, y)
        state = update_trust_from_loss(state, loss, lr=loss_lr)
        if ae_scores is not None and ae_threshold is not None:
            state = update_trust_with_autoencoder(
                state, y, ae_scores, ae_threshold, step=0.06
            )
        return state

    def train_with_iterative_trust(
        self,
        X: np.ndarray,
        y: np.ndarray,
        user_ids: np.ndarray,
        ae_scores: np.ndarray,
        ae_threshold: float,
        rounds: int = 4,
        trust_power: float = 1.6,
        seed: int = 42,
    ) -> tuple[object, TrustState]:
        return iterative_trust_weighted_train(
            X,
            y,
            user_ids,
            ae_scores=ae_scores,
            ae_threshold=ae_threshold,
            rounds=rounds,
            trust_power=trust_power,
            seed=seed,
        )


__all__ = [
    "DynamicTrustScoringEngine",
    "TrustState",
    "train_xgb",
]

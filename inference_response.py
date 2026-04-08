"""Real-time inference & response: fuse AE, XGBoost, and trust into alerts and actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from autoencoder_model import InsiderAutoencoder, reconstruction_errors


@dataclass
class ThreatDecision:
    user_id: str
    threat_level: str
    risk_score: float
    xgb_malicious_proba: float
    ae_reconstruction_mse: float
    trust: float
    actions: list[str]
    notes: str


class RealtimeInferenceResponseLayer:
    """
    Final architecture stage: threat scoring, alerts, and recommended preventive actions.
    Trust for unknown users defaults to `default_trust`.
    """

    def __init__(
        self,
        ae_model: InsiderAutoencoder,
        xgb_clf,
        ae_threshold: float,
        trust_by_user: dict[str, float],
        default_trust: float = 0.85,
    ):
        self.ae_model = ae_model
        self.xgb_clf = xgb_clf
        self.ae_threshold = max(ae_threshold, 1e-6)
        self.trust_by_user = trust_by_user
        self.default_trust = default_trust

    def infer_one(self, x: np.ndarray, user_id: str) -> ThreatDecision:
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        ae_err = float(reconstruction_errors(self.ae_model, x)[0])
        proba = float(self.xgb_clf.predict_proba(x)[0, 1])
        trust = float(self.trust_by_user.get(str(user_id), self.default_trust))

        ae_component = min(1.0, ae_err / self.ae_threshold)
        risk = 0.42 * proba + 0.38 * ae_component + 0.20 * (1.0 - trust)
        risk = float(np.clip(risk, 0.0, 1.0))

        if risk >= 0.78:
            level = "critical"
            actions = [
                "block_high_risk_session",
                "alert_soc_priority",
                "forensic_snapshot",
            ]
            notes = "Combined model and low trust exceed critical threshold."
        elif risk >= 0.55:
            level = "elevated"
            actions = ["alert_analyst", "step_up_auth", "enhanced_monitoring_24h"]
            notes = "Elevated insider-risk score; manual review recommended."
        elif risk >= 0.35:
            level = "watch"
            actions = ["log_for_trending", "optional_manager_notification"]
            notes = "Borderline signals; continue passive monitoring."
        else:
            level = "normal"
            actions = ["allow"]
            notes = "Within expected behavior envelope."

        return ThreatDecision(
            user_id=str(user_id),
            threat_level=level,
            risk_score=risk,
            xgb_malicious_proba=proba,
            ae_reconstruction_mse=ae_err,
            trust=trust,
            actions=actions,
            notes=notes,
        )

    def batch_infer(
        self,
        X: np.ndarray,
        user_ids: np.ndarray,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        n = min(limit, len(X))
        for i in range(n):
            d = self.infer_one(X[i], str(user_ids[i]))
            out.append(
                {
                    "user_id": d.user_id,
                    "threat_level": d.threat_level,
                    "risk_score": d.risk_score,
                    "xgb_malicious_proba": d.xgb_malicious_proba,
                    "ae_reconstruction_mse": d.ae_reconstruction_mse,
                    "trust": d.trust,
                    "actions": d.actions,
                    "notes": d.notes,
                }
            )
        return out


def trust_map_from_training(
    user_ids: np.ndarray,
    trust_values: np.ndarray,
) -> dict[str, float]:
    return {str(u): float(t) for u, t in zip(user_ids, trust_values)}

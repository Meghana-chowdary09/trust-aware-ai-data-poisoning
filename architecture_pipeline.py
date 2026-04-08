"""
Full architecture orchestration: ingestion → provenance → feature pipeline (existing)
→ trust engine + autoencoder + XGBoost → trust-weighted learning →
SHAP / drift monitoring → audit logs → real-time inference & response.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inference_response import RealtimeInferenceResponseLayer, trust_map_from_training
from ingestion_layer import ingest_batch_files
from monitoring_layer import (
    AuditLogger,
    drift_detection_ks,
    json_safe_architecture,
    shap_summary_for_xgboost,
)
from provenance import build_provenance_batch
from run_experiment import run_pipeline


def run_full_architecture_pipeline(
    data_dir: str | None = None,
    seed: int = 42,
    force_synthetic: bool = False,
    label_flip_fraction: float = 0.14,
    malicious_noise_scale: float = 0.4,
    fraction_malicious_feature_poison: float = 0.55,
    ae_epochs: int = 45,
    trust_rounds: int = 4,
    trust_power: float = 1.6,
    audit_log_path: str | Path | None = None,
    inference_limit: int = 20,
    apply_realism: bool = True,
    label_noise_fraction: float = 0.08,
) -> dict[str, Any]:
    """
    Runs run_pipeline with artifacts, then monitoring + inference stages.
    Returned dict includes `architecture` (JSON-friendly) and `_artifacts` (Python objects).
    """
    log_path = Path(audit_log_path) if audit_log_path else None
    audit = AuditLogger(log_path)

    pdir = Path(data_dir) if data_dir else None
    prov = build_provenance_batch(
        pdir,
        synthetic_seed=seed,
        force_synthetic=force_synthetic or pdir is None,
    )
    audit.log("provenance", prov.to_dict())

    manifest = ingest_batch_files(pdir, audit_hook=lambda s, p: audit.log(s, p))
    audit.log("ingestion_manifest", manifest.to_dict())

    audit.log(
        "feature_engineering",
        {
            "event": "start",
            "note": "CERT aggregate per user + StandardScaler (preprocess.dataframe_to_xy)",
        },
    )

    out = run_pipeline(
        data_dir=data_dir,
        seed=seed,
        force_synthetic=force_synthetic,
        label_flip_fraction=label_flip_fraction,
        malicious_noise_scale=malicious_noise_scale,
        fraction_malicious_feature_poison=fraction_malicious_feature_poison,
        ae_epochs=ae_epochs,
        trust_rounds=trust_rounds,
        trust_power=trust_power,
        return_artifacts=True,
        apply_realism=apply_realism,
        label_noise_fraction=label_noise_fraction,
    )
    audit.log(
        "feature_engineering",
        {
            "event": "complete",
            "n_users_train": out["report"]["n_train"],
            "n_features": out["report"]["n_features"],
        },
    )

    art = out.pop("_artifacts", {})
    audit.log(
        "trust_weighted_learning",
        {
            "event": "training_complete",
            "label_poisoned_samples": out["report"]["label_poisoned_samples"],
            "mean_trust_flipped": out["trust_summary"]["mean_trust_label_flipped_users"],
        },
    )

    X_tr = art["X_tr"]
    X_te = art["X_te"]
    y_te = art["y_te"]
    fnames = art["feature_names"]
    clf_trust = art["clf_trust"]
    ae_model = art["ae_model"]
    ae_thresh = art["ae_threshold"]
    users_te = art["users_te"]
    users_tr = art["users_tr"]
    ts = art["trust_state"]

    shap_report = shap_summary_for_xgboost(
        clf_trust, X_tr, X_te, fnames
    )
    audit.log("shap_explainability", {"summary": shap_report})

    drift_report = drift_detection_ks(X_tr, X_te, fnames)
    audit.log(
        "drift_detection",
        {
            "n_features_drifted": drift_report["n_features_drifted"],
            "method": drift_report["method"],
        },
    )

    trust_map = trust_map_from_training(ts.user_ids, ts.trust)
    response = RealtimeInferenceResponseLayer(
        ae_model, clf_trust, ae_thresh, trust_map, default_trust=0.85
    )
    decisions = response.batch_infer(X_te, users_te, limit=inference_limit)
    audit.log(
        "realtime_inference_response",
        {
            "n_scored": len(decisions),
            "critical_count": sum(1 for d in decisions if d["threat_level"] == "critical"),
            "elevated_count": sum(1 for d in decisions if d["threat_level"] == "elevated"),
        },
    )

    audit.log(
        "audit",
        {"event": "pipeline_complete", "dataset_id": prov.dataset_id},
    )

    architecture: dict[str, Any] = {
        "provenance": prov.to_dict(),
        "ingestion": manifest.to_dict(),
        "dataset_meta": out.get("dataset_meta", {}),
        "shap": shap_report,
        "drift": {
            "method": drift_report["method"],
            "alpha": drift_report["alpha"],
            "n_features_drifted": drift_report["n_features_drifted"],
            "per_feature": drift_report["per_feature"],
        },
        "inference_sample": decisions,
        "audit_tail": audit.tail_json(40),
        "audit_log_file": str(log_path.resolve()) if log_path else None,
    }

    out["architecture"] = json_safe_architecture(architecture)
    out["_artifacts"] = art
    out["_audit_logger"] = audit
    return out


def architecture_report_for_json(full_out: dict) -> dict:
    """Drop non-serializable keys for CLI / download."""
    d = {k: v for k, v in full_out.items() if not k.startswith("_")}
    d["architecture"] = full_out.get("architecture", {})
    return d


def print_architecture_json(full_out: dict) -> None:
    print(json.dumps(architecture_report_for_json(full_out), indent=2, default=str))

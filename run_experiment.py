#!/usr/bin/env python3
"""
End-to-end experiment: CERT-style data (or synthetic), poisoning, AE, XGBoost,
dynamic trust, trust-weighted learning, and comparison vs poisoned baseline.

Usage:
  python run_experiment.py
  python run_experiment.py --data_dir "C:/path/to/cert_csv_folder"

Kaggle: download https://www.kaggle.com/datasets/nitishabharathi/cert-insider-threat
and point --data_dir at the folder containing logon.csv, device.csv, etc.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from autoencoder_model import anomaly_scores_from_ae, train_autoencoder
from cert_loader import (
    load_cert_user_features,
    load_generic_user_feature_csv,
    load_malicious_users,
)
from custom_insider_csv import load_custom_insider_from_directory
from preprocess import (
    apply_feature_poisoning,
    apply_label_poisoning,
    dataframe_to_xy,
    dataframe_to_xy_with_ordinal_cats,
    synthetic_cert_like,
)
from realism import (
    apply_label_noise,
    apply_numeric_feature_noise_dataframe,
    overfitting_warnings,
    summarize_realism,
)
from trust_learning import (
    evaluate_classifier,
    iterative_trust_weighted_train,
    train_xgb,
)


def build_dataset_from_cert(
    data_dir: str, random_state: int
) -> tuple[pd.DataFrame, np.ndarray, list[str], str, dict | None]:
    """
    Returns (features_df, labels, warnings, source_tag, pipeline_opts).
    pipeline_opts may include categorical_cols + dataset_meta for custom insider CSV.
    """
    data_path = Path(data_dir)

    custom = load_custom_insider_from_directory(data_path, seed=random_state)
    if custom is not None:
        df_c, y_c, meta, w_c = custom
        opts = {
            "categorical_cols": meta.get("categorical_cols", []),
            "dataset_meta": {
                "label_source": meta["label_source"],
                "n_rows": meta["n_rows"],
                "feature_columns": meta["feature_columns"],
                "n_features_engineered": len(meta["feature_columns"]),
                "preview_records": meta.get("preview_records", []),
                "preview_columns": meta.get("preview_columns", []),
                "schema": "custom_insider_events",
            },
        }
        return df_c, y_c, w_c, "custom_insider_events", opts

    feat = load_cert_user_features(data_path)
    if feat is not None and not feat.empty:
        mal_users = load_malicious_users(data_path)
        users = feat["user_id"].astype(str).values
        warnings: list[str] = []
        if not mal_users:
            num = feat.select_dtypes(include=[np.number])
            score = num.sum(axis=1).values
            thresh = np.percentile(score, 92)
            y = (score >= thresh).astype(np.int32)
            msg = (
                "No insider answer files found; generated labels from aggregate activity scores (CERT path)."
            )
            warnings.append(msg)
            print(f"Warning: {msg}", file=sys.stderr)
        else:
            y = np.array([1 if u in mal_users else 0 for u in users], dtype=np.int32)
        return feat, y, warnings, "cert", None

    generic = load_generic_user_feature_csv(data_path)
    if generic is not None:
        feat_g, y_g, w_g = generic
        return feat_g, y_g, w_g, "csv_tabular", None

    raise RuntimeError("No CERT CSVs found in data_dir")


def run_pipeline(
    data_dir: str | None = None,
    seed: int = 42,
    force_synthetic: bool = False,
    label_flip_fraction: float = 0.14,
    malicious_noise_scale: float = 0.4,
    fraction_malicious_feature_poison: float = 0.55,
    ae_epochs: int = 45,
    trust_rounds: int = 4,
    trust_power: float = 1.6,
    return_artifacts: bool = False,
    apply_realism: bool = True,
    label_noise_fraction: float = 0.08,
) -> dict:
    """
    Run full trust-aware experiment. Returns dict with report, warnings, trust_summary, data_source.
    If return_artifacts=True, includes _artifacts (models, arrays) for monitoring / inference — not JSON-serializable.
    """
    warnings: list[str] = []
    data_source = "synthetic"
    df: pd.DataFrame
    y: np.ndarray

    pipeline_opts: dict | None = None
    if (
        not force_synthetic
        and data_dir
        and Path(data_dir).is_dir()
    ):
        try:
            df, y, w, src_tag, pipeline_opts = build_dataset_from_cert(data_dir, seed)
            warnings.extend(w)
            data_source = src_tag
        except RuntimeError:
            msg = "CERT folder had no usable CSVs; using synthetic data."
            warnings.append(msg)
            print(msg, file=sys.stderr)
            df, y, _ = synthetic_cert_like(random_state=seed, realistic_messy=True)
            data_source = "synthetic"
            pipeline_opts = None
    else:
        df, y, _ = synthetic_cert_like(random_state=seed, realistic_messy=True)
        pipeline_opts = None

    realism_stats: dict = {
        "apply_realism": bool(apply_realism),
        "label_noise_fraction": float(label_noise_fraction),
        "labels_flipped_eval_noise": 0,
        "gaussian_feature_noise_applied": False,
    }
    if apply_realism:
        df = apply_numeric_feature_noise_dataframe(
            df, user_col="user_id", seed=seed + 11, volume_std=50.0
        )
        realism_stats["gaussian_feature_noise_applied"] = True
        y, n_flip_eval = apply_label_noise(
            y, noise_fraction=label_noise_fraction, seed=seed + 13
        )
        realism_stats["labels_flipped_eval_noise"] = int(n_flip_eval)
        if pipeline_opts and pipeline_opts.get("dataset_meta") is not None:
            dm0 = pipeline_opts["dataset_meta"]
            prev = df.head(15)
            dm0["preview_records"] = prev.to_dict(orient="records")
            dm0["preview_columns"] = list(prev.columns)

    if pipeline_opts and pipeline_opts.get("categorical_cols"):
        train_ds, test_ds = dataframe_to_xy_with_ordinal_cats(
            df,
            y,
            categorical_cols=pipeline_opts["categorical_cols"],
            random_state=seed,
        )
    else:
        train_ds, test_ds = dataframe_to_xy(df, y, random_state=seed)

    dataset_meta: dict = {
        "label_source": "n/a",
        "n_features_used": len(train_ds.feature_names),
        "feature_names": list(train_ds.feature_names),
        "n_rows_input": int(len(y)),
        "evaluation_label_noise_fraction": float(
            label_noise_fraction if apply_realism else 0.0
        ),
        "evaluation_label_flips": int(realism_stats.get("labels_flipped_eval_noise", 0)),
        "gaussian_feature_noise_applied": bool(
            realism_stats.get("gaussian_feature_noise_applied", False)
        ),
        "overfitting_warnings_clean": [],
    }
    if pipeline_opts and pipeline_opts.get("dataset_meta"):
        dm = pipeline_opts["dataset_meta"]
        ls = dm.get("label_source", "n/a")
        dataset_meta["label_source"] = ls
        dataset_meta["label_message"] = (
            "Using provided labels from CSV."
            if ls == "provided"
            else (
                "Generated labels from probabilistic risk scores (failed_attempts, volume, location, privilege) plus uniform noise; not strict logical rules."
                if ls == "probabilistic_rules"
                else (
                    "Generated labels from rules: failed_attempts > 3 OR data_volume_mb > 500 OR location not in ['Bangalore']."
                    if ls == "rules"
                    else "See warnings for label definition."
                )
            )
        )
        dataset_meta["n_rows_input"] = dm.get("n_rows", len(y))
        dataset_meta["preview_records"] = dm.get("preview_records", [])
        dataset_meta["preview_columns"] = dm.get("preview_columns", [])
        dataset_meta["schema"] = dm.get("schema", "")
    elif data_source == "cert":
        dataset_meta["label_message"] = (
            "Labels from insider answer files, or generated from aggregate activity (CERT)."
        )
    elif data_source == "csv_tabular":
        dataset_meta["label_message"] = "Labels from CSV column, answer files, or aggregate numeric features."
    else:
        dataset_meta["label_message"] = (
            "Synthetic overlapping user-level features; optional evaluation-time feature + label noise (see dataset_meta)."
            if apply_realism
            else "Synthetic data with built-in ground-truth labels."
        )

    X_tr, y_tr = train_ds.X, train_ds.y
    X_te, y_te = test_ds.X, test_ds.y
    users_tr = train_ds.user_ids

    benign_mask = y_tr == 0
    if benign_mask.sum() < 10:
        benign_mask = np.ones(len(y_tr), dtype=bool)
    ae_model, _ = train_autoencoder(
        X_tr[benign_mask],
        epochs=max(5, ae_epochs),
        batch_size=min(128, max(1, len(X_tr[benign_mask]))),
        latent_dim=min(8, X_tr.shape[1]),
        seed=seed,
    )
    ae_scores_tr, ae_thresh = anomaly_scores_from_ae(ae_model, X_tr[benign_mask], X_tr)
    ae_scores_te, _ = anomaly_scores_from_ae(ae_model, X_tr[benign_mask], X_te)

    ae_pred_te = (ae_scores_te > ae_thresh).astype(int)
    ae_f1 = float(f1_score(y_te, ae_pred_te, zero_division=0))

    clf_clean = train_xgb(X_tr, y_tr, sample_weight=None, seed=seed)
    metrics_clean = evaluate_classifier(clf_clean, X_te, y_te)
    of_msgs = overfitting_warnings(metrics_clean, threshold=0.98)
    for msg in of_msgs:
        warnings.append(msg)
    realism_stats["overfitting_warnings"] = of_msgs
    realism_stats["realism_summary"] = summarize_realism(
        label_noise_fraction if apply_realism else 0.0,
        int(realism_stats.get("labels_flipped_eval_noise", 0)),
        bool(realism_stats.get("gaussian_feature_noise_applied")),
        of_msgs,
    )
    dataset_meta["overfitting_warnings_clean"] = list(of_msgs)

    y_poison, poison_label_mask = apply_label_poisoning(
        y_tr,
        users_tr,
        flip_fraction=label_flip_fraction,
        target_benign_only=True,
        random_state=seed,
    )
    X_poison = apply_feature_poisoning(
        X_tr,
        y_tr,
        users_tr,
        malicious_noise_scale=malicious_noise_scale,
        fraction_malicious_rows=fraction_malicious_feature_poison,
        random_state=seed + 7,
    )

    clf_poison = train_xgb(X_poison, y_poison, sample_weight=None, seed=seed)
    metrics_poison = evaluate_classifier(clf_poison, X_te, y_te)

    clf_trust, trust_state = iterative_trust_weighted_train(
        X_poison,
        y_poison,
        users_tr,
        ae_scores=ae_scores_tr,
        ae_threshold=ae_thresh,
        rounds=max(1, trust_rounds),
        trust_power=trust_power,
        seed=seed,
    )
    metrics_trust = evaluate_classifier(clf_trust, X_te, y_te)

    report = {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_features": int(X_tr.shape[1]),
        "label_poisoned_samples": int(poison_label_mask.sum()),
        "autoencoder_test_f1_vs_gt": ae_f1,
        "ae_threshold": float(ae_thresh),
        "xgboost_clean": metrics_clean,
        "xgboost_poisoned_no_trust": metrics_poison,
        "xgboost_poisoned_trust_weighted": metrics_trust,
    }

    mean_trust_poisoned = float(
        trust_state.trust[poison_label_mask].mean()
        if poison_label_mask.any()
        else float("nan")
    )
    mean_trust_clean = float(
        trust_state.trust[~poison_label_mask].mean()
        if (~poison_label_mask).any()
        else float("nan")
    )

    out: dict = {
        "report": report,
        "warnings": warnings,
        "trust_summary": {
            "mean_trust_label_flipped_users": mean_trust_poisoned,
            "mean_trust_other_users": mean_trust_clean,
        },
        "data_source": data_source,
        "dataset_meta": dataset_meta,
        "realism": realism_stats,
    }
    if return_artifacts:
        out["_artifacts"] = {
            "ae_model": ae_model,
            "clf_trust": clf_trust,
            "X_tr": X_tr,
            "X_te": X_te,
            "y_te": y_te,
            "users_te": test_ds.user_ids,
            "users_tr": users_tr,
            "feature_names": train_ds.feature_names,
            "ae_threshold": float(ae_thresh),
            "trust_state": trust_state,
            "poison_label_mask": poison_label_mask,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Folder with CERT CSVs (logon.csv, device.csv, ...). Empty = synthetic data.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-realism",
        action="store_true",
        help="Disable evaluation-time Gaussian feature noise and symmetric label flips.",
    )
    parser.add_argument(
        "--label-noise",
        type=float,
        default=0.08,
        help="Fraction of labels randomly flipped after load (default 0.08). Ignored with --no-realism.",
    )
    parser.add_argument(
        "--full-architecture",
        action="store_true",
        help="Run ingestion, provenance, SHAP, drift, audit log, and inference response layer.",
    )
    parser.add_argument(
        "--audit-log",
        type=str,
        default="",
        help="JSONL audit log path (default: artifacts/audit.jsonl under cwd when --full-architecture).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir and Path(args.data_dir).is_dir() else None
    force_syn = data_dir is None

    if args.full_architecture:
        from architecture_pipeline import architecture_report_for_json, run_full_architecture_pipeline

        audit_path = args.audit_log or str(Path.cwd() / "artifacts" / "audit.jsonl")
        out = run_full_architecture_pipeline(
            data_dir=data_dir,
            seed=args.seed,
            force_synthetic=force_syn,
            audit_log_path=audit_path,
            apply_realism=not args.no_realism,
            label_noise_fraction=float(args.label_noise),
        )
        for w in out["warnings"]:
            print(f"Warning: {w}", file=sys.stderr)
        print(json.dumps(architecture_report_for_json(out), indent=2, default=str))
    else:
        out = run_pipeline(
            data_dir=data_dir,
            seed=args.seed,
            force_synthetic=force_syn,
            apply_realism=not args.no_realism,
            label_noise_fraction=float(args.label_noise),
        )
        for w in out["warnings"]:
            print(f"Warning: {w}", file=sys.stderr)
        print(json.dumps(out["report"], indent=2))
        ts = out["trust_summary"]
        print(
            "\nTrust summary (training users): "
            f"mean trust among label-flipped users = {ts['mean_trust_label_flipped_users']:.3f}; "
            f"mean trust among non-flipped = {ts['mean_trust_other_users']:.3f}"
        )


if __name__ == "__main__":
    main()

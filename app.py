"""
Streamlit demo: upload CERT-style CSVs or a ZIP, or use synthetic data;
run trust-aware poisoning experiment and show metrics in the browser.

Run: streamlit run app.py
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from architecture_pipeline import architecture_report_for_json, run_full_architecture_pipeline

st.set_page_config(
    page_title="Trust-aware insider threat demo",
    page_icon="🛡️",
    layout="wide",
)

st.title("Trust-aware insider threat — full architecture demo")
st.markdown(
    "End-to-end flow: **ingestion** (batch / Kafka-ready) → **provenance** → **feature engineering** → "
    "**dynamic trust** + **autoencoder** + **XGBoost** → **trust-weighted learning** → "
    "**SHAP** + **drift detection** → **audit logs** → **real-time inference & response**. "
    "Upload **CERT-style** CSVs or a **ZIP**, or use **synthetic** data."
)


def prepare_upload_dir_csv(files: list) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="cert_upload_"))
    for f in files:
        name = getattr(f, "name", "upload.csv") or "upload.csv"
        dest = tmp / Path(name).name
        dest.write_bytes(f.getvalue())
    return tmp


def prepare_upload_dir_zip(zipped) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="cert_zip_"))
    with zipfile.ZipFile(zipped, "r") as zf:
        zf.extractall(tmp)
    return find_cert_root(tmp)


def find_cert_root(base: Path) -> Path:
    """If CSVs are nested (e.g. Kaggle zip), use the folder that contains logon.csv."""
    if (base / "logon.csv").is_file():
        return base
    found = next(base.rglob("logon.csv"), None)
    if found is not None:
        return found.parent
    return base


with st.sidebar:
    st.header("Data source")
    mode = st.radio(
        "Choose input",
        ("Synthetic demo data", "Upload CSV file(s)", "Upload ZIP archive"),
        index=0,
    )
    data_dir: str | None = None
    if mode == "Upload CSV file(s)":
        uploaded = st.file_uploader(
            "**Custom insider events:** `user_id`, `timestamp`, `action`, `device`, `location`, "
            "`data_volume_mb`, `failed_attempts`, `privilege_level`, optional `label` — plus "
            "CERT or wide numeric tables. **≥50 rows** recommended.",
            type=["csv"],
            accept_multiple_files=True,
        )
        if uploaded:
            data_dir = str(prepare_upload_dir_csv(list(uploaded)))
            st.caption(f"Using temp folder with {len(uploaded)} file(s).")
    elif mode == "Upload ZIP archive":
        zf = st.file_uploader("ZIP containing CSVs (and optional answer CSVs)", type=["zip"])
        if zf is not None:
            data_dir = str(prepare_upload_dir_zip(zf))
            st.caption("ZIP extracted to a temporary folder.")

    st.header("Experiment parameters")
    seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1)
    apply_eval_realism = st.checkbox(
        "Evaluation realism (Gaussian feature noise + random label flips)",
        value=True,
        help="Applied after loading data, before train/test split. Improves metric realism for papers/demos.",
    )
    eval_label_noise = st.slider(
        "Eval label noise (fraction flipped)",
        0.0,
        0.15,
        0.08,
        0.01,
        disabled=not apply_eval_realism,
    )
    label_flip = st.slider("Label poisoning rate (benign flipped)", 0.0, 0.4, 0.14, 0.01)
    feat_noise = st.slider("Feature noise scale (malicious rows)", 0.0, 1.0, 0.4, 0.05)
    mal_frac = st.slider("Fraction of malicious rows feature-poisoned", 0.0, 1.0, 0.55, 0.05)
    ae_epochs = st.slider("Autoencoder epochs", 10, 120, 45, 5)
    trust_rounds = st.slider("Trust-weighted refit rounds", 1, 8, 4, 1)
    trust_power = st.slider("Trust weight power", 0.5, 3.0, 1.6, 0.1)

run_btn = st.button("Run experiment", type="primary", width="stretch")

if not run_btn:
    st.info("Configure the sidebar and click **Run experiment**.")
    st.stop()

if mode != "Synthetic demo data" and not data_dir:
    st.error("Please upload at least one CSV or a ZIP before running.")
    st.stop()

_audit_path = Path(__file__).resolve().parent / "artifacts" / "audit.jsonl"

with st.spinner("Running full pipeline (train + SHAP + drift + inference)…"):
    try:
        out = run_full_architecture_pipeline(
            data_dir=data_dir,
            seed=int(seed),
            force_synthetic=(mode == "Synthetic demo data"),
            label_flip_fraction=float(label_flip),
            malicious_noise_scale=float(feat_noise),
            fraction_malicious_feature_poison=float(mal_frac),
            ae_epochs=int(ae_epochs),
            trust_rounds=int(trust_rounds),
            trust_power=float(trust_power),
            audit_log_path=str(_audit_path),
            inference_limit=25,
            apply_realism=apply_eval_realism,
            label_noise_fraction=float(eval_label_noise),
        )
    except Exception as e:
        st.exception(e)
        st.stop()

report = out["report"]
for w in out.get("warnings") or []:
    st.warning(w)

st.success(f"Data source: **{out['data_source']}**")

dm = out.get("dataset_meta") or {}
if dm.get("label_message"):
    st.info(dm["label_message"])
arch_dm = (out.get("architecture") or {}).get("dataset_meta") or dm
if arch_dm.get("label_source") and arch_dm["label_source"] != "n/a":
    st.caption(
        f"Label mode: **{arch_dm['label_source']}** "
        f"(`provided` = CSV column; `probabilistic_rules` = scored risk + noise; `rules` = hard thresholds)"
    )

rn = out.get("realism") or {}
if rn.get("apply_realism"):
    st.subheader("Evaluation realism")
    r1, r2, r3 = st.columns(3)
    r1.metric("Label noise fraction", f"{rn.get('label_noise_fraction', 0):.2f}")
    r2.metric("Labels flipped (eval noise)", int(rn.get("labels_flipped_eval_noise", 0)))
    r3.metric("Gaussian feature noise", "on" if rn.get("gaussian_feature_noise_applied") else "off")
elif rn:
    st.caption("Evaluation realism disabled for this run.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Train rows", report["n_train"])
c2.metric("Test rows", report["n_test"])
c3.metric("Features used (models)", dm.get("n_features_used", report["n_features"]))
c4.metric("Label-poisoned (train)", report["label_poisoned_samples"])
fn = dm.get("feature_names") or []
if fn:
    with st.expander("Feature names passed to Autoencoder & XGBoost", expanded=False):
        st.code(", ".join(fn))

preview = dm.get("preview_records") or []
if preview:
    st.subheader("Processed dataset preview (engineered columns)")
    st.dataframe(pd.DataFrame(preview), width="stretch")

st.subheader("Autoencoder (unsupervised, test F1 vs ground truth)")
st.metric("AE test F1 (calibration)", f"{report['autoencoder_test_f1_vs_gt']:.3f}")
st.caption(f"Reconstruction threshold (MSE): {report['ae_threshold']:.4f}")

st.subheader("XGBoost comparison (held-out test set)")
rows = []
for name, key in (
    ("Clean (no poisoning)", "xgboost_clean"),
    ("Poisoned, no trust", "xgboost_poisoned_no_trust"),
    ("Poisoned + trust-weighted", "xgboost_poisoned_trust_weighted"),
):
    m = report[key]
    rows.append(
        {
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "ROC-AUC": m["roc_auc"],
        }
    )
df_metrics = pd.DataFrame(rows)
st.dataframe(
    df_metrics.style.format(
        {
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1": "{:.3f}",
            "ROC-AUC": "{:.3f}",
        }
    ),
    width="stretch",
)

chart_df = df_metrics.set_index("Model")[["F1", "ROC-AUC"]]
st.bar_chart(chart_df)

ts = out["trust_summary"]
st.subheader("Trust summary (training users)")
t1, t2 = st.columns(2)
t1.metric("Mean trust — label-flipped users", f"{ts['mean_trust_label_flipped_users']:.3f}")
t2.metric("Mean trust — other users", f"{ts['mean_trust_other_users']:.3f}")

arch = out.get("architecture") or {}
with st.expander("Ingestion & provenance", expanded=False):
    st.json({"ingestion": arch.get("ingestion"), "provenance": arch.get("provenance")})

with st.expander("SHAP explainability (trust-weighted XGBoost)", expanded=False):
    sh = arch.get("shap") or {}
    if sh.get("available"):
        st.dataframe(pd.DataFrame(sh.get("top_features", [])), width="stretch")
    else:
        st.info(sh.get("reason", "SHAP not available."))

with st.expander("Drift detection (train vs test)", expanded=False):
    drift = arch.get("drift") or {}
    st.metric("Features with significant KS drift (p<α)", drift.get("n_features_drifted", 0))
    if drift.get("per_feature"):
        st.dataframe(pd.DataFrame(drift["per_feature"]), width="stretch")

with st.expander("Real-time inference & response (sample)", expanded=True):
    inf = arch.get("inference_sample") or []
    if inf:
        st.dataframe(pd.DataFrame(inf), width="stretch")

with st.expander("Audit log (recent entries)", expanded=False):
    tail = arch.get("audit_tail") or []
    st.caption(arch.get("audit_log_file") or "In-memory only")
    st.json(tail[-25:] if tail else [])

_safe = architecture_report_for_json(out)
st.download_button(
    "Download full JSON report (architecture + metrics)",
    data=json.dumps(_safe, indent=2, default=str),
    file_name="trust_aware_architecture_report.json",
    mime="application/json",
)

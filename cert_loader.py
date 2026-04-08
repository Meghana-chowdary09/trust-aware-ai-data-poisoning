"""
Load CERT Insider Threat–style CSVs from a Kaggle download folder.

Expected layout (common on Kaggle mirrors of CERT r4.2):
  DATA_DIR/
    logon.csv
    device.csv
    email.csv
    file.csv
    http.csv          (optional)
    psychometric.csv  (optional)
    answers/ or ground truth files (optional)

If files are missing, callers should use synthetic_cert_features() from preprocess.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def load_logon_features(data_dir: str | Path) -> Optional[pd.DataFrame]:
    """Aggregate logon-related counts per user."""
    p = Path(data_dir) / "logon.csv"
    df = _read_csv_safe(p)
    if df is None or df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}
    user_col = next(
        (cols[k] for k in ("user", "username", "employee_name") if k in cols),
        df.columns[2] if len(df.columns) > 2 else df.columns[0],
    )
    u = df[user_col].astype(str)
    feats = pd.DataFrame({"user_id": u})
    if "activity" in cols:
        act = df[cols["activity"]].astype(str).str.lower()
        feats["logon_count"] = (act == "logon").astype(int)
        feats["logoff_count"] = (act == "logoff").astype(int)
    else:
        feats["logon_count"] = 1
        feats["logoff_count"] = 0
    # after-hours if hour in date
    date_col = next((cols[k] for k in ("date", "datetime", "time") if k in cols), None)
    if date_col:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        hour = dt.dt.hour.fillna(12)
        feats["after_hours"] = ((hour < 7) | (hour > 19)).astype(int)
    else:
        feats["after_hours"] = 0
    g = feats.groupby("user_id", as_index=False).sum()
    return g


def load_device_features(data_dir: str | Path) -> Optional[pd.DataFrame]:
    p = Path(data_dir) / "device.csv"
    df = _read_csv_safe(p)
    if df is None or df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}
    user_col = next(
        (cols[k] for k in ("user", "username", "employee_name") if k in cols),
        df.columns[0],
    )
    g = df.groupby(df[user_col].astype(str), as_index=False).size()
    g.columns = ["user_id", "device_events"]
    return g


def load_email_features(data_dir: str | Path) -> Optional[pd.DataFrame]:
    p = Path(data_dir) / "email.csv"
    df = _read_csv_safe(p)
    if df is None or df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}
    user_col = next(
        (cols[k] for k in ("user", "username", "employee_name", "from") if k in cols),
        df.columns[0],
    )
    g = df.groupby(df[user_col].astype(str), as_index=False).size()
    g.columns = ["user_id", "email_events"]
    return g


def load_file_features(data_dir: str | Path) -> Optional[pd.DataFrame]:
    p = Path(data_dir) / "file.csv"
    df = _read_csv_safe(p)
    if df is None or df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}
    user_col = next(
        (cols[k] for k in ("user", "username", "employee_name") if k in cols),
        df.columns[0],
    )
    g = df.groupby(df[user_col].astype(str), as_index=False).size()
    g.columns = ["user_id", "file_events"]
    return g


def merge_user_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    if not tables:
        raise ValueError("No tables to merge")
    out = tables[0]
    for t in tables[1:]:
        out = out.merge(t, on="user_id", how="outer")
    return out.fillna(0)


def load_cert_user_features(data_dir: str | Path) -> Optional[pd.DataFrame]:
    """
    Build one row per user_id with numeric activity aggregates.
    Returns None if no recognizable CSVs exist.
    """
    data_dir = Path(data_dir)
    parts = []
    for loader in (
        load_logon_features,
        load_device_features,
        load_email_features,
        load_file_features,
    ):
        f = loader(data_dir)
        if f is not None:
            parts.append(f)
    if not parts:
        return None
    return merge_user_tables(parts)


USER_ID_COLUMN_KEYS = (
    "user_id",
    "userid",
    "user",
    "username",
    "employee_id",
    "employee_name",
    "emp_id",
    "employee",
    "name",
)
LABEL_COLUMN_KEYS = (
    "label",
    "target",
    "y",
    "is_insider",
    "insider",
    "malicious",
    "class",
    "threat",
    "is_malicious",
    "anomaly",
)


def _column_map(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().lower(): str(c).strip() for c in df.columns}


def _labels_from_series(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s):
        v = s.astype(float).fillna(0).astype(np.int32)
        return np.clip(v, 0, 1)
    sl = s.astype(str).str.lower().str.strip()
    pos = {
        "1",
        "true",
        "yes",
        "malicious",
        "insider",
        "positive",
        "attack",
        "threat",
        "anomaly",
    }
    return sl.isin(pos).astype(np.int32).values


def load_generic_user_feature_csv(data_dir: str | Path) -> Optional[tuple[pd.DataFrame, np.ndarray, list[str]]]:
    """
    Load a single wide CSV: one row per user with a user-id column and numeric features.
    Optional label column (label, target, is_insider, class, ...).

    Returns (feature_df with user_id + numeric cols, y, warnings) or None if no suitable file.
    """
    data_dir = Path(data_dir)
    warnings: list[str] = []
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        return None

    best: Optional[tuple[pd.DataFrame, np.ndarray, list[str], int]] = None

    for path in csvs:
        df = _read_csv_safe(path)
        if df is None or df.empty or len(df) < 8:
            continue
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        cmap = _column_map(df)

        user_col = None
        for k in USER_ID_COLUMN_KEYS:
            if k in cmap:
                user_col = cmap[k]
                break
        if user_col is None:
            for low, orig in cmap.items():
                if "user" in low or low.endswith("_id"):
                    user_col = orig
                    break
        if user_col is None:
            continue

        label_col = None
        for k in LABEL_COLUMN_KEYS:
            if k in cmap:
                label_col = cmap[k]
                break

        feature_cols = []
        for c in df.columns:
            if c == user_col or c == label_col:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                feature_cols.append(c)
            else:
                coerced = pd.to_numeric(df[c], errors="coerce")
                if coerced.notna().sum() >= max(3, int(0.85 * len(df))):
                    df[c] = coerced
                    feature_cols.append(c)

        if len(feature_cols) < 1:
            continue

        cols_use = [user_col] + feature_cols + ([label_col] if label_col else [])
        sub = df[cols_use].copy()
        sub[user_col] = sub[user_col].astype(str).str.strip()

        if label_col:
            work = sub.copy()
            work["_yl"] = _labels_from_series(work[label_col])
            gfeat = work.groupby(user_col, as_index=False)[feature_cols].mean()
            gy = work.groupby(user_col)["_yl"].max().reset_index()
            gy.columns = [user_col, "_ym"]
            merged = gfeat.merge(gy, on=user_col)
            y = merged["_ym"].astype(np.int32).values
            sub = merged.drop(columns=["_ym"])
        else:
            sub = sub.groupby(user_col, as_index=False)[feature_cols].mean()
            mal_users = load_malicious_users(data_dir)
            users = sub[user_col].astype(str).values
            if mal_users:
                y = np.array([1 if u in mal_users else 0 for u in users], dtype=np.int32)
            else:
                num = sub[feature_cols].astype(float)
                score = num.sum(axis=1).values
                thresh = np.percentile(score, 92)
                y = (score >= thresh).astype(np.int32)
                warnings.append(
                    f"No label column in {path.name} and no insider answer file; "
                    "generated labels from aggregate numeric features."
                )

        out = sub.rename(columns={user_col: "user_id"})
        score = len(feature_cols) * 1000 + len(df)
        if best is None or score > best[3]:
            w = list(warnings)
            if len(df) < 50:
                w.append(
                    f"Tabular CSV {path.name} has fewer than 50 rows; metrics may be unreliable."
                )
            if label_col:
                w.append(f"Loaded tabular features from {path.name} using column '{label_col}' as label.")
            else:
                w.append(f"Loaded tabular features from {path.name} (no label column).")
            best = (out, y, w, score)

    if best is None:
        return None
    feat, y, w, _ = best
    return feat, y, w


def _csv_looks_like_feature_matrix(path: Path) -> bool:
    """Wide numeric tables (e.g. insider_dataset.csv) are not insider-ID list files."""
    try:
        df = pd.read_csv(path, nrows=30, low_memory=False)
    except Exception:
        return False
    if df.shape[1] < 3:
        return False
    num = sum(1 for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))
    return num >= 2


def load_malicious_users(data_dir: str | Path) -> set[str]:
    """Best-effort load of insider user ids from common answer file names."""
    data_dir = Path(data_dir)
    candidates = list(data_dir.glob("**/insider*.csv")) + list(
        data_dir.glob("**/malicious*.csv")
    )
    users: set[str] = set()
    for path in candidates:
        if _csv_looks_like_feature_matrix(path):
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
            for c in df.columns:
                cl = c.lower()
                if "user" in cl or "insider" in cl or "employee" in cl:
                    users.update(df[c].dropna().astype(str).unique())
                    break
        except Exception:
            continue
    return users

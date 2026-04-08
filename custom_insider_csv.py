"""
Custom insider-threat event CSV: timestamp, actions, volumes, probabilistic or provided labels.

Expected columns (aliases supported): user_id, timestamp, action, file_accessed, device,
location, data_volume_mb, login_status, failed_attempts, privilege_level, label (optional).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MIN_ROWS_CUSTOM = 50
ALLOWED_LOCATION_NORMALIZED = {"bangalore"}
CAT_COLS = ["action", "device", "location"]


def _norm_key(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _build_col_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {_norm_key(c): c for c in df.columns}


def _pick_col(lookup: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in lookup:
            return lookup[c]
    return None


def is_custom_insider_event_schema(df: pd.DataFrame) -> bool:
    """
    Event-level insider CSV: user + time + at least one of volume/failures/location/privilege.
    Avoids matching bare CERT logon.csv (user + date only).
    """
    if df is None or df.empty:
        return False
    lu = _build_col_lookup(df)
    user = _pick_col(lu, ("user_id", "userid", "user"))
    ts = _pick_col(lu, ("timestamp", "datetime", "date", "time", "event_time"))
    if user is None or ts is None:
        return False
    has_ctx = (
        _pick_col(lu, ("data_volume_mb", "data_volume", "volume_mb", "volume", "bytes_mb"))
        is not None
        or _pick_col(lu, ("failed_attempts", "failures", "failed_login", "failed_logins"))
        is not None
        or _pick_col(lu, ("location", "city", "geo", "site")) is not None
        or _pick_col(lu, ("privilege_level", "privilege", "role", "access_level")) is not None
    )
    return bool(has_ctx)


def _series_or_default(df: pd.DataFrame, col: str | None, default, n: int) -> pd.Series:
    if col and col in df.columns:
        return df[col]
    return pd.Series([default] * n, index=df.index)


def load_custom_insider_from_directory(
    data_dir: str | Path,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any], list[str]] | None:
    """
    Load first CSV in folder matching event schema. Returns None if no match.

    Returns (feature_df with user_id + engineered columns, y, meta, warnings).
    meta includes: label_source, n_rows, categorical_cols, preview_records
    """
    data_dir = Path(data_dir)
    warnings: list[str] = []
    candidates = sorted(data_dir.glob("*.csv"))
    best: tuple[pd.DataFrame, int] | None = None

    for path in candidates:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if not is_custom_insider_event_schema(df):
            continue
        if len(df) > (best[1] if best else -1):
            best = (df, len(df))

    if best is None:
        return None

    df = best[0].copy()
    if len(df) < MIN_ROWS_CUSTOM:
        warnings.append(
            f"Custom insider CSV has only {len(df)} rows; recommend at least {MIN_ROWS_CUSTOM} for reliable training."
        )

    out_df, y, meta = engineer_custom_insider_dataframe(df, seed=seed)
    meta["source_file_rows"] = int(len(df))
    preview = out_df.head(15)
    meta["preview_records"] = preview.to_dict(orient="records")
    meta["preview_columns"] = list(preview.columns)
    return out_df, y, meta, warnings


def engineer_custom_insider_dataframe(
    df: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    lu = _build_col_lookup(df)
    n = len(df)
    rng = np.random.default_rng(seed)

    user_c = _pick_col(lu, ("user_id", "userid", "user"))
    ts_c = _pick_col(lu, ("timestamp", "datetime", "date", "time", "event_time"))
    assert user_c is not None and ts_c is not None

    action_c = _pick_col(lu, ("action", "activity", "event_type"))
    file_c = _pick_col(lu, ("file_accessed", "file", "resource", "path"))
    dev_c = _pick_col(lu, ("device", "host", "pc", "computer"))
    loc_c = _pick_col(lu, ("location", "geo", "city", "site"))
    vol_c = _pick_col(lu, ("data_volume_mb", "data_volume", "volume_mb", "volume", "bytes_mb"))
    login_c = _pick_col(lu, ("login_status", "login", "auth_status"))
    fail_c = _pick_col(lu, ("failed_attempts", "failures", "failed_login", "failed_logins"))
    priv_c = _pick_col(lu, ("privilege_level", "privilege", "role", "access_level"))
    label_c = _pick_col(lu, ("label", "target", "y", "is_insider", "insider", "threat"))

    ts = pd.to_datetime(df[ts_c], errors="coerce")
    hour = ts.dt.hour.fillna(12).astype(np.int32)
    is_night = (hour < 6).astype(np.int32)

    action = _series_or_default(df, action_c, "unknown", n).astype(str).fillna("unknown")
    device = _series_or_default(df, dev_c, "unknown", n).astype(str).fillna("unknown")
    location = _series_or_default(df, loc_c, "Bangalore", n).astype(str).fillna("Bangalore")

    data_volume = _series_or_default(df, vol_c, 0.0, n)
    data_volume = pd.to_numeric(data_volume, errors="coerce").fillna(0.0).astype(np.float64)

    failed = _series_or_default(df, fail_c, 0, n)
    failed = pd.to_numeric(failed, errors="coerce").fillna(0).astype(np.float64)

    priv_raw = _series_or_default(df, priv_c, "user", n).astype(str).str.lower().str.strip()
    privilege_level = priv_raw.map(lambda x: 1 if x == "admin" else 0).astype(np.int32)

    high_data_transfer = (data_volume > 500).astype(np.int32)

    uid = df[user_c].astype(str).str.strip()
    login_frequency = uid.groupby(uid).transform("count").astype(np.float64)
    session_duration = rng.integers(1, 121, size=n).astype(np.float64)
    file_sensitivity = rng.integers(0, 3, size=n).astype(np.float64)

    work = pd.DataFrame(
        {
            "user_id": uid,
            "action": action,
            "device": device,
            "location": location,
            "data_volume_mb": data_volume,
            "failed_attempts": failed,
            "privilege_level": privilege_level.astype(np.float64),
            "hour": hour.astype(np.float64),
            "is_night": is_night.astype(np.float64),
            "high_data_transfer": high_data_transfer.astype(np.float64),
            "session_duration": session_duration,
            "login_frequency": login_frequency,
            "file_sensitivity": file_sensitivity,
        },
        index=df.index,
    )

    if label_c and label_c in df.columns:
        y = _labels_to_binary(df[label_c])
        label_source = "provided"
    else:
        loc_norm = location.str.strip().str.lower()
        risky_loc = ~loc_norm.isin(ALLOWED_LOCATION_NORMALIZED)
        score = np.zeros(n, dtype=np.float64)
        score += (failed > 3).astype(np.float64) * 0.4
        score += (data_volume > 500).astype(np.float64) * 0.3
        score += risky_loc.astype(np.float64) * 0.2
        score += (privilege_level == 1).astype(np.float64) * 0.1
        noise = rng.uniform(-0.2, 0.2, size=n)
        y = ((score + noise) > 0.5).astype(np.int32)
        label_source = "probabilistic_rules"

    meta: dict[str, Any] = {
        "label_source": label_source,
        "n_rows": int(len(work)),
        "categorical_cols": list(CAT_COLS),
        "feature_columns": [c for c in work.columns if c != "user_id"],
    }

    return work, y, meta


def _labels_to_binary(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce").fillna(0).astype(np.int32)
        return np.clip(v, 0, 1).values
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

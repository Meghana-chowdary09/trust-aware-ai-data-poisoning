"""Feature matrix construction, labels, and poisoning simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

_LEAK_COL_NAMES = frozenset(
    {"label", "target", "y", "is_insider", "insider", "threat", "is_threat"}
)


def _norm_feat_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def numeric_feature_columns(df: pd.DataFrame, user_col: str = "user_id") -> list[str]:
    """Numeric columns only; excludes user_id and common label-like column names (leakage guard)."""
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    out: list[str] = []
    for c in num:
        if c == user_col:
            continue
        if _norm_feat_name(c) in _LEAK_COL_NAMES:
            continue
        out.append(c)
    return out


@dataclass
class UserDataset:
    """Per-user supervised dataset for XGBoost + alignment arrays for trust."""

    X: np.ndarray
    y: np.ndarray
    user_ids: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler


def synthetic_cert_like(
    n_users: int = 800,
    n_features: int = 12,
    malicious_rate: float = 0.12,
    random_state: int = 42,
    realistic_messy: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Synthetic user-level data mimicking aggregated CERT-style behavior features.
    Malicious users have shifted distributions (more activity / after-hours pattern).
    With realistic_messy=True, class overlap is higher (weaker shift + global noise).
    """
    rng = np.random.default_rng(random_state)
    n_mal = max(1, int(n_users * malicious_rate))
    y = np.zeros(n_users, dtype=np.int32)
    y[:n_mal] = 1
    rng.shuffle(y)

    X = rng.normal(0, 1, size=(n_users, n_features))
    mal_mask = y == 1
    if realistic_messy:
        # Overlap + measurable lift; with ~5–10% label flips, metrics stay in a publication-realistic band.
        X += rng.normal(0, 0.12, size=X.shape)
        X[mal_mask] += rng.normal(1.02, 0.28, size=(mal_mask.sum(), n_features))
        X[mal_mask, :4] += 0.62
    else:
        X[mal_mask] += rng.normal(1.2, 0.35, size=(mal_mask.sum(), n_features))
        X[mal_mask, :4] += 0.8

    names = [f"f{i}" for i in range(n_features)]
    users = np.array([f"U{i:05d}" for i in range(n_users)])
    df = pd.DataFrame(X, columns=names)
    df.insert(0, "user_id", users)
    return df, y, users


def dataframe_to_xy(
    df: pd.DataFrame,
    y: np.ndarray,
    user_col: str = "user_id",
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[UserDataset, UserDataset]:
    """Scale features and split by users (rows = users)."""
    users = df[user_col].astype(str).values
    feat_cols = numeric_feature_columns(df, user_col=user_col)
    X_raw = df[feat_cols].values.astype(np.float64)
    y = np.asarray(y).astype(np.int32)

    idx = np.arange(len(y))
    strat = y if len(np.unique(y)) > 1 else None
    tr_idx, te_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=strat
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_raw[tr_idx])
    X_te = scaler.transform(X_raw[te_idx])

    train_ds = UserDataset(
        X=X_tr,
        y=y[tr_idx],
        user_ids=users[tr_idx],
        feature_names=feat_cols,
        scaler=scaler,
    )
    test_ds = UserDataset(
        X=X_te,
        y=y[te_idx],
        user_ids=users[te_idx],
        feature_names=feat_cols,
        scaler=scaler,
    )
    return train_ds, test_ds


def dataframe_to_xy_with_ordinal_cats(
    df: pd.DataFrame,
    y: np.ndarray,
    categorical_cols: list[str],
    user_col: str = "user_id",
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[UserDataset, UserDataset]:
    """
    Split first, then OrdinalEncoder (unknown=-1) on categoricals fit on train only;
    numeric columns as float; StandardScaler on full feature block.
    """
    y = np.asarray(y).astype(np.int32)
    users = df[user_col].astype(str).values
    feat_df = df.drop(columns=[user_col]).copy()

    cats_present = [c for c in categorical_cols if c in feat_df.columns]
    allowed_num = set(numeric_feature_columns(df, user_col=user_col))
    num_cols = [c for c in feat_df.columns if c in allowed_num and c not in cats_present]

    for c in num_cols:
        feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce")
    X_num = feat_df[num_cols].fillna(0).astype(np.float64)

    idx = np.arange(len(y))
    strat = y if len(np.unique(y)) > 1 else None
    tr_idx, te_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=strat
    )

    if cats_present:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_block = feat_df[cats_present].astype(str).fillna("nan")
        oe.fit(cat_block.iloc[tr_idx])
        X_cat_tr = oe.transform(cat_block.iloc[tr_idx])
        X_cat_te = oe.transform(cat_block.iloc[te_idx])
        X_tr_raw = np.hstack([X_num.iloc[tr_idx].values, X_cat_tr.astype(np.float64)])
        X_te_raw = np.hstack([X_num.iloc[te_idx].values, X_cat_te.astype(np.float64)])
        feature_names = num_cols + [f"{c}_enc" for c in cats_present]
    else:
        X_tr_raw = X_num.iloc[tr_idx].values.astype(np.float64)
        X_te_raw = X_num.iloc[te_idx].values.astype(np.float64)
        feature_names = num_cols

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    train_ds = UserDataset(
        X=X_tr,
        y=y[tr_idx],
        user_ids=users[tr_idx],
        feature_names=feature_names,
        scaler=scaler,
    )
    test_ds = UserDataset(
        X=X_te,
        y=y[te_idx],
        user_ids=users[te_idx],
        feature_names=feature_names,
        scaler=scaler,
    )
    return train_ds, test_ds


def apply_label_poisoning(
    y: np.ndarray,
    user_ids: np.ndarray,
    flip_fraction: float = 0.12,
    target_benign_only: bool = True,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flip labels for a subset of samples (simulates insider poisoning the training set).
    Returns (y_poisoned, poison_mask).
    """
    rng = np.random.default_rng(random_state)
    y2 = y.copy()
    mask = np.zeros(len(y), dtype=bool)
    eligible = np.arange(len(y))
    if target_benign_only:
        eligible = eligible[y == 0]
    n_flip = max(1, int(len(eligible) * flip_fraction))
    pick = rng.choice(eligible, size=n_flip, replace=False)
    y2[pick] = 1 - y2[pick]
    mask[pick] = True
    return y2, mask


def apply_feature_poisoning(
    X: np.ndarray,
    y: np.ndarray,
    user_ids: np.ndarray,
    malicious_noise_scale: float = 0.35,
    fraction_malicious_rows: float = 0.6,
    random_state: int = 1,
) -> np.ndarray:
    """
    Add noise to features for a subset of malicious users' rows so they look more benign
    (attackers try to evade detection by scrubbing their trail in exported training data).
    """
    rng = np.random.default_rng(random_state)
    Xp = X.copy()
    mal = np.where(y == 1)[0]
    if len(mal) == 0:
        return Xp
    n_tamper = max(1, int(len(mal) * fraction_malicious_rows))
    tamper_idx = rng.choice(mal, size=n_tamper, replace=False)
    noise = rng.normal(0, malicious_noise_scale, size=Xp[tamper_idx].shape)
    # Pull malicious toward global mean (0 after scaling) on a random feature subset
    cols = rng.choice(X.shape[1], size=max(1, X.shape[1] // 2), replace=False)
    Xp[tamper_idx[:, None], cols] *= 0.3
    Xp[tamper_idx] += noise
    return Xp

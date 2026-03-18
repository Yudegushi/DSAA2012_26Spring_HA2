from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2


def bootstrap_accuracy_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 3407,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    boot = np.empty(n_boot, dtype=np.float64)
    correct = (y_true == y_pred).astype(np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(correct[idx].mean())
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    acc = float(correct.mean())
    return {
        "acc": acc,
        "ci_low": lo,
        "ci_high": hi,
        "ci_half_width": (hi - lo) / 2.0,
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return {
            "b": b,
            "c": c,
            "chi2": 0.0,
            "p_value": 1.0,
            "n_discordant": 0,
        }

    chi2_stat = (abs(b - c) - 1.0) ** 2 / n
    p = float(1.0 - chi2.cdf(chi2_stat, df=1))
    return {
        "b": b,
        "c": c,
        "chi2": float(chi2_stat),
        "p_value": p,
        "n_discordant": int(n),
    }


def top_confusions(y_true: np.ndarray, y_pred: np.ndarray, top_k: int = 10) -> pd.DataFrame:
    mask = y_true != y_pred
    if not np.any(mask):
        return pd.DataFrame(columns=["gt", "pred", "count"])
    df = pd.DataFrame({"gt": y_true[mask], "pred": y_pred[mask]})
    out = (
        df.groupby(["gt", "pred"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    return out


def build_paired_frame(pred_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    s = pred_df[(pred_df["model"] == model_name) & (pred_df["exp"] == "B1_simple")].copy()
    e = pred_df[(pred_df["model"] == model_name) & (pred_df["exp"] == "M1_ensemble_all")].copy()

    s = s[["sample_index", "label_id", "pred_id", "conf"]].rename(
        columns={"pred_id": "pred_simple", "conf": "conf_simple"}
    )
    e = e[["sample_index", "label_id", "pred_id", "conf"]].rename(
        columns={"pred_id": "pred_ens", "conf": "conf_ens"}
    )

    merged = s.merge(e, on=["sample_index", "label_id"], how="inner")
    merged["simple_correct"] = merged["pred_simple"] == merged["label_id"]
    merged["ens_correct"] = merged["pred_ens"] == merged["label_id"]
    merged["improved"] = (~merged["simple_correct"]) & (merged["ens_correct"])
    merged["degraded"] = (merged["simple_correct"]) & (~merged["ens_correct"])
    merged["both_wrong"] = (~merged["simple_correct"]) & (~merged["ens_correct"])
    return merged


def select_case_indices(
    paired_df: pd.DataFrame,
    n_improved: int = 12,
    n_failed: int = 12,
    seed: int = 2026,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    improved_pool = paired_df.loc[paired_df["improved"], "sample_index"].to_numpy(dtype=np.int64)
    failed_pool = paired_df.loc[paired_df["both_wrong"], "sample_index"].to_numpy(dtype=np.int64)

    if improved_pool.size > n_improved:
        improved_idx = np.sort(rng.choice(improved_pool, size=n_improved, replace=False))
    else:
        improved_idx = np.sort(improved_pool)

    if failed_pool.size > n_failed:
        failed_idx = np.sort(rng.choice(failed_pool, size=n_failed, replace=False))
    else:
        failed_idx = np.sort(failed_pool)

    return improved_idx, failed_idx

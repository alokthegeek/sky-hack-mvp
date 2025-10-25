"""Scoring and export helpers for calibrated probabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .utils import ensure_directory, get_logger

LOGGER = get_logger("score")

_REQUIRED_COLUMNS: Sequence[str] = (
    "company_id",
    "flight_number",
    "scheduled_departure_station_code",
    "scheduled_arrival_station_code",
    "dep_date",
    "dep_hour",
    "dow",
    "sched_turn_slack",
    "turn_sched",
    "turn_min",
    "transfer_bag_ratio",
    "load_factor",
    "ssr_rate",
    "pct_children",
    "pct_basic_econ",
    "stroller_rate",
    "is_international",
    "dep_delay_min",
    "is_high_delay",
)


def _validate_quantiles(quantiles: Sequence[float]) -> tuple[float, float]:
    if len(quantiles) != 2:
        msg = "class_quantiles must contain exactly two values"
        raise ValueError(msg)
    low, high = sorted(float(q) for q in quantiles)
    if not (0 < low < high < 1):
        msg = f"Invalid class quantiles {quantiles}; expect 0 < q1 < q2 < 1"
        raise ValueError(msg)
    return low, high


def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        msg = f"Missing columns required for export: {missing}"
        raise ValueError(msg)


def _compute_daily_z(group: pd.DataFrame) -> pd.Series:
    mean = group.mean()
    std = group.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=group.index)
    return (group - mean) / std


import numpy as np
import pandas as pd


import numpy as np
import pandas as pd


def score_and_rank(df: pd.DataFrame,
                   probs: np.ndarray,
                   class_quantiles=(0.2, 0.7)) -> pd.DataFrame:
    """Add fds_raw, fds_z, daily_rank, daily_class and return a view with all
    columns required by validate_export. Ties: fds_raw desc, sched_turn_slack asc,
    transfer_bag_ratio desc. Classification via per-day quantiles (vectorized).
    """
    assert "dep_date" in df.columns, "dep_date missing in feature table"

    scored = df.copy()

    scored["fds_raw"] = pd.Series(probs, index=scored.index).astype(float)

    g = scored.groupby("dep_date")["fds_raw"]
    mu = g.transform("mean")
    sd = g.transform("std").replace({0.0: np.nan})
    scored["fds_z"] = ((scored["fds_raw"] - mu) / sd).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    if "sched_turn_slack" not in scored.columns:
        scored["sched_turn_slack"] = np.nan
    if "transfer_bag_ratio" not in scored.columns:
        scored["transfer_bag_ratio"] = np.nan

    scored = scored.sort_values(
        by=["dep_date", "fds_raw", "sched_turn_slack", "transfer_bag_ratio"],
        ascending=[True, False, True, False],
        kind="mergesort",
    )

    scored["daily_rank"] = scored.groupby("dep_date").cumcount() + 1

    lo_q, hi_q = class_quantiles
    p = scored["fds_raw"].astype(float)

    q_lo = float(p.quantile(lo_q))
    q_hi = float(p.quantile(hi_q))

    if q_lo == q_hi:
        scored["daily_class"] = "Medium"
    else:
        scored["daily_class"] = np.where(
            p <= q_lo, "Difficult",
            np.where(p > q_hi, "Easy", "Medium")
        )

    share = scored["daily_class"].value_counts(normalize=True).reindex(["Difficult", "Medium", "Easy"]).fillna(0)
    print(
        "Class shares — Difficult: %.2f%%, Medium: %.2f%%, Easy: %.2f%%"
        % (share.get("Difficult", 0) * 100, share.get("Medium", 0) * 100, share.get("Easy", 0) * 100)
    )

    desired = [
        "company_id", "flight_number",
        "scheduled_departure_station_code", "scheduled_arrival_station_code",
        "dep_date", "dep_hour", "dow",
        "turn_sched", "turn_min", "sched_turn_slack",
        "transfer_bag_ratio", "load_factor",
        "ssr_rate", "pct_children", "pct_basic_econ", "stroller_rate",
        "dep_delay_min", "is_high_delay", "is_international",
        "fds_raw", "fds_z", "daily_rank", "daily_class",
    ]
    export_cols = [c for c in desired if c in scored.columns]

    return scored.loc[:, export_cols].sort_values(["dep_date", "daily_rank"], kind="mergesort")


def export_results(df: pd.DataFrame, your_name: str, out_dir: Path | str) -> Path:
    """Persist the ranked dataframe to disk with deterministic ordering."""
    _ensure_columns(df, list(_REQUIRED_COLUMNS) + ["fds_raw", "fds_z", "daily_rank", "daily_class"])

    safe_name = your_name.strip().lower().replace(" ", "_") or "output"
    sorted_df = df.sort_values(
        by=[
            "dep_date",
            "daily_rank",
            "fds_raw",
            "sched_turn_slack",
            "transfer_bag_ratio",
        ],
        ascending=[True, True, False, True, False],
    )

    output_path = Path(out_dir) / f"test_{safe_name}.csv"
    ensure_directory(output_path.parent)
    sorted_df.to_csv(output_path, index=False)
    LOGGER.info("Scores exported to %s", output_path)
    return output_path

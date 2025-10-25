"""Shared utility helpers for the fds_ord project."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

LOGGER_NAME = "fds_ord"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure global logging once."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced logger anchored at LOGGER_NAME."""
    resolved_name = f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME
    return logging.getLogger(resolved_name)


def ensure_directory(path: str | Path) -> Path:
    """Create the directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_path(*parts: str | Path) -> Path:
    """Return a resolved path relative to the project root."""
    return Path(__file__).resolve().parent.parent.joinpath(*parts).resolve()


def sanitize_name(raw: str) -> str:
    """Return a filesystem-safe, lowercase identifier for the provided string."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower())
    return cleaned.strip("_") or "output"


@lru_cache(maxsize=1)
def have_pyarrow() -> bool:
    """Return whether pyarrow is importable."""
    try:
        import pyarrow.csv  # noqa: F401
    except ImportError:
        return False
    return True


REQUIRED_EXPORT_COLUMNS: Sequence[str] = (
    "company_id",
    "flight_number",
    "scheduled_departure_station_code",
    "scheduled_arrival_station_code",
    "dep_date",
    "dep_hour",
    "dow",
    "turn_sched",
    "turn_min",
    "sched_turn_slack",
    "transfer_bag_ratio",
    "load_factor",
    "ssr_rate",
    "pct_children",
    "pct_basic_econ",
    "stroller_rate",
    "is_international",
    "dep_delay_min",
    "is_high_delay",
    "fds_raw",
    "fds_z",
    "daily_rank",
    "daily_class",
)


def _validate_quantiles(quantiles: Sequence[float]) -> tuple[float, float]:
    if len(quantiles) != 2:
        raise ValueError("Expected exactly two quantiles")
    low, high = sorted(float(q) for q in quantiles)
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid quantiles {quantiles}; require 0 < low < high < 1")
    return low, high


def validate_export(
    df: pd.DataFrame,
    quantiles: Sequence[float],
    *,
    tolerance: float = 0.05,
    required_columns: Sequence[str] | None = None,
) -> None:
    """Run sanity checks on the scored export DataFrame."""
    if df.empty:
        raise AssertionError("Export is empty")

    required = tuple(required_columns) if required_columns else REQUIRED_EXPORT_COLUMNS
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise AssertionError(f"Export missing required columns: {missing_cols}")

    null_checks = df[list(required)].isna().any()
    if null_checks.any():
        offenders = ", ".join(null_checks[null_checks].index.tolist())
        raise AssertionError(f"Found NaNs in required columns: {offenders}")

    if "dep_date" not in df.columns or "daily_rank" not in df.columns:
        raise AssertionError("Columns 'dep_date' and 'daily_rank' are required for rank checks")

    for dep_date, group in df.groupby("dep_date", dropna=False):
        ranks = np.sort(group["daily_rank"].to_numpy())
        expected = np.arange(1, len(ranks) + 1)
        if not np.array_equal(ranks, expected):
            raise AssertionError(
                f"Daily ranks invalid for {dep_date}: expected {expected.tolist()}, got {ranks.tolist()}"
            )

    low_q, high_q = _validate_quantiles(quantiles)
    total = len(df)
    if total == 0:
        raise AssertionError("Export contains no rows")

    expected_shares = {
        "Difficult": low_q,
        "Medium": high_q - low_q,
        "Easy": 1 - high_q,
    }
    actual_shares = df["daily_class"].value_counts(normalize=True).to_dict()
    for label, expected_share in expected_shares.items():
        actual_share = actual_shares.get(label, 0.0)
        if abs(actual_share - expected_share) > tolerance:
            raise AssertionError(
                f"Class share for {label} out of tolerance: expected ~{expected_share:.2f}, got {actual_share:.2f}"
            )

    slack_violations = df.loc[df["turn_sched"] <= df["turn_min"], "sched_turn_slack"]
    if not slack_violations.empty and (slack_violations > 1e-6).any():
        raise AssertionError("Found turn_sched <= turn_min but sched_turn_slack > 0")


def find_latest_export(out_dir: Path) -> Path | None:
    """Return the most recent test_*.csv in the output directory."""
    out_path = Path(out_dir)
    candidates = sorted(out_path.glob("test_*.csv"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None

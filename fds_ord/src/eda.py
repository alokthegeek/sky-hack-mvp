"""Exploratory data analysis helpers."""

from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .utils import ensure_directory, get_logger

LOGGER = get_logger("eda")


def _as_series(df: pd.DataFrame, column: str, dtype: str = "float") -> pd.Series:
    if column not in df:
        return pd.Series(dtype=dtype)
    series = df[column]
    if dtype == "float":
        return pd.to_numeric(series, errors="coerce")
    if dtype == "bool":
        return series.astype("boolean")
    return series


def _safe_mean(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.mean()) if not clean.empty else float("nan")


def _safe_quantile(series: pd.Series, q: float) -> float:
    clean = series.dropna()
    return float(clean.quantile(q)) if not clean.empty else float("nan")


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    clean = pd.concat([x, y], axis=1).dropna()
    if clean.empty:
        return float("nan")
    return float(clean.iloc[:, 0].corr(clean.iloc[:, 1]))


def _fit_logistic(df: pd.DataFrame) -> Dict[str, float]:
    required = ["is_high_delay", "ssr_rate", "load_factor"]
    if any(col not in df for col in required):
        LOGGER.warning("Missing columns for logistic regression; skipping")
        return {"coef": float("nan"), "std_err": float("nan"), "p_value": float("nan")}

    subset = df[required].dropna()
    if subset.empty or subset["is_high_delay"].nunique() < 2:
        LOGGER.warning("Insufficient data for logistic regression")
        return {"coef": float("nan"), "std_err": float("nan"), "p_value": float("nan")}

    X = subset[["ssr_rate", "load_factor"]].to_numpy(dtype=float)
    y = subset["is_high_delay"].astype(int).to_numpy()

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    W = probs * (1 - probs)
    X_design = np.column_stack([np.ones(len(X)), X])
    fisher = X_design.T @ (X_design * W[:, None])
    try:
        cov = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        LOGGER.warning("Singular Fisher information; std errors unavailable")
        return {"coef": float(model.coef_[0, 0]), "std_err": float("nan"), "p_value": float("nan")}

    coef_ssr = float(model.coef_[0, 0])
    std_err = float(np.sqrt(cov[1, 1]))
    if np.isnan(std_err) or std_err == 0:
        p_value = float("nan")
    else:
        z_score = coef_ssr / std_err
        p_value = float(erfc(abs(z_score) / sqrt(2)))

    return {"coef": coef_ssr, "std_err": std_err, "p_value": p_value}


def compute_eda_metrics(df: pd.DataFrame) -> Dict[str, object]:
    metrics: Dict[str, object] = {}

    dep_delay = _as_series(df, "dep_delay_min")
    is_high_delay = _as_series(df, "is_high_delay", dtype="float")

    metrics["avg_dep_delay_min"] = _safe_mean(dep_delay)
    metrics["pct_high_delay"] = _safe_mean(is_high_delay)

    if {"turn_sched", "turn_min"}.issubset(df.columns):
        turns = df[["turn_sched", "turn_min"]].dropna()
        metrics["tight_turns"] = int((turns["turn_sched"] <= turns["turn_min"]).sum())
    else:
        metrics["tight_turns"] = 0

    metrics["avg_transfer_bag_ratio"] = _safe_mean(_as_series(df, "transfer_bag_ratio"))

    load_factor = _as_series(df, "load_factor")
    metrics["load_summary"] = {
        "mean": _safe_mean(load_factor),
        "median": _safe_quantile(load_factor, 0.5),
        "p95": _safe_quantile(load_factor, 0.95),
        "corr_delay": _safe_corr(load_factor, dep_delay),
        "corr_high_delay": _safe_corr(load_factor, is_high_delay),
    }

    metrics["logistic_ssr_effect"] = _fit_logistic(df)

    return metrics


def make_eda_plots(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    ensure_directory(out_dir)
    paths: List[Path] = []

    if df.empty:
        LOGGER.warning("EDA plots skipped: empty DataFrame")
        return paths

    if "dep_delay_min" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df["dep_delay_min"].dropna(), bins=40)
        ax.axvline(15, linestyle="--", linewidth=1.5)
        ax.set_title("Distribution of Departure Delay (minutes)")
        ax.set_xlabel("Departure delay (min)")
        ax.set_ylabel("Flights")
        fig.tight_layout()
        path = Path(out_dir) / "dep_delay_hist.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)

    if {"dep_date", "turn_sched", "turn_min"}.issubset(df.columns):
        grouped = df.assign(dep_date=pd.to_datetime(df["dep_date"]))
        grouped["tight_turn"] = grouped["turn_sched"] <= grouped["turn_min"]
        daily = grouped.groupby("dep_date")["tight_turn"].sum()
        if not daily.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            daily.plot(kind="bar", ax=ax)
            ax.set_title("Flights with Scheduled Turn <= Minimum")
            ax.set_xlabel("Departure date")
            ax.set_ylabel("Count of flights")
            fig.autofmt_xdate(rotation=45)
            fig.tight_layout()
            path = Path(out_dir) / "tight_turns_by_day.png"
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)

    if {"transfer_bag_ratio", "is_high_delay"}.issubset(df.columns):
        data = [
            df.loc[df["is_high_delay"] == 0, "transfer_bag_ratio"].dropna(),
            df.loc[df["is_high_delay"] == 1, "transfer_bag_ratio"].dropna(),
        ]
        if any(len(part) for part in data):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.violinplot(data, showmeans=True, showmedians=False)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["On time", "High delay"])
            ax.set_ylabel("Transfer bag ratio")
            ax.set_title("Transfer Bag Ratio by Delay Class")
            fig.tight_layout()
            path = Path(out_dir) / "transfer_bag_ratio_violin.png"
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)

    if {"load_factor", "dep_delay_min"}.issubset(df.columns):
        clean = df[["load_factor", "dep_delay_min"]].dropna()
        if not clean.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(clean["load_factor"], clean["dep_delay_min"], alpha=0.2)
            ax.set_title("Load Factor vs Departure Delay")
            ax.set_xlabel("Load factor")
            ax.set_ylabel("Departure delay (min)")
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess

                if len(clean) > 10:
                    lowess_fit = lowess(clean["dep_delay_min"], clean["load_factor"], frac=0.2)
                    ax.plot(lowess_fit[:, 0], lowess_fit[:, 1], linewidth=2)
            except ImportError:
                LOGGER.debug("statsmodels not available; skipping LOWESS overlay")
            fig.tight_layout()
            path = Path(out_dir) / "load_factor_vs_delay.png"
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)

    return paths


def write_markdown_summary(metrics: Dict[str, object], out_path: Path) -> Path:
    ensure_directory(out_path.parent)

    mean_delay = metrics.get("avg_dep_delay_min", float("nan"))
    pct_delay = metrics.get("pct_high_delay", float("nan"))
    turn_count = metrics.get("tight_turns", 0)
    transfer_ratio = metrics.get("avg_transfer_bag_ratio", float("nan"))
    load_summary = metrics.get("load_summary", {})
    logistic = metrics.get("logistic_ssr_effect", {})

    lines = [
        "# EDA Summary\n",
        f"- Average departure delay: {mean_delay:.2f} min; share high delay: {pct_delay:.2%}",
        f"- Flights with scheduled turn <= minimum: {turn_count}",
        f"- Average transfer bag ratio: {transfer_ratio:.2%}",
        (
            "- Load factor mean/median/p95: "
            f"{load_summary.get('mean', float('nan')):.2f} / "
            f"{load_summary.get('median', float('nan')):.2f} / "
            f"{load_summary.get('p95', float('nan')):.2f}; "
            f"corr w/ delay: {load_summary.get('corr_delay', float('nan')):.2f}; "
            f"corr w/ high delay: {load_summary.get('corr_high_delay', float('nan')):.2f}"
        ),
        (
            "- Logistic regression (is_high_delay ~ ssr_rate + load_factor): "
            f"ssr_rate coef={logistic.get('coef', float('nan')):.3f}, "
            f"std err={logistic.get('std_err', float('nan')):.3f}, "
            f"p-value={logistic.get('p_value', float('nan')):.3f}"
        ),
    ]

    content = "\n".join(lines) + "\n"
    out_path.write_text(content)
    LOGGER.info("EDA summary written to %s", out_path)
    return out_path

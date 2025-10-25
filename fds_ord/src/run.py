"""Command-line entrypoint stitching together the fds_ord workflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from .eda import compute_eda_metrics, make_eda_plots, write_markdown_summary
from .features import build_feature_table
from .load import load_dataset_bundle
from .model import predict_proba, report_feature_importance, train_calibrated_model
from .score import export_results, score_and_rank
from .utils import (
    ensure_directory,
    find_latest_export,
    get_logger,
    sanitize_name,
    setup_logging,
    validate_export,
)

LOGGER = get_logger("run")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fds_ord end-to-end pipeline")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing CSV inputs")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory for outputs")
    parser.add_argument("--your_name", type=str, required=True, help="Your name for tagging outputs")
    parser.add_argument(
        "--class_quantiles",
        nargs=2,
        type=float,
        default=(0.2, 0.7),
        metavar=("LOW", "HIGH"),
        help="Two quantile thresholds for difficulty bands",
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed for modelling")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Skip training and validate the latest exported test_*.csv",
    )
    return parser.parse_args(argv)




EXPECTED_DATA_COLUMNS = {
    "flight_level": [
        "scheduled_departure_datetime_local",
        "actual_departure_datetime_local",
        "scheduled_departure_station_code",
        "scheduled_arrival_station_code",
        "scheduled_ground_time_minutes",
        "minimum_turn_minutes",
        "total_seats",
    ],
    "pnr_flight_level": [
        "basic_economy_flag",
        "basic_economy_pax",
        "is_child",
        "child_flag",
        "lap_child",
        "lap_child_flag",
        "stroller_flag",
        "stroller_indicator",
    ],
    "pnr_remark_level": [],
    "bag_level": [
        "bag_count",
        "bags",
        "transfer_bag_count",
        "transfer_bags",
    ],
    "airports": [
        "station_code",
        "iso_country_code",
        "country_iso_code",
    ],
}

def _collect_column_notes(frames: dict[str, pd.DataFrame]) -> list[str]:
    notes: list[str] = []
    for key, columns in EXPECTED_DATA_COLUMNS.items():
        if key not in frames:
            notes.append(f"{key} file missing; related features defaulted to zero or False")
            continue
        frame = frames[key]
        missing = [col for col in columns if col not in frame.columns]
        if missing:
            notes.append(f"{key} missing columns: {', '.join(sorted(missing))} (defaults applied)")
    return notes

def _fmt(value: float | int | None, *, percent: bool = False, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    factor = 100 if percent else 1
    suffix = "%" if percent else ""
    return f"{value * factor:.{digits}f}{suffix}"

def _print_eda_answers(metrics: dict[str, object]) -> None:
    load = metrics.get("load_summary", {}) or {}
    logistic = metrics.get("logistic_ssr_effect", {}) or {}
    print("EDA quick answers:")
    print(f"- Avg dep delay: {_fmt(metrics.get('avg_dep_delay_min'))} min; high-delay share: {_fmt(metrics.get('pct_high_delay'), percent=True)}")
    print(f"- Flights with turn_sched <= turn_min: {int(metrics.get('tight_turns', 0))}")
    print(f"- Avg transfer bag ratio: {_fmt(metrics.get('avg_transfer_bag_ratio'), percent=True)}")
    print(
        "- Load factor mean/median/p95: "
        f"{_fmt(load.get('mean'))} / {_fmt(load.get('median'))} / {_fmt(load.get('p95'))}; "
        f"corr with delay: {_fmt(load.get('corr_delay'))}; corr with high-delay: {_fmt(load.get('corr_high_delay'))}"
    )
    print(
        "- Logistic(ssr_rate | load_factor): "
        f"coef={_fmt(logistic.get('coef'))}, "
        f"std={_fmt(logistic.get('std_err'))}, "
        f"p={_fmt(logistic.get('p_value'))}"
    )
def _load_data(data_dir: Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    try:
        bundle = load_dataset_bundle(data_dir)
        LOGGER.info("All expected datasets loaded successfully")
        return bundle.frames, []
    except FileNotFoundError as exc:
        LOGGER.warning("Standard bundle load failed (%s); attempting partial load", exc)

    loaded_frames: dict[str, pd.DataFrame] = {}
    missing_files: list[str] = []
    for filename, key in _FILE_MAP.items():
        csv_path = data_dir / filename
        if not csv_path.exists():
            LOGGER.warning("Optional input missing: %s", csv_path)
            missing_files.append(filename)
            continue
        try:
            loaded_frames[key] = read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Failed to read %s: %s", csv_path, exc)
            missing_files.append(filename)

    if "flight_level" not in loaded_frames:
        raise FileNotFoundError("Primary flight dataset is required and was not found")

    if missing_files:
        LOGGER.warning("Workflow continuing without: %s", ", ".join(missing_files))
    return loaded_frames, missing_files


def _console_summary(scored: pd.DataFrame) -> None:
    LOGGER.info("Daily class counts:\n%s", scored["daily_class"].value_counts().to_string())
    difficult = scored.loc[scored["daily_class"] == "Difficult"].copy()
    if difficult.empty:
        LOGGER.info("No flights classified as Difficult")
        return

    if "scheduled_departure_datetime_local" in difficult.columns:
        difficult["dep_time"] = pd.to_datetime(difficult["scheduled_departure_datetime_local"])
    else:
        difficult["dep_time"] = pd.to_datetime(difficult["dep_date"]) + pd.to_timedelta(difficult["dep_hour"], unit="h")
    difficult["dep_to_arr"] = difficult["scheduled_departure_station_code"].astype(str) + "->" + difficult["scheduled_arrival_station_code"].astype(str)
    difficult = difficult.sort_values(["dep_date", "daily_rank"])
    subset = difficult.groupby("dep_date").head(5).copy()
    subset.loc[:, "dep_time"] = pd.to_datetime(subset["dep_time"]).dt.strftime("%Y-%m-%d %H:%M")

    columns = [
        "dep_time",
        "flight_number",
        "dep_to_arr",
        "fds_raw",
        "fds_z",
        "daily_rank",
        "sched_turn_slack",
        "transfer_bag_ratio",
        "load_factor",
        "ssr_rate",
    ]
    table = subset[columns].rename(columns={"dep_to_arr": "dep->arr"})
    LOGGER.info("Per-day top 5 Difficult flights:\n%s", table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    ensure_directory(out_dir)

    if args.check_only:
        latest = find_latest_export(out_dir)
        if latest is None:
            LOGGER.error("No exported files found in %s", out_dir)
            print("FAIL")
            return 1
        LOGGER.info("Validating %s", latest.name)
        df = pd.read_csv(latest)
        quantiles = tuple(float(q) for q in args.class_quantiles)
        try:
            validate_export(df, quantiles)
        except (AssertionError, ValueError) as exc:
            LOGGER.error("Validation failed: %s", exc)
            print("FAIL")
            return 1
        LOGGER.info("All checks passed for %s", latest.name)
        print("PASS")
        return 0

    LOGGER.info("Loading input CSVs from %s", data_dir)
    frames, missing_inputs = _load_data(data_dir)
    notes: list[str] = []
    if missing_inputs:
        notes.append(f"Missing CSVs: {', '.join(sorted(missing_inputs))}")
    notes.extend(_collect_column_notes(frames))

    LOGGER.info("Building feature table")
    flights = frames.get("flight_level", pd.DataFrame())
    feature_table = build_feature_table(
        flights,
        frames.get("pnr_flight_level", pd.DataFrame()),
        frames.get("pnr_remark_level", pd.DataFrame()),
        frames.get("bag_level", pd.DataFrame()),
        frames.get("airports", pd.DataFrame()),
    )

    if feature_table.empty:
        LOGGER.error("Feature table is empty; aborting")
        return 1
    LOGGER.info("Feature table prepared with %d rows", len(feature_table))

    LOGGER.info("Running EDA outputs")
    metrics = compute_eda_metrics(feature_table)
    _print_eda_answers(metrics)
    eda_summary_path = write_markdown_summary(metrics, out_dir / "eda_summary.md")
    plot_paths = make_eda_plots(feature_table, out_dir)
    LOGGER.info("EDA summary written to %s", eda_summary_path)
    if plot_paths:
        LOGGER.info("Generated %d EDA plots", len(plot_paths))

    LOGGER.info("Training calibrated model")
    trained = train_calibrated_model(feature_table, seed=args.seed)

    LOGGER.info("Scoring entire dataset")
    probabilities = predict_proba(trained, feature_table)

    LOGGER.info("Ranking flights by risk")
    quantiles = tuple(float(q) for q in args.class_quantiles)
    LOGGER.info("Using class quantiles: %s", quantiles)
    scored = score_and_rank(feature_table, probabilities, class_quantiles=quantiles)
    validate_export(scored, quantiles)

    LOGGER.info("Exporting outputs")
    sanitized_name = sanitize_name(args.your_name)
    export_path = export_results(scored, sanitized_name, out_dir)
    importance_path = report_feature_importance(
        trained.model,
        trained.pipeline,
        trained.feature_names,
        out_dir,
    )
    LOGGER.info("Predictions exported to %s", export_path)
    LOGGER.info("Feature importances saved to %s", importance_path)

    notes_path = out_dir / "notes_assumptions.txt"
    if notes:
        notes_path.write_text("\n".join(sorted(set(notes))) + "\n")
    else:
        notes_path.write_text("No missing inputs detected; all required columns present.\n")
    LOGGER.info("Notes saved to %s", notes_path)

    _console_summary(scored)
    LOGGER.info("Pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Feature engineering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _to_day(series: pd.Series) -> pd.Series:
    """Return a tz-naive datetime64[ns] day key for any date/time-like series."""
    dt = pd.to_datetime(series, errors="coerce")
    try:
        dt = dt.dt.tz_convert(None)
    except Exception:
        try:
            dt = dt.dt.tz_localize(None)
        except Exception:
            pass
    return pd.to_datetime(dt.dt.date)


from .utils import get_logger

LOGGER = get_logger("features")

_KEY_COLUMNS = ["company_id", "flight_number", "flight_day"]
_FLIGHT_DAY_CANDIDATES = [
    "flight_day",
    "scheduled_departure_datetime_local",
    "scheduled_departure_date_local",
    "dep_date",
    "departure_date",
    "flight_date",
]
_INDICATOR_MAP = {
    "y": 1.0,
    "yes": 1.0,
    "true": 1.0,
    "t": 1.0,
    "1": 1.0,
    "n": 0.0,
    "no": 0.0,
    "false": 0.0,
    "f": 0.0,
    "0": 0.0,
}
_PNR_SUM_OPTIONS = {
    "basic_economy_pax": [
        "basic_economy_pax",
        "basic_economy_flag",
        "is_basic_economy",
        "basic_economy_indicator",
    ],
    "is_child_count": ["is_child", "child_flag", "is_child_indicator"],
    "lap_child_count": ["lap_child", "lap_child_flag", "lap_infant"],
    "stroller_users": [
        "stroller_users",
        "stroller_flag",
        "stroller",
        "stroller_indicator",
    ],
}
_BAG_COUNT_OPTIONS = ["bag_count", "bags", "bag_quantity", "bag_total"]
_BAG_TRANSFER_OPTIONS = [
    "bags_transfer",
    "transfer_bag_count",
    "transfer_bags",
    "transfer_indicator",
    "is_transfer",
    "is_transfer_bag",
]
_AIRPORT_CODE_OPTIONS = [
    "station_code",
    "airport_code",
    "iata_code",
    "iata",
    "code",
]
_COUNTRY_CODE_OPTIONS = ["iso_country_code", "country_iso_code", "country_code", "country"]
_SEAT_COLUMNS = [
    "total_seats",
    "seats_total",
    "seat_capacity",
    "seat_count",
    "available_seats",
    "seats_available",
    "capacity",
]
_TURN_SCHED_COLUMNS = [
    "scheduled_ground_time_minutes",
    "scheduled_ground_minutes",
    "scheduled_turn_time_minutes",
    "scheduled_turn_minutes",
]
_TURN_MIN_COLUMNS = [
    "minimum_turn_minutes",
    "minimum_turn_time_minutes",
    "min_turn_minutes",
]

def _first_existing_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series([], dtype="float64", index=series.index)
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int).astype(float)
    numeric = pd.to_numeric(series, errors="coerce")
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        normalized = series.fillna("").astype(str).str.strip().str.lower()
        mapped = normalized.map(_INDICATOR_MAP)
        numeric = numeric.where(numeric.notna(), mapped)
    return numeric.fillna(0.0).astype(float)


def _extract_numeric_series(
    df: pd.DataFrame,
    candidates: Sequence[str],
    dataset_name: str,
    *,
    default_value: float = 0.0,
) -> tuple[pd.Series, str | None]:
    column = _first_existing_column(df.columns, candidates)
    if column is None:
        LOGGER.debug("%s missing columns %s", dataset_name, list(candidates))
        return pd.Series(default_value, index=df.index, dtype="float64"), None
    return _coerce_numeric(df[column]), column


def _ensure_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    dataset_name: str,
) -> pd.DataFrame:
    missing = [column for column in required if column not in df.columns]
    if missing:
        LOGGER.warning("%s missing columns %s; filling with <NA>", dataset_name, missing)
        for column in missing:
            df[column] = pd.NA
    return df


def _add_flight_day(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.copy()
    if "flight_day" in df.columns:
        converted = _to_day(df["flight_day"])
        if converted.notna().any():
            df["flight_day"] = converted
            return df
    for candidate in _FLIGHT_DAY_CANDIDATES:
        if candidate == "flight_day":
            continue
        if candidate in df.columns:
            converted = _to_day(df[candidate])
            if converted.notna().any():
                df["flight_day"] = converted
                return df
    LOGGER.warning("%s missing usable departure date column", dataset_name)
    df["flight_day"] = pd.NaT
    return df


def _ensure_join_keys(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = _ensure_columns(df.copy(), ["company_id", "flight_number"], dataset_name)
    df = _add_flight_day(df, dataset_name)
    return df


def _empty_aggregate(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=[*_KEY_COLUMNS, *columns])


def _prepare_flights(flights: pd.DataFrame) -> pd.DataFrame:
    dataset_name = "flights"
    if flights.empty:
        return pd.DataFrame(
            columns=[
                "company_id",
                "flight_number",
                "scheduled_departure_station_code",
                "scheduled_arrival_station_code",
                "flight_day",
                "dep_date",
                "dep_hour",
                "dow",
                "turn_sched",
                "turn_min",
                "sched_turn_slack",
                "total_seats",
                "dep_delay_min",
                "is_high_delay",
            ]
        )

    df = flights.copy()
    if "scheduled_departure_datetime_local" not in df.columns:
        fallback_col = _first_existing_column(df.columns, ["scheduled_departure_date_local", "flight_date"])
        if fallback_col:
            df["scheduled_departure_datetime_local"] = pd.to_datetime(df[fallback_col], errors="coerce")
    df = _ensure_columns(
        df,
        [
            "company_id",
            "flight_number",
            "scheduled_departure_station_code",
            "scheduled_arrival_station_code",
            "scheduled_departure_datetime_local",
        ],
        dataset_name,
    )

    schedule = pd.to_datetime(df["scheduled_departure_datetime_local"], errors="coerce")
    df["flight_day"] = _to_day(schedule)
    mask = schedule.notna()
    if not mask.any():
        LOGGER.warning("No flights with valid scheduled departure datetime")
        return pd.DataFrame(
            columns=[
                "company_id",
                "flight_number",
                "scheduled_departure_station_code",
                "scheduled_arrival_station_code",
                "flight_day",
                "dep_date",
                "dep_hour",
                "dow",
                "turn_sched",
                "turn_min",
                "sched_turn_slack",
                "total_seats",
                "dep_delay_min",
                "is_high_delay",
            ]
        )

    df = df.loc[mask].copy()
    df["flight_day"] = _to_day(df["scheduled_departure_datetime_local"])

    schedule = df["scheduled_departure_datetime_local"]
    df["dep_date"] = df["flight_day"].dt.date
    df["dep_hour"] = schedule.dt.hour.astype(int)
    df["dow"] = schedule.dt.dayofweek.astype(int)

    turn_sched, _ = _extract_numeric_series(df, _TURN_SCHED_COLUMNS, dataset_name)
    turn_min, _ = _extract_numeric_series(df, _TURN_MIN_COLUMNS, dataset_name)
    df["turn_sched"] = turn_sched.fillna(0.0)
    df["turn_min"] = turn_min.fillna(0.0)
    df["sched_turn_slack"] = df["turn_sched"] - df["turn_min"]

    seats, _ = _extract_numeric_series(df, _SEAT_COLUMNS, dataset_name)
    df["total_seats"] = seats.clip(lower=0)

    actual = pd.to_datetime(df.get("actual_departure_datetime_local"), errors="coerce")
    df["dep_delay_min"] = ((actual - schedule).dt.total_seconds() / 60.0).fillna(0.0)
    df["is_high_delay"] = df["dep_delay_min"] >= 15

    return df[
        [
            "company_id",
            "flight_number",
            "scheduled_departure_station_code",
            "scheduled_arrival_station_code",
            "flight_day",
            "dep_date",
            "dep_hour",
            "dow",
            "turn_sched",
            "turn_min",
            "sched_turn_slack",
            "total_seats",
            "dep_delay_min",
            "is_high_delay",
        ]
    ]


def _aggregate_pnr(pnr_flight: pd.DataFrame) -> pd.DataFrame:
    dataset_name = "pnr_flight"
    if pnr_flight.empty:
        return _empty_aggregate(
            [
                "total_pax",
                "basic_economy_pax",
                "is_child_count",
                "lap_child_count",
                "stroller_users",
            ]
        )

    df = _ensure_join_keys(pnr_flight, dataset_name)
    if df.empty:
        return _empty_aggregate(
            [
                "total_pax",
                "basic_economy_pax",
                "is_child_count",
                "lap_child_count",
                "stroller_users",
            ]
        )

    df["flight_day"] = _to_day(df["flight_day"])
    df = df.dropna(subset=["flight_day"])
    if df.empty:
        return _empty_aggregate(
            [
                "total_pax",
                "basic_economy_pax",
                "is_child_count",
                "lap_child_count",
                "stroller_users",
            ]
        )

    working = df.copy()
    working["row_count"] = 1.0
    for target, options in _PNR_SUM_OPTIONS.items():
        series, _ = _extract_numeric_series(working, options, dataset_name)
        working[target] = series

    grouped = (
        working.groupby(_KEY_COLUMNS, dropna=False)
        .agg(
            row_count=("row_count", "sum"),
            basic_economy_pax=("basic_economy_pax", "sum"),
            is_child_count=("is_child_count", "sum"),
            lap_child_count=("lap_child_count", "sum"),
            stroller_users=("stroller_users", "sum"),
        )
        .reset_index()
    )
    grouped = grouped.rename(columns={"row_count": "total_pax"})
    return grouped


def _aggregate_ssr(pnr_remarks: pd.DataFrame, pnr_flight: pd.DataFrame) -> pd.DataFrame:
    dataset_name = "pnr_remarks"
    if pnr_remarks.empty:
        return _empty_aggregate(["ssr_count"])

    remarks = pnr_remarks.copy()
    remarks = _ensure_columns(remarks, ["company_id", "flight_number"], dataset_name)

    if "flight_day" in remarks.columns:
        remarks["flight_day"] = _to_day(remarks["flight_day"])

    if ("flight_day" not in remarks.columns) or remarks["flight_day"].isna().all():
        if not pnr_flight.empty:
            pnr_keys = _ensure_join_keys(pnr_flight, "pnr_flight")[_KEY_COLUMNS]
            pnr_keys = pnr_keys.dropna(subset=["flight_day"]).drop_duplicates()
            remarks = remarks.merge(
                pnr_keys,
                on=["company_id", "flight_number"],
                how="left",
                suffixes=("", "_pnr"),
            )
            if "flight_day_pnr" in remarks.columns:
                remarks["flight_day"] = remarks["flight_day"].fillna(remarks["flight_day_pnr"])
                remarks = remarks.drop(columns=["flight_day_pnr"])
        else:
            remarks["flight_day"] = pd.NaT

    remarks = remarks.dropna(subset=["flight_day"])
    if remarks.empty:
        return _empty_aggregate(["ssr_count"])

    grouped = (
        remarks.groupby(_KEY_COLUMNS, dropna=False)
        .size()
        .rename("ssr_count")
        .reset_index()
    )
    return grouped


def _is_count_column(name: str | None) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return any(marker in lowered for marker in ("count", "total", "qty", "quantity", "bags", "num"))


def _aggregate_bags(bags: pd.DataFrame) -> pd.DataFrame:
    dataset_name = "bags"
    if bags.empty:
        return _empty_aggregate(["bags_total", "bags_transfer"])

    df = _ensure_join_keys(bags, dataset_name)
    if df.empty:
        return _empty_aggregate(["bags_total", "bags_transfer"])

    df["flight_day"] = _to_day(df["flight_day"])
    df = df.dropna(subset=["flight_day"])
    if df.empty:
        return _empty_aggregate(["bags_total", "bags_transfer"])

    working = df.copy()
    total_series, total_col = _extract_numeric_series(working, _BAG_COUNT_OPTIONS, dataset_name)
    if total_col is None:
        working["bags_total"] = 1.0
    else:
        working["bags_total"] = total_series

    transfer_series, transfer_col = _extract_numeric_series(working, _BAG_TRANSFER_OPTIONS, dataset_name)
    if transfer_col is None:
        working["bags_transfer"] = 0.0
    elif _is_count_column(transfer_col):
        working["bags_transfer"] = transfer_series
    else:
        factor = working["bags_total"] if total_col is not None else 1.0
        working["bags_transfer"] = transfer_series * factor

    grouped = (
        working.groupby(_KEY_COLUMNS, dropna=False)
        .agg(bags_total=("bags_total", "sum"), bags_transfer=("bags_transfer", "sum"))
        .reset_index()
    )
    return grouped


def _prepare_airport_flags(airports: pd.DataFrame) -> pd.DataFrame:
    if airports.empty:
        return pd.DataFrame(columns=["scheduled_arrival_station_code", "is_international"])
    df = airports.copy()
    code_col = _first_existing_column(df.columns, _AIRPORT_CODE_OPTIONS)
    if code_col is None:
        LOGGER.warning("airports dataset missing airport code column; assuming domestic")
        return pd.DataFrame(columns=["scheduled_arrival_station_code", "is_international"])
    country_col = _first_existing_column(df.columns, _COUNTRY_CODE_OPTIONS)
    if country_col is None:
        LOGGER.warning("airports dataset missing country code column; assuming domestic")
        df["is_international"] = False
    else:
        country = df[country_col].fillna("").astype(str).str.strip().str.upper()
        df["is_international"] = country.ne("US") & country.ne("USA")
    result = (
        df[[code_col, "is_international"]]
        .drop_duplicates(subset=[code_col])
        .rename(columns={code_col: "scheduled_arrival_station_code"})
    )
    result["is_international"] = result["is_international"].fillna(False).astype(bool)
    return result


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: np.nan, 0.0: np.nan})
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = numerator / denominator
    return ratio.fillna(0.0)


def build_feature_table(
    flights: pd.DataFrame,
    pnr_flight: pd.DataFrame,
    pnr_remarks: pd.DataFrame,
    bags: pd.DataFrame,
    airports: pd.DataFrame,
) -> pd.DataFrame:
    """Return one record per flight-day enriched with aggregated features."""
    base = _prepare_flights(flights)
    if base.empty:
        columns = [
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
            "total_pax",
            "basic_economy_pax",
            "is_child_count",
            "lap_child_count",
            "stroller_users",
            "load_factor",
            "pct_basic_econ",
            "pct_children",
            "stroller_rate",
            "ssr_count",
            "ssr_rate",
            "bags_total",
            "bags_transfer",
            "transfer_bag_ratio",
            "is_international",
            "dep_delay_min",
            "is_high_delay",
        ]
        return pd.DataFrame(columns=columns)

    pnr_agg = _aggregate_pnr(pnr_flight)
    ssr_agg = _aggregate_ssr(pnr_remarks, pnr_flight)
    bag_agg = _aggregate_bags(bags)
    airport_flags = _prepare_airport_flags(airports)

    features = base.merge(pnr_agg, on=_KEY_COLUMNS, how="left")
    features = features.merge(ssr_agg, on=_KEY_COLUMNS, how="left")
    features = features.merge(bag_agg, on=_KEY_COLUMNS, how="left")
    features = features.merge(
        airport_flags,
        on="scheduled_arrival_station_code",
        how="left",
    )

    count_columns = [
        "total_pax",
        "basic_economy_pax",
        "is_child_count",
        "lap_child_count",
        "stroller_users",
        "ssr_count",
        "bags_total",
        "bags_transfer",
    ]
    for column in count_columns:
        features[column] = features[column].fillna(0).round().astype(int)

    features["turn_sched"] = features["turn_sched"].fillna(0.0)
    features["turn_min"] = features["turn_min"].fillna(0.0)
    features["sched_turn_slack"] = features["turn_sched"] - features["turn_min"]

    features["total_seats"] = features["total_seats"].fillna(0.0)
    features["dep_delay_min"] = features["dep_delay_min"].fillna(0.0)
    features["is_high_delay"] = features["dep_delay_min"] >= 15

    features["load_factor"] = _safe_divide(features["total_pax"], features["total_seats"]).clip(0, 1)
    features["pct_basic_econ"] = _safe_divide(features["basic_economy_pax"], features["total_pax"]).clip(0, 1)
    features["pct_children"] = _safe_divide(features["is_child_count"], features["total_pax"]).clip(0, 1)
    features["stroller_rate"] = _safe_divide(features["stroller_users"], features["total_pax"]).clip(0, 1)
    features["ssr_rate"] = _safe_divide(features["ssr_count"], features["total_pax"])
    features["transfer_bag_ratio"] = _safe_divide(features["bags_transfer"], features["bags_total"]).clip(0, 1)

    features["is_international"] = features["is_international"].fillna(False).astype(bool)
    features["dep_hour"] = features["dep_hour"].clip(lower=0, upper=23).astype(int)
    features["dow"] = features["dow"].clip(lower=0, upper=6).astype(int)
    features["dep_date"] = pd.to_datetime(features["dep_date"], errors="coerce").dt.date

    columns = [
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
        "total_pax",
        "basic_economy_pax",
        "is_child_count",
        "lap_child_count",
        "stroller_users",
        "load_factor",
        "pct_basic_econ",
        "pct_children",
        "stroller_rate",
        "ssr_count",
        "ssr_rate",
        "bags_total",
        "bags_transfer",
        "transfer_bag_ratio",
        "is_international",
        "dep_delay_min",
        "is_high_delay",
    ]

    return features[columns].sort_values(["company_id", "dep_date", "flight_number"]).reset_index(drop=True)


# Legacy feature bundle implementation retained for backward compatibility.

from dataclasses import dataclass, field
class FeatureBundle:
    """Normalized model inputs."""

    features: pd.DataFrame
    target: pd.Series
    context: pd.DataFrame
    numeric_columns: List[str]
    categorical_columns: List[str]
    date_column: str | None


_DATE_CANDIDATES = (
    "scheduled_departure_date_local",
    "flight_date",
    "fl_date",
    "date",
    "scheduled_departure_datetime_local",
)
_TIME_CANDIDATES = (
    "crs_dep_time",
    "scheduled_departure",
    "dep_time",
    "scheduled_departure_datetime_local",
)
_CONTEXT_COLUMNS = (
    "scheduled_departure_date_local",
    "flight_date",
    "fl_date",
    "date",
    "flight_number",
    "flight_num",
    "tail_number",
    "tail_num",
    "carrier",
    "mkt_carrier",
    "mkt_unique_carrier",
    "airline",
    "origin",
    "dest",
    "destination",
    "scheduled_departure_station_code",
    "scheduled_arrival_station_code",
    "scheduled_departure_datetime_local",
)


def _identify_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _parse_time_to_minutes(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return (series.dt.hour * 60 + series.dt.minute).astype(float)

    def to_minutes(value: object) -> float | pd.NA:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, (int, np.integer)):
            text = f"{int(value):04d}"
        else:
            text = str(value).strip()
        if not text:
            return pd.NA
        if ":" in text:
            hour_str, minute_str = text.split(":", 1)
        else:
            text = text.zfill(4)
            hour_str, minute_str = text[:2], text[2:]
        try:
            return int(hour_str) * 60 + int(minute_str)
        except ValueError:
            return pd.NA

    return series.map(to_minutes, na_action="ignore")


def _compute_target(df: pd.DataFrame, threshold_minutes: float = 15.0) -> tuple[pd.Series, str]:
    for column in ("is_high_delay", "dep_delay", "departure_delay", "departure_delay_minutes"):
        if column in df.columns:
            delay_minutes = pd.to_numeric(df[column], errors="coerce")
            target = (delay_minutes >= threshold_minutes).astype(int)
            return target, column
    msg = "No delay column found to derive target"
    raise ValueError(msg)


def _extract_context(df: pd.DataFrame) -> pd.DataFrame:
    available = [col for col in _CONTEXT_COLUMNS if col in df.columns]
    return df[available].copy() if available else pd.DataFrame(index=df.index)


def _augment_temporal_features(df: pd.DataFrame, date_column: str | None) -> tuple[pd.DataFrame, str | None]:
    frame = df.copy()
    if date_column:
        dt = pd.to_datetime(frame[date_column], errors="coerce")
        frame["day_of_week"] = dt.dt.dayofweek
        frame["month"] = dt.dt.month
        frame["weekofyear"] = dt.dt.isocalendar().week.astype(float)
        frame["is_weekend"] = frame["day_of_week"].isin({5, 6}).astype(int)
        frame["hour_of_day"] = dt.dt.hour
        frame = frame.drop(columns=[date_column])
    for time_col in _TIME_CANDIDATES:
        if time_col in frame.columns:
            frame[f"{time_col}_minutes"] = _parse_time_to_minutes(frame[time_col])
            frame = frame.drop(columns=[time_col])
    return frame, date_column


def _split_feature_types(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=["number", "boolean", "floating", "integer"]).columns.tolist()
    categorical_cols = [
        col
        for col in df.columns
        if col not in numeric_cols and df[col].dtype in ("object", "string", "category")
    ]
    return numeric_cols, categorical_cols


def build_feature_bundle(df: pd.DataFrame) -> FeatureBundle:
    """Create model inputs, returning features, target, and context."""
    working = df.copy()
    target, target_column = _compute_target(working)
    if target_column in working.columns:
        working = working.drop(columns=[target_column])
    date_column = _identify_column(working.columns, _DATE_CANDIDATES)
    context = _extract_context(working)
    working, _ = _augment_temporal_features(working, date_column)
    numeric_cols, categorical_cols = _split_feature_types(working)
    LOGGER.info(
        "Feature matrix built with %d numeric and %d categorical columns",
        len(numeric_cols),
        len(categorical_cols),
    )
    return FeatureBundle(
        features=working,
        target=target,
        context=context,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        date_column=date_column,
    )

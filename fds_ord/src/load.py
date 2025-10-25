"""Data loading utilities tailored to the flight delay datasets."""

from __future__ import annotations

# from dataclasses import dataclass
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd

from .utils import get_logger, have_pyarrow

LOGGER = get_logger("load")

_FILE_MAP: Mapping[str, str] = {
    "Flight Level Data.csv": "flight_level",
    "PNR+Flight+Level+Data.csv": "pnr_flight_level",
    "PNR Remark Level Data.csv": "pnr_remark_level",
    "Bag+Level+Data.csv": "bag_level",
    "Airports Data.csv": "airports",
}

_EXPECTED_KEYS: Tuple[str, ...] = (
    "company_id",
    "flight_number",
    "scheduled_departure_date_local",
    "scheduled_departure_station_code",
    "scheduled_arrival_station_code",
)

_DATETIME_COLUMNS: Tuple[str, ...] = (
    "scheduled_departure_datetime_local",
    "actual_departure_datetime_local",
)


def _snake_case(name: str) -> str:
    import re

    cleaned = re.sub(r"[\s+/]+", "_", name.strip())
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", cleaned)
    cleaned = re.sub(r"(?<!^)(?=[A-Z])", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_").lower()


def _standardize_columns(df: pd.DataFrame, *, dataset_name: str) -> pd.DataFrame:
    if df.empty:
        return df
    renamed = {col: _snake_case(col) for col in df.columns}
    duplicate_targets = [val for val in renamed.values() if list(renamed.values()).count(val) > 1]
    if duplicate_targets:
        deduped: Dict[str, int] = {}
        for original, target in renamed.items():
            if target in duplicate_targets:
                idx = deduped.get(target, 0)
                deduped[target] = idx + 1
                renamed[original] = f"{target}_{idx}"
    df = df.rename(columns=renamed)
    LOGGER.debug("Standardised columns for %s: %s", dataset_name, list(df.columns))
    return df


def _read_with_pyarrow(path: Path) -> pd.DataFrame:
    import pyarrow.csv as pv

    table = pv.read_csv(path)
    return table.to_pandas(strings_to_categorical=False)


def _read_with_pandas(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype_backend="pyarrow", keep_default_na=True)


def _read_csv(path: Path) -> pd.DataFrame:
    LOGGER.info("Reading %s", path.name)
    if have_pyarrow():
        try:
            return _read_with_pyarrow(path)
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.warning("PyArrow failed for %s; falling back to pandas", path.name, exc_info=exc)
    return _read_with_pandas(path)


def _coerce_datatypes(df: pd.DataFrame, *, dataset_name: str) -> pd.DataFrame:
    if df.empty:
        return df

    for column in list(df.columns):
        series = df[column]
        if series.dtype == "object":
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().sum() and numeric.notna().sum() >= 0.8 * len(series):
                df[column] = numeric
                continue
            datetime = pd.to_datetime(series, errors="coerce", utc=False)
            if datetime.notna().sum() and datetime.notna().sum() >= 0.8 * len(series):
                df[column] = datetime
    LOGGER.debug("Coerced datatypes for %s", dataset_name)
    return df


def _parse_datetime_columns(df: pd.DataFrame, *, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")


def _ensure_expected_columns(
    df: pd.DataFrame,
    *,
    expected: Sequence[str],
    dataset_name: str,
) -> Dict[str, List[str]]:
    present = [column for column in expected if column in df.columns]
    missing = [column for column in expected if column not in df.columns]
    for column in missing:
        LOGGER.warning("%s missing expected column '%s'; filling with <NA>", dataset_name, column)
        df[column] = pd.NA
    return {"present": present, "missing": missing}


from dataclasses import dataclass, field

@dataclass
class LoadedData:
    flights: pd.DataFrame
    pnr_flight: pd.DataFrame
    pnr_remarks: pd.DataFrame
    bags: pd.DataFrame
    airports: pd.DataFrame
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def frames(self) -> Dict[str, pd.DataFrame]:
        return {
            "flight_level": self.flights,
            "pnr_flight_level": self.pnr_flight,
            "pnr_remark_level": self.pnr_remarks,
            "bag_level": self.bags,
            "airports": self.airports,
        }


def load_dataset_bundle(data_dir: Path) -> LoadedData:
    """Load the five required datasets from ``data_dir`` and validate schema."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        msg = f"Data directory not found: {data_dir}"
        raise FileNotFoundError(msg)

    dataframes: Dict[str, pd.DataFrame] = {}
    for filename, key in _FILE_MAP.items():
        path = data_dir / filename
        if not path.exists():
            msg = f"Expected data file missing: {path}"
            raise FileNotFoundError(msg)
        df = _read_csv(path)
        df = _standardize_columns(df, dataset_name=key)
        df = _coerce_datatypes(df, dataset_name=key)
        if key == "flight_level":
            _parse_datetime_columns(df, columns=_DATETIME_COLUMNS)
        dataframes[key] = df

    df_flights = dataframes.get("flight_level", pd.DataFrame())
    key_summary = _ensure_expected_columns(
        df_flights,
        expected=_EXPECTED_KEYS,
        dataset_name="flight_level",
    )
    if key_summary["missing"]:
        LOGGER.warning("Missing key columns: %s", key_summary["missing"])
    else:
        LOGGER.info("All expected key columns present")

    meta = {"key_column_summary": key_summary}
    df_pnr = dataframes.get("pnr_flight_level", pd.DataFrame())
    df_remarks = dataframes.get("pnr_remark_level", pd.DataFrame())
    df_bags = dataframes.get("bag_level", pd.DataFrame())
    df_airports = dataframes.get("airports", pd.DataFrame())

    return LoadedData(
        flights=df_flights,
        pnr_flight=df_pnr,
        pnr_remarks=df_remarks,
        bags=df_bags,
        airports=df_airports,
        meta=meta,
    )

"""Model training utilities for flight delay scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import ensure_directory, get_logger

LOGGER = get_logger("model")

FEATURE_COLUMNS: Sequence[str] = (
    "sched_turn_slack",
    "transfer_bag_ratio",
    "load_factor",
    "ssr_rate",
    "pct_children",
    "pct_basic_econ",
    "stroller_rate",
    "dep_hour",
    "is_international",
)
CONTINUOUS_FEATURES: Sequence[str] = (
    "sched_turn_slack",
    "transfer_bag_ratio",
    "load_factor",
    "ssr_rate",
    "pct_children",
    "pct_basic_econ",
    "stroller_rate",
    "dep_hour",
)
BINARY_FEATURES: Sequence[str] = ("is_international",)
TARGET_COLUMN = "is_high_delay"
DEFAULT_MODEL_PATH = Path("out/model_bundle.joblib")


@dataclass
class TrainedModel:
    model: CalibratedClassifierCV
    pipeline: Pipeline
    feature_names: List[str]


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        msg = f"Missing required feature columns: {missing}"
        raise ValueError(msg)
    features = df[list(FEATURE_COLUMNS)].copy()
    features["is_international"] = (
        features["is_international"].fillna(0).astype(int).clip(lower=0, upper=1)
    )
    return features


def _prepare_target(df: pd.DataFrame) -> pd.Series:
    if TARGET_COLUMN not in df.columns:
        msg = f"Target column '{TARGET_COLUMN}' not found"
        raise ValueError(msg)
    return pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)


def _build_pipeline() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(CONTINUOUS_FEATURES)),
            ("bin", binary_pipeline, list(BINARY_FEATURES)),
        ],
        remainder="drop",
    )


def train_calibrated_model(df: pd.DataFrame, seed: int = 17) -> TrainedModel:
    """Fit a calibrated logistic regression model on the provided features."""
    if df.empty:
        raise ValueError("Input dataframe is empty")

    X = _prepare_features(df)
    y = _prepare_target(df)

    pipeline = _build_pipeline()
    X_transformed = pipeline.fit_transform(X)

    base_estimator = LogisticRegression(
        max_iter=200,
        class_weight=None,
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        random_state=seed,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    kwargs = dict(cv=cv, method="sigmoid")
    try:
        model = CalibratedClassifierCV(estimator=base_estimator, **kwargs)
    except TypeError:  # sklearn < 1.2
        model = CalibratedClassifierCV(base_estimator=base_estimator, **kwargs)
    model.fit(X_transformed, y)

    try:
        feature_names = pipeline.get_feature_names_out().tolist()
    except Exception:
        feature_names = list(FEATURE_COLUMNS)

    LOGGER.info("Calibrated logistic model trained with %d records", len(df))
    return TrainedModel(model=model, pipeline=pipeline, feature_names=feature_names)


def predict_proba(trained: TrainedModel, df: pd.DataFrame) -> np.ndarray:
    """Return calibrated probabilities for the positive class aligned to rows."""
    if df.empty:
        return np.array([])
    X = _prepare_features(df)
    X_transformed = trained.pipeline.transform(X)
    return trained.model.predict_proba(X_transformed)[:, 1]



def _transformed_feature_names(pipe: ColumnTransformer | Pipeline, raw_names: List[str]) -> List[str]:
    """Return feature names after preprocessing when available."""
    try:
        names = pipe.get_feature_names_out()
        return names.tolist() if hasattr(names, "tolist") else list(names)
    except Exception:  # pragma: no cover - best effort
        return list(raw_names)


def report_feature_importance(
    model: CalibratedClassifierCV,
    pipe: ColumnTransformer | Pipeline,
    feature_names: List[str],
    out_dir: str | Path,
) -> Path:
    """Persist averaged absolute coefficients from calibrated folds."""
    names = _transformed_feature_names(pipe, feature_names)

    coef_list: list[np.ndarray] = []
    if hasattr(model, "calibrated_classifiers_"):
        for calibrated in model.calibrated_classifiers_:
            est = getattr(calibrated, "estimator", getattr(calibrated, "base_estimator", None))
            if est is not None and hasattr(est, "coef_"):
                coef_list.append(np.ravel(est.coef_))

    if not coef_list and hasattr(model, "base_estimator_") and hasattr(model.base_estimator_, "coef_"):
        coef_list.append(np.ravel(model.base_estimator_.coef_))

    output_path = Path(out_dir) / "feature_importance.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not coef_list:
        pd.DataFrame({"feature": names, "abs_coef": np.nan}).to_csv(output_path, index=False)
        return output_path

    coef_mean = np.mean(np.vstack(coef_list), axis=0)
    abs_coefs = np.abs(coef_mean)
    m = min(len(abs_coefs), len(names))
    importance = (
        pd.DataFrame({"feature": names[:m], "abs_coef": abs_coefs[:m]})
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(output_path, index=False)
    return output_path


def save_model(bundle: TrainedModel, path: Path | str = DEFAULT_MODEL_PATH) -> Path:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    joblib.dump(bundle, output_path)
    LOGGER.info("Model bundle saved to %s", output_path)
    return output_path


def load_model(path: Path | str = DEFAULT_MODEL_PATH) -> TrainedModel:
    model_path = Path(path)
    if not model_path.exists():
        msg = f"Model artifact not found: {model_path}"
        raise FileNotFoundError(msg)
    bundle: TrainedModel = joblib.load(model_path)
    LOGGER.info("Loaded model bundle from %s", model_path)
    return bundle

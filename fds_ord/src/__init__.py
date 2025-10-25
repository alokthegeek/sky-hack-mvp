"""Core package for the fds_ord project."""

from .load import LoadedData, load_dataset_bundle
from .features import FeatureBundle, build_feature_bundle, build_feature_table
from .model import (
    TrainedModel,
    load_model,
    predict_proba,
    report_feature_importance,
    save_model,
    train_calibrated_model,
)
from .score import export_results, score_and_rank
from .eda import compute_eda_metrics, make_eda_plots, write_markdown_summary
from .utils import ensure_directory, find_latest_export, get_logger, sanitize_name, setup_logging, validate_export

__all__ = [
    "LoadedData",
    "FeatureBundle",
    "TrainedModel",
    "build_feature_bundle",
    "build_feature_table",
    "load_dataset_bundle",
    "load_model",
    "predict_proba",
    "report_feature_importance",
    "save_model",
    "score_and_rank",
    "export_results",
    "compute_eda_metrics",
    "make_eda_plots",
    "write_markdown_summary",
    "ensure_directory",
    "find_latest_export",
    "get_logger",
    "sanitize_name",
    "setup_logging",
    "validate_export",
]

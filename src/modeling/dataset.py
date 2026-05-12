"""Assemble leakage-safe modeling tables from features, target, and splits."""

from __future__ import annotations

from typing import Any
import json
from pathlib import Path

import numpy as np
import pandas as pd


FORBIDDEN_MODEL_FEATURE_PREFIXES = (
    "jobs_",
    "amen_",
    "amenity_",
    "E_",
    "A_",
    "Y_",
)
FORBIDDEN_MODEL_FEATURE_COLUMNS = {
    "E",
    "A",
    "Y",
    "Y_60_40",
    "Y_40_60",
    "x_m",
    "y_m",
    "lon",
    "lat",
    "split",
    "spatial_cell",
}


def load_feature_columns(metadata_path: str | Path) -> list[str]:
    """Load the explicit feature allowlist from feature metadata."""
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature metadata not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    columns = metadata.get("feature_columns")
    if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        raise ValueError("Feature metadata must contain a string list named 'feature_columns'.")
    validate_model_feature_columns(columns)
    return columns


def load_modeling_table(
    city: str,
    interim_root: str | Path,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Load a city modeling table with its approved feature allowlist."""
    modeling_dir = Path(interim_root) / city / "modeling"
    dataset_path = modeling_dir / "model_dataset.parquet"
    metadata_path = modeling_dir / "metadata.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Model dataset not found: {dataset_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model dataset metadata not found: {metadata_path}")

    dataset = pd.read_parquet(dataset_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not all(
        isinstance(col, str) for col in feature_columns
    ):
        raise ValueError(
            "Model metadata must contain a string list named 'feature_columns'."
        )
    validate_model_feature_columns(feature_columns)
    return dataset, feature_columns, metadata


def validate_model_feature_columns(feature_columns: list[str]) -> None:
    """Reject columns that would leak target ingredients or spatial IDs into X."""
    forbidden = [
        col
        for col in feature_columns
        if col in FORBIDDEN_MODEL_FEATURE_COLUMNS
        or col.startswith(FORBIDDEN_MODEL_FEATURE_PREFIXES)
    ]
    if forbidden:
        raise ValueError(f"Model feature list contains leakage-prone columns: {forbidden}")


def assemble_model_dataset(
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "Y",
) -> pd.DataFrame:
    """Join approved features, target label, and split labels by block_id."""
    validate_model_feature_columns(feature_columns)

    required_feature_cols = {"block_id", *feature_columns}
    missing_features = required_feature_cols - set(feature_df.columns)
    if missing_features:
        raise KeyError(f"Missing feature columns: {sorted(missing_features)}")

    required_target_cols = {"block_id", target_column}
    missing_target = required_target_cols - set(target_df.columns)
    if missing_target:
        raise KeyError(f"Missing target columns: {sorted(missing_target)}")

    required_split_cols = {"block_id", "split"}
    missing_splits = required_split_cols - set(splits_df.columns)
    if missing_splits:
        raise KeyError(f"Missing split columns: {sorted(missing_splits)}")

    features = feature_df[["block_id", *feature_columns]].copy()
    target = target_df[["block_id", target_column]].copy()
    splits = splits_df[["block_id", "split"]].copy()
    if "spatial_cell" in splits_df.columns:
        splits["spatial_cell"] = splits_df["spatial_cell"]

    for df in [features, target, splits]:
        df["block_id"] = df["block_id"].astype(str)

    if features["block_id"].duplicated().any():
        raise ValueError("Feature table contains duplicate block_id values.")
    if target["block_id"].duplicated().any():
        raise ValueError("Target table contains duplicate block_id values.")
    if splits["block_id"].duplicated().any():
        raise ValueError("Splits table contains duplicate block_id values.")

    dataset = features.merge(target, on="block_id", how="inner")
    dataset = dataset.merge(splits, on="block_id", how="inner")

    expected_n = len(features)
    if len(dataset) != expected_n:
        raise ValueError(
            f"Model dataset has {len(dataset)} rows, expected {expected_n}; "
            "check block_id alignment across features, target, and splits."
        )

    numeric = dataset[feature_columns + [target_column]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad_cols = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"Model dataset contains non-numeric or missing values in: {bad_cols}")
    dataset[feature_columns + [target_column]] = numeric.astype(float)

    if not np.isfinite(dataset[feature_columns + [target_column]].to_numpy()).all():
        raise ValueError("Model dataset contains non-finite feature or target values.")

    split_values = set(dataset["split"].astype(str))
    required_splits = {"train", "val", "test"}
    if not required_splits.issubset(split_values):
        raise ValueError(
            f"Expected splits {sorted(required_splits)}, found {sorted(split_values)}"
        )

    return dataset

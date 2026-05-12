"""Diagnostics for spatial train/validation/test split balance."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.models.metrics import regression_metrics


def _summary_stats(series: pd.Series) -> dict[str, float]:
    desc = series.astype(float).describe()
    return {str(key): float(value) for key, value in desc.items()}


def _standardized_mean_difference(
    reference: pd.Series,
    comparison: pd.Series,
) -> float:
    ref = reference.astype(float)
    comp = comparison.astype(float)
    ref_var = float(ref.var(ddof=1))
    comp_var = float(comp.var(ddof=1))
    if not np.isfinite(comp_var):
        comp_var = ref_var
    pooled_var = (ref_var + comp_var) / 2
    if pooled_var <= 0 or not np.isfinite(pooled_var):
        return 0.0
    return float((comp.mean() - ref.mean()) / np.sqrt(pooled_var))


def build_split_diagnostics(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "Y",
    split_column: str = "split",
) -> dict[str, object]:
    """Summarize target and feature shifts across model splits."""
    required = {target_column, split_column, *feature_columns}
    missing = required - set(dataset.columns)
    if missing:
        raise KeyError(f"Dataset is missing diagnostic columns: {sorted(missing)}")

    split_counts = dataset[split_column].value_counts().to_dict()
    target_summary = {
        split: _summary_stats(group[target_column])
        for split, group in dataset.groupby(split_column)
    }

    if "train" not in set(dataset[split_column]):
        raise ValueError("Split diagnostics require a train split.")
    train = dataset[dataset[split_column] == "train"]
    train_mean = float(train[target_column].mean())

    train_mean_baseline = {}
    for split, group in dataset.groupby(split_column):
        y_true = group[target_column].astype(float).to_numpy()
        y_pred = np.full(len(group), train_mean, dtype=float)
        train_mean_baseline[split] = regression_metrics(y_true, y_pred)

    feature_shift_rows: list[dict[str, float | str]] = []
    for split, group in dataset.groupby(split_column):
        if split == "train":
            continue
        for feature in feature_columns:
            smd = _standardized_mean_difference(train[feature], group[feature])
            feature_shift_rows.append(
                {
                    "split": str(split),
                    "feature": feature,
                    "standardized_mean_difference": smd,
                    "abs_standardized_mean_difference": abs(smd),
                    "train_mean": float(train[feature].mean()),
                    "split_mean": float(group[feature].mean()),
                }
            )

    feature_shift = pd.DataFrame(feature_shift_rows)
    if feature_shift.empty:
        top_feature_shifts: list[dict[str, float | str]] = []
    else:
        top_feature_shifts = _string_key_records(
            feature_shift.sort_values(
                "abs_standardized_mean_difference",
                ascending=False,
            ).head(20)
        )

    return {
        "n_rows": int(len(dataset)),
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "target_summary_by_split": target_summary,
        "train_mean_baseline_metrics": train_mean_baseline,
        "top_feature_shifts_vs_train": top_feature_shifts,
    }


def _string_key_records(df: pd.DataFrame) -> list[dict[str, float | str]]:
    """Convert pandas record dictionaries into string-keyed dictionaries."""
    records: list[dict[str, float | str]] = []
    for row in df.to_dict(orient="records"):
        records.append(
            {str(key): cast(float | str, value) for key, value in row.items()}
        )
    return records


def build_feature_shift_table(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    split_column: str = "split",
) -> pd.DataFrame:
    """Return one row per non-train split and feature with shift diagnostics."""
    if split_column not in dataset.columns:
        raise KeyError(f"Dataset is missing split column: {split_column}")
    missing = set(feature_columns) - set(dataset.columns)
    if missing:
        raise KeyError(f"Dataset is missing feature columns: {sorted(missing)}")

    train = dataset[dataset[split_column] == "train"]
    if train.empty:
        raise ValueError("Feature shift diagnostics require a train split.")

    rows: list[dict[str, float | str]] = []
    for split, group in dataset.groupby(split_column):
        if split == "train":
            continue
        for feature in feature_columns:
            smd = _standardized_mean_difference(train[feature], group[feature])
            rows.append(
                {
                    "split": str(split),
                    "feature": feature,
                    "standardized_mean_difference": smd,
                    "abs_standardized_mean_difference": abs(smd),
                    "train_mean": float(train[feature].mean()),
                    "split_mean": float(group[feature].mean()),
                }
            )
    return pd.DataFrame(rows)

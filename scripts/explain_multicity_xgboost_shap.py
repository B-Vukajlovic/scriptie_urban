"""Explain pooled multi-city XGBoost models with SHAP.

This script targets the current thesis baseline:
- pooled multi-city XGBoost
- one pooled multi-city spatial split

It computes SHAP values on held-out spatial-test rows, so the explanations
describe out-of-sample predictions rather than fitted training rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from numbers import Integral

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_multicity_xgboost import (  # noqa: E402
    FEATURE_SET_CHOICES,
    TARGET_VIEW_CHOICES,
    FEATURE_VIEW_CHOICES,
    apply_target_view,
    build_feature_view,
    build_single_city_splits,
    load_multicity_inputs,
    model_params,
    select_feature_set,
    string_key_records,
    summarize_columns,
)
from src.models.metrics import regression_metrics  # noqa: E402
from src.utils.cities import DEFAULT_CITIES, validate_cities  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SHAP explanations for pooled multi-city XGBoost."
    )
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    parser.add_argument("--interim-root", default="data/interim")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--target-column", default="Y_global_log_minmax")
    parser.add_argument(
        "--target-view",
        choices=TARGET_VIEW_CHOICES,
        default="stored",
        help="Target definition to explain; default uses the stored global-log target column.",
    )
    parser.add_argument("--split-seed", type=int, default=1000)
    parser.add_argument("--split-val-frac", type=float, default=0.15)
    parser.add_argument("--split-test-frac", type=float, default=0.15)
    parser.add_argument("--split-grid-bins-x", type=int, default=8)
    parser.add_argument("--split-grid-bins-y", type=int, default=8)
    parser.add_argument(
        "--feature-view",
        choices=[choice for choice in FEATURE_VIEW_CHOICES if choice != "all"],
        default="log1p",
        help="Feature transformation to explain. Default matches the current main XGBoost setup.",
    )
    parser.add_argument(
        "--feature-set",
        choices=[choice for choice in FEATURE_SET_CHOICES if choice != "all_ablation"],
        default="full",
        help="Feature family subset to explain.",
    )
    parser.add_argument(
        "--reduced-feature-set",
        default="data/interim/modeling/reduced_feature_set.json",
        help="JSON feature list used when --feature-set reduced.",
    )
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel threads for XGBoost; use a positive value to avoid CPU oversubscription.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=1500,
        help="Maximum held-out test rows explained; 0 means all test rows.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top features to include in the JSON summary.",
    )
    parser.add_argument(
        "--output-label",
        default=None,
        help="Optional suffix for artifact names, for example 'smoke'.",
    )
    return parser.parse_args()


def _import_shap() -> Any:
    try:
        import shap  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for this script. Install project requirements first, "
            "for example: .venv/bin/pip install -r requirements.txt"
        ) from exc
    return shap


def _safe_expected_value(value: Any) -> float:
    arr = np.asarray(value, dtype=float).ravel()
    return float(arr[0]) if len(arr) else 0.0


def pandas_group_key_to_int(value: object, name: str) -> int:
    """Convert a pandas group key to int with a clear error for bad keys."""
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"{name} must be integer-like, got {value!r}.")


def _as_2d_shap_values(values: Any) -> np.ndarray:
    if isinstance(values, list):
        if not values:
            raise ValueError("SHAP returned an empty list of values.")
        values = values[0]
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values, got shape {arr.shape}.")
    return arr


def sample_test_rows(
    test: pd.DataFrame,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Sample held-out rows while preserving approximate city representation."""
    if max_rows <= 0 or len(test) <= max_rows:
        return test.copy()

    frac = max_rows / len(test)
    sampled_parts: list[pd.DataFrame] = []
    for _, group in test.groupby("city", sort=False):
        n = max(1, int(round(len(group) * frac)))
        sampled_parts.append(group.sample(n=min(n, len(group)), random_state=seed))

    sampled = pd.concat(sampled_parts, ignore_index=False)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=seed)
    return sampled.sort_index().copy()


def summarize_shap_values(
    shap_values: np.ndarray,
    feature_values: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for i, feature in enumerate(feature_columns):
        values = feature_values[feature].astype(float).to_numpy()
        shap_col = shap_values[:, i]
        if np.std(values) <= 0 or np.std(shap_col) <= 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(values, shap_col)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(np.mean(np.abs(shap_col))),
                "mean_shap": float(np.mean(shap_col)),
                "mean_feature_value": float(np.mean(values)),
                "feature_shap_corr": corr,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)


def summarize_city_shap_values(
    shap_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    shap_cols = [f"shap__{feature}" for feature in feature_columns]
    for city, group in shap_df.groupby("city", sort=False):
        values = group[shap_cols].to_numpy(dtype=float)
        mean_abs = np.mean(np.abs(values), axis=0)
        for feature, importance in zip(feature_columns, mean_abs):
            rows.append(
                {
                    "city": str(city),
                    "feature": feature,
                    "mean_abs_shap": float(importance),
                    "n_explained_rows": int(len(group)),
                }
            )
    return pd.DataFrame(rows).sort_values(["city", "mean_abs_shap"], ascending=[True, False])


def main() -> None:
    args = parse_args()
    shap = _import_shap()
    cities = validate_cities([str(city) for city in args.cities])

    dataset, base_feature_columns, coords, metadata_by_city = load_multicity_inputs(
        cities=cities,
        interim_root=args.interim_root,
        target_column=args.target_column,
    )
    target_view = str(args.target_view)
    dataset, target_view_metadata = apply_target_view(
        dataset=dataset,
        cities=cities,
        interim_root=args.interim_root,
        target_column=args.target_column,
        target_view=target_view,
    )
    selected_features, feature_set_metadata = select_feature_set(
        base_feature_columns,
        str(args.feature_set),
        args.reduced_feature_set,
    )
    dataset, feature_columns, feature_view_metadata = build_feature_view(
        dataset,
        selected_features,
        str(args.feature_view),
    )
    splits = build_single_city_splits(coords, args)

    shap_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = []

    seed = int(args.split_seed)
    repeat_df = dataset.merge(
        splits[["node_id", "split", "spatial_cell"]],
        on="node_id",
        how="inner",
    )
    if len(repeat_df) != len(dataset):
        raise ValueError("Spatial split does not cover all rows.")

    train = repeat_df[repeat_df["split"] == "train"].copy()
    test = repeat_df[repeat_df["split"] == "test"].copy()
    model = XGBRegressor(**model_params(args, 0))
    model.fit(train[feature_columns], train[args.target_column], verbose=False)

    explained = sample_test_rows(test=test, max_rows=args.max_test_rows, seed=seed)
    x_explain = explained[feature_columns]
    y_pred = model.predict(x_explain).astype(float)
    metrics = regression_metrics(explained[args.target_column].to_numpy(), y_pred)
    metric_rows.append(
        {
            "repeat": 0,
            "seed": seed,
            "split": "test_explained_sample",
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "n_explained_rows": int(len(explained)),
        }
    )

    explainer = shap.TreeExplainer(model)
    shap_values = _as_2d_shap_values(
        explainer.shap_values(x_explain, check_additivity=False)
    )
    base_value = _safe_expected_value(explainer.expected_value)

    prediction_frames.append(
        pd.DataFrame(
            {
                "repeat": 0,
                "seed": seed,
                "node_id": explained["node_id"].to_numpy(),
                "city": explained["city"].to_numpy(),
                "block_id": explained["block_id"].to_numpy(),
                "y_true": explained[args.target_column].to_numpy(dtype=float),
                "y_pred": y_pred,
                "shap_base_value": base_value,
            }
        )
    )

    shap_frame = pd.DataFrame(
        {
            f"shap__{feature}": shap_values[:, i]
            for i, feature in enumerate(feature_columns)
        }
    )
    shap_frame.insert(0, "block_id", explained["block_id"].to_numpy())
    shap_frame.insert(0, "city", explained["city"].to_numpy())
    shap_frame.insert(0, "node_id", explained["node_id"].to_numpy())
    shap_frame.insert(0, "seed", seed)
    shap_frame.insert(0, "repeat", 0)
    shap_frames.append(shap_frame)

    shap_df = pd.concat(shap_frames, ignore_index=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)

    feature_value_frames: list[pd.DataFrame] = []
    for repeat, pred_ids in predictions_df.groupby("repeat")["node_id"]:
        repeat_id = pandas_group_key_to_int(repeat, "repeat")
        repeat_values = dataset[dataset["node_id"].isin(set(pred_ids))][
            ["node_id", *feature_columns]
        ].copy()
        repeat_values.insert(0, "repeat", repeat_id)
        feature_value_frames.append(repeat_values)
    feature_values = pd.concat(feature_value_frames, ignore_index=True)

    shap_with_values = shap_df.merge(
        feature_values,
        on=["repeat", "node_id"],
        how="left",
        validate="one_to_one",
    )
    if shap_with_values[feature_columns].isna().any().any():
        raise ValueError("Failed to align explained feature values with SHAP rows.")

    shap_values_matrix = shap_df[[f"shap__{feature}" for feature in feature_columns]].to_numpy(
        dtype=float
    )
    importance_df = summarize_shap_values(
        shap_values=shap_values_matrix,
        feature_values=shap_with_values[feature_columns],
        feature_columns=feature_columns,
    )
    city_importance_df = summarize_city_shap_values(shap_df, feature_columns)

    output_root = Path(args.outputs_root)
    metrics_dir = output_root / "metrics"
    tables_dir = output_root / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    label = "xgboost"
    prefix = f"multicity_{label}_shap"
    if args.output_label:
        safe_label = str(args.output_label).strip().lower().replace(" ", "_")
        prefix = f"{prefix}_{safe_label}"
    shap_values_path = tables_dir / f"{prefix}_values.parquet"
    predictions_path = tables_dir / f"{prefix}_predictions.csv"
    importance_path = tables_dir / f"{prefix}_feature_importance.csv"
    city_importance_path = tables_dir / f"{prefix}_city_feature_importance.csv"
    metrics_path = tables_dir / f"{prefix}_metrics.csv"

    shap_df.to_parquet(shap_values_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    importance_df.to_csv(importance_path, index=False)
    city_importance_df.to_csv(city_importance_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    summary = {
        "model": label,
        "cities": cities,
        "target_column": args.target_column,
        "target_view": target_view,
        "target_view_metadata": target_view_metadata,
        "n_rows": int(len(dataset)),
        "n_features": int(len(feature_columns)),
        "base_n_features": int(len(base_feature_columns)),
        "feature_set": str(args.feature_set),
        "feature_view": str(args.feature_view),
        "feature_set_metadata": feature_set_metadata,
        "feature_view_metadata": feature_view_metadata,
        "n_spatial_splits": 1,
        "n_explained_rows": int(len(shap_df)),
        "max_test_rows": int(args.max_test_rows),
        "output_label": args.output_label,
        "split_seed": int(args.split_seed),
        "split_config": {
            "val_frac": float(args.split_val_frac),
            "test_frac": float(args.split_test_frac),
            "grid_bins_x": int(args.split_grid_bins_x),
            "grid_bins_y": int(args.split_grid_bins_y),
        },
        "hyperparameters": {
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "max_depth": int(args.max_depth),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "min_child_weight": float(args.min_child_weight),
            "reg_alpha": float(args.reg_alpha),
            "reg_lambda": float(args.reg_lambda),
            "random_state": int(args.random_state),
            "n_jobs": int(args.n_jobs),
        },
        "city_inputs": metadata_by_city,
        "explained_sample_metrics": summarize_columns(
            metrics_df,
            "split",
            ["rmse", "mae", "r2"],
        ),
        "top_features": string_key_records(importance_df.head(args.top_n)),
        "artifacts": {
            "shap_values": str(shap_values_path),
            "predictions": str(predictions_path),
            "feature_importance": str(importance_path),
            "city_feature_importance": str(city_importance_path),
            "metrics": str(metrics_path),
        },
    }
    summary_path = metrics_dir / f"{prefix}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Evaluate XGBoost across multiple cities with one spatial split and LOCO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.spatial_splits import build_spatial_train_val_test_splits  # noqa: E402
from src.modeling.dataset import load_modeling_table  # noqa: E402
from src.models.metrics import regression_metrics  # noqa: E402
from src.target.global_target import (  # noqa: E402
    build_global_target_columns,
    load_multicity_target_components,
    target_value_frame,
)
from src.utils.cities import DEFAULT_CITIES, validate_cities  # noqa: E402

FeatureView = str
FeatureSet = str
TargetView = str
FEATURE_VIEW_CHOICES = (
    "raw",
    "log1p",
    "city_zscore",
    "raw_plus_city_context",
    "city_zscore_plus_city_context",
    "all",
)
FEATURE_SET_CHOICES = (
    "full",
    "reduced",
    "pt",
    "be",
    "acs",
    "pt_be",
    "pt_acs",
    "be_acs",
    "full_no_acs_commute",
    "all_ablation",
)
TARGET_VIEW_CHOICES = (
    "stored",
    "city_relative",
    "global_minmax",
    "global_log_minmax",
    "city_rank",
)
ACS_COMMUTE_BEHAVIOR_COLUMNS = (
    "acs_avg_commute_time_min",
    "acs_commute_60_plus_min_share",
    "acs_public_transit_commute_share",
    "acs_car_commute_share",
    "acs_walk_commute_share",
    "acs_bike_commute_share",
    "acs_work_from_home_share",
)
CITY_CONTEXT_SOURCE_COLUMNS = (
    "pt_stop_density_per_km2",
    "pt_route_count",
    "pt_weekday_departures",
    "pt_peak_departures",
    "pt_stops_within_800m",
    "pt_departures_within_800m",
    "pt_stops_within_2000m",
    "pt_departures_within_2000m",
    "be_street_length_density_m_per_km2",
    "be_intersection_density_per_km2",
    "be_bikeable_street_share",
    "acs_population_density_per_km2",
    "acs_median_household_income",
    "acs_poverty_share",
    "acs_zero_vehicle_household_share",
    "acs_public_transit_commute_share",
)
LOG1P_FEATURE_KEYWORDS = (
    "count",
    "density",
    "departures",
    "routes",
    "stops",
    "length",
    "income",
    "rent",
    "population",
    "dist_m",
    "area_m2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pooled single-spatial-split and LOCO XGBoost models."
    )
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    parser.add_argument("--interim-root", default="data/interim")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--target-column", default="Y_global_log_minmax")
    parser.add_argument(
        "--target-view",
        choices=TARGET_VIEW_CHOICES,
        default="stored",
        help=(
            "Target definition used for modeling. stored uses --target-column as-is; "
            "city_relative keeps legacy per-city Y; global_* rebuilds Y from raw "
            "jobs/amenities with cross-city normalization; city_rank predicts "
            "within-city percentile rank."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=1000)
    parser.add_argument("--split-val-frac", type=float, default=0.15)
    parser.add_argument("--split-test-frac", type=float, default=0.15)
    parser.add_argument("--split-grid-bins-x", type=int, default=8)
    parser.add_argument("--split-grid-bins-y", type=int, default=8)
    parser.add_argument(
        "--feature-views",
        nargs="+",
        choices=FEATURE_VIEW_CHOICES,
        default=["log1p"],
        help=(
            "Feature transformations to evaluate. Use 'all' to run raw, log1p, "
            "city_zscore, and raw_plus_city_context."
        ),
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=FEATURE_SET_CHOICES,
        default=["full"],
        help=(
            "Feature families to evaluate. Use 'all_ablation' to run full, PT only, "
            "BE only, ACS only, PT+BE, and full without ACS commute-behavior columns."
        ),
    )
    parser.add_argument(
        "--reduced-feature-set",
        default="data/interim/modeling/reduced_feature_set.json",
        help="JSON feature list used when --feature-sets includes 'reduced'.",
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
        "--output-prefix",
        default=None,
        help=(
            "Optional artifact prefix. Defaults to multicity_xgboost."
        ),
    )
    return parser.parse_args()


def _std(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def model_params(args: argparse.Namespace, seed_offset: int = 0) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": float(getattr(args, "min_child_weight", 1.0)),
        "reg_alpha": float(getattr(args, "reg_alpha", 0.0)),
        "reg_lambda": float(getattr(args, "reg_lambda", 1.0)),
        "random_state": int(args.random_state + seed_offset),
        "n_jobs": int(args.n_jobs),
        "tree_method": "hist",
        "eval_metric": "rmse",
    }


def summarize_columns(
    df: pd.DataFrame,
    group_col: str,
    value_cols: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for group_value, group_df in df.groupby(group_col):
        group_summary: dict[str, dict[str, float]] = {}
        for col in value_cols:
            values = group_df[col].astype(float)
            group_summary[col] = {
                "mean": float(values.mean()),
                "std": _std(values),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        summary[str(group_value)] = group_summary
    return summary


def resolve_feature_views(requested: list[str]) -> list[FeatureView]:
    """Expand and de-duplicate requested feature views."""
    if "all" in requested:
        requested = [
            "raw",
            "log1p",
            "city_zscore",
            "raw_plus_city_context",
            "city_zscore_plus_city_context",
        ]
    views: list[FeatureView] = []
    for view in requested:
        if view == "all":
            continue
        if view not in views:
            views.append(view)
    if not views:
        raise ValueError("At least one feature view must be selected.")
    return views


def resolve_feature_sets(requested: list[str]) -> list[FeatureSet]:
    """Expand and de-duplicate requested feature family sets."""
    if "all_ablation" in requested:
        requested = [
            "full",
            "pt",
            "be",
            "acs",
            "pt_be",
            "full_no_acs_commute",
        ]
    sets: list[FeatureSet] = []
    for feature_set in requested:
        if feature_set == "all_ablation":
            continue
        if feature_set not in sets:
            sets.append(feature_set)
    if not sets:
        raise ValueError("At least one feature set must be selected.")
    return sets


def select_feature_set(
    feature_columns: list[str],
    feature_set: FeatureSet,
    reduced_feature_set_path: str | Path = "data/interim/modeling/reduced_feature_set.json",
) -> tuple[list[str], dict[str, Any]]:
    """Select leakage-safe feature families for thesis ablations."""
    if feature_set == "full":
        selected = feature_columns.copy()
    elif feature_set == "reduced":
        path = Path(reduced_feature_set_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Reduced feature set not found: {path}. "
                "Pass one of the final feature-set JSON files in data/interim/modeling/."
            )
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        raw_selected = payload.get("feature_columns")
        if not isinstance(raw_selected, list) or not all(
            isinstance(col, str) for col in raw_selected
        ):
            raise ValueError("Reduced feature set JSON must contain string list 'feature_columns'.")
        missing = [col for col in raw_selected if col not in feature_columns]
        if missing:
            raise ValueError(f"Reduced feature set contains unknown features: {missing}")
        selected = raw_selected
    elif feature_set == "pt":
        selected = [col for col in feature_columns if col.startswith("pt_")]
    elif feature_set == "be":
        selected = [col for col in feature_columns if col.startswith("be_")]
    elif feature_set == "acs":
        selected = [col for col in feature_columns if col.startswith("acs_")]
    elif feature_set == "pt_be":
        selected = [
            col for col in feature_columns if col.startswith(("pt_", "be_"))
        ]
    elif feature_set == "pt_acs":
        selected = [
            col for col in feature_columns if col.startswith(("pt_", "acs_"))
        ]
    elif feature_set == "be_acs":
        selected = [
            col for col in feature_columns if col.startswith(("be_", "acs_"))
        ]
    elif feature_set == "full_no_acs_commute":
        selected = [
            col for col in feature_columns if col not in ACS_COMMUTE_BEHAVIOR_COLUMNS
        ]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    if not selected:
        raise ValueError(f"Feature set '{feature_set}' selected no columns.")

    excluded = [col for col in feature_columns if col not in selected]
    metadata: dict[str, Any] = {
        "feature_set": feature_set,
        "n_features": int(len(selected)),
        "n_excluded_features": int(len(excluded)),
        "family_counts": {
            "pt": int(sum(col.startswith("pt_") for col in selected)),
            "be": int(sum(col.startswith("be_") for col in selected)),
            "acs": int(sum(col.startswith("acs_") for col in selected)),
        },
    }
    if feature_set == "reduced":
        metadata["reduced_feature_set_path"] = str(reduced_feature_set_path)
    if feature_set == "full_no_acs_commute":
        metadata["excluded_acs_commute_behavior_columns"] = [
            col for col in ACS_COMMUTE_BEHAVIOR_COLUMNS if col in feature_columns
        ]
    return selected, metadata


def build_global_target_view(
    target_components: pd.DataFrame,
    cities: list[str],
    interim_root: str | Path,
    target_view: TargetView,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build cross-city-comparable target values from raw reachability columns."""
    global_targets, metadata = build_global_target_columns(
        target_components=target_components,
        cities=cities,
        interim_root=interim_root,
        target_view=target_view,
    )
    return target_value_frame(global_targets, target_view), metadata


def apply_target_view(
    dataset: pd.DataFrame,
    cities: list[str],
    interim_root: str | Path,
    target_column: str,
    target_view: TargetView,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace the modeling label with the requested target view."""
    out = dataset.copy()
    if target_view == "stored":
        return out, {
            "target_view": target_view,
            "target_column": target_column,
            "target_normalization_scope": "stored_modeling_table",
        }

    if target_view == "city_relative":
        return out, {
            "target_view": target_view,
            "target_column": target_column,
            "target_normalization_scope": "per_city_existing_target_table",
        }

    if target_view == "city_rank":
        out[target_column] = (
            out.groupby("city", sort=False)[target_column]
            .rank(method="average", pct=True)
            .astype(float)
        )
        return out, {
            "target_view": target_view,
            "target_normalization_scope": "within_city_percentile_rank",
        }

    target_components = load_multicity_target_components(cities, interim_root)
    target_values, metadata = build_global_target_view(
        target_components,
        cities,
        interim_root,
        target_view,
    )
    out = out.drop(columns=[target_column]).merge(target_values, on="node_id", how="inner")
    if len(out) != len(dataset):
        raise ValueError("Global target view did not align with all modeling rows.")
    out = out.rename(columns={"target_value": target_column})
    return out, metadata


def _log1p_columns(dataset: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    """Pick nonnegative, skew-prone magnitude columns for log1p transforms."""
    columns: list[str] = []
    for column in feature_columns:
        if not any(keyword in column for keyword in LOG1P_FEATURE_KEYWORDS):
            continue
        values = dataset[column].astype(float)
        if float(values.min()) < 0 or float(values.max()) <= 1:
            continue
        columns.append(column)
    return columns


def _apply_log1p(dataset: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = dataset.copy()
    transformed = _log1p_columns(out, feature_columns)
    for column in transformed:
        out[column] = np.log1p(out[column].astype(float).clip(lower=0.0))
    return out, transformed


def _apply_city_zscore(dataset: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    out = dataset.copy()
    grouped = out.groupby("city", sort=False)
    for column in feature_columns:
        means = grouped[column].transform("mean").astype(float)
        stds = grouped[column].transform("std").astype(float).replace(0.0, np.nan)
        out[column] = ((out[column].astype(float) - means) / stds).fillna(0.0)
    return out


def _add_city_context_features(
    dataset: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    out = dataset.copy()
    available = [col for col in CITY_CONTEXT_SOURCE_COLUMNS if col in feature_columns]
    added: list[str] = []
    grouped = out.groupby("city", sort=False)
    for column in available:
        mean_col = f"city_mean__{column}"
        std_col = f"city_std__{column}"
        out[mean_col] = grouped[column].transform("mean").astype(float)
        out[std_col] = grouped[column].transform("std").astype(float).fillna(0.0)
        added.extend([mean_col, std_col])
    return out, [*feature_columns, *added]


def build_feature_view(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    view: FeatureView,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Create a modeling feature view without mutating the source table."""
    metadata: dict[str, Any] = {"feature_view": view}
    if view == "raw":
        return dataset.copy(), feature_columns.copy(), metadata
    if view == "log1p":
        out, transformed = _apply_log1p(dataset, feature_columns)
        metadata["log1p_columns"] = transformed
        return out, feature_columns.copy(), metadata
    if view == "city_zscore":
        out = _apply_city_zscore(dataset, feature_columns)
        metadata["city_zscore_columns"] = feature_columns.copy()
        return out, feature_columns.copy(), metadata
    if view == "raw_plus_city_context":
        out, view_features = _add_city_context_features(dataset, feature_columns)
        metadata["city_context_columns"] = [
            col for col in view_features if col not in feature_columns
        ]
        return out, view_features, metadata
    if view == "city_zscore_plus_city_context":
        out = _apply_city_zscore(dataset, feature_columns)
        out, view_features = _add_city_context_features(out, feature_columns)
        metadata["city_zscore_columns"] = feature_columns.copy()
        metadata["city_context_columns"] = [
            col for col in view_features if col not in feature_columns
        ]
        return out, view_features, metadata
    raise ValueError(f"Unknown feature view: {view}")


def load_target_coordinates(city: str, interim_root: str | Path) -> pd.DataFrame:
    target_path = Path(interim_root) / city / "target" / "target_table.parquet"
    if not target_path.exists():
        raise FileNotFoundError(f"Target table not found: {target_path}")
    coords = pd.read_parquet(target_path, columns=["block_id", "x_m", "y_m"])
    coords["block_id"] = coords["block_id"].astype(str)
    coords.insert(0, "city", city)
    return coords


def load_city_model_frame(
    city: str,
    interim_root: str | Path,
    target_column: str,
    expected_features: list[str] | None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    dataset, feature_columns, metadata = load_modeling_table(city, interim_root)
    if expected_features is not None and feature_columns != expected_features:
        raise ValueError(f"{city} feature columns do not match the first city.")

    required = {"block_id", target_column, *feature_columns}
    missing = required - set(dataset.columns)
    if missing:
        raise KeyError(f"{city} model dataset is missing columns: {sorted(missing)}")

    base = dataset[["block_id", target_column, *feature_columns]].copy()
    base["block_id"] = base["block_id"].astype(str)
    frame = pd.concat(
        [
            base[["block_id", target_column]].reset_index(drop=True),
            base[feature_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    frame.insert(0, "city", city)
    frame.insert(0, "node_id", city + ":" + frame["block_id"].astype(str))
    return frame, feature_columns, metadata


def load_multicity_inputs(
    cities: list[str],
    interim_root: str | Path,
    target_column: str,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    coords: list[pd.DataFrame] = []
    metadata_by_city: dict[str, Any] = {}
    expected_base_features: list[str] | None = None
    feature_columns: list[str] | None = None

    for city in cities:
        frame, city_features, metadata = load_city_model_frame(
            city=city,
            interim_root=interim_root,
            target_column=target_column,
            expected_features=expected_base_features,
        )
        if expected_base_features is None:
            expected_base_features = city_features
            feature_columns = city_features
        elif feature_columns != city_features:
            raise ValueError(f"{city} feature columns do not match.")
        frames.append(frame)
        coords.append(load_target_coordinates(city, interim_root))
        metadata_by_city[city] = {
            "n_rows": int(len(frame)),
            "single_split_counts": metadata.get("split_counts"),
            "inputs": metadata.get("inputs"),
        }

    if feature_columns is None:
        raise ValueError("No city inputs were loaded.")
    return (
        pd.concat(frames, ignore_index=True),
        feature_columns,
        pd.concat(coords, ignore_index=True),
        metadata_by_city,
    )


def build_single_city_splits(
    coords: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    splits: list[pd.DataFrame] = []
    for city, city_coords in coords.groupby("city", sort=False):
        split_df = build_spatial_train_val_test_splits(
            city_coords[["block_id", "x_m", "y_m"]],
            seed=args.split_seed,
            val_frac=args.split_val_frac,
            test_frac=args.split_test_frac,
            grid_bins_x=args.split_grid_bins_x,
            grid_bins_y=args.split_grid_bins_y,
        )
        split_df.insert(0, "seed", int(args.split_seed))
        split_df.insert(0, "repeat", 0)
        split_df.insert(0, "city", str(city))
        split_df.insert(0, "node_id", str(city) + ":" + split_df["block_id"].astype(str))
        splits.append(split_df)
    return pd.concat(splits, ignore_index=True)


def append_metrics(
    rows: list[dict[str, float | int | str]],
    *,
    target_view: TargetView,
    feature_view: FeatureView,
    feature_set: FeatureSet,
    experiment: str,
    repeat: int,
    seed: int,
    split: str,
    city: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    train_mean: float,
) -> None:
    metrics = regression_metrics(y_true.to_numpy(), y_pred)
    baseline_pred = np.full(len(y_true), train_mean, dtype=float)
    baseline = regression_metrics(y_true.to_numpy(), baseline_pred)
    y = y_true.to_numpy(dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    pearson = float(np.corrcoef(y, pred)[0, 1]) if np.std(y) > 0 and np.std(pred) > 0 else 0.0
    spearman_value = pd.Series(y).corr(pd.Series(pred), method="spearman")
    spearman = float(spearman_value) if pd.notna(spearman_value) else 0.0
    bias_corrected_pred = pred + (float(np.mean(y)) - float(np.mean(pred)))
    bias_corrected = regression_metrics(y, bias_corrected_pred)
    if np.std(pred) > 0:
        affine_pred = (pred - np.mean(pred)) / np.std(pred) * np.std(y) + np.mean(y)
        affine_corrected = regression_metrics(y, affine_pred)
    else:
        affine_corrected = {"r2": 0.0}
    rows.append(
        {
            "target_view": target_view,
            "feature_view": feature_view,
            "feature_set": feature_set,
            "experiment": experiment,
            "repeat": int(repeat),
            "seed": int(seed),
            "split": split,
            "city": city,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "pearson_r": pearson,
            "spearman_r": spearman,
            "bias_corrected_r2": bias_corrected["r2"],
            "affine_corrected_r2": affine_corrected["r2"],
            "baseline_rmse": baseline["rmse"],
            "baseline_mae": baseline["mae"],
            "baseline_r2": baseline["r2"],
            "rmse_improvement_vs_baseline_pct": float(
                100 * (baseline["rmse"] - metrics["rmse"]) / baseline["rmse"]
            ),
            "n_rows": int(len(y_true)),
            "target_mean": float(y_true.mean()),
            "target_std": _std(y_true),
            "prediction_mean": float(np.mean(y_pred)),
            "prediction_bias": float(np.mean(y_pred) - y_true.mean()),
        }
    )


def run_pooled_spatial_cv(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    splits: pd.DataFrame,
    args: argparse.Namespace,
    target_view: TargetView,
    feature_view: FeatureView,
    feature_set: FeatureSet,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, float | int | str]] = []
    predictions: list[pd.DataFrame] = []
    importances: list[dict[str, float | int | str]] = []

    seed = int(splits["seed"].iloc[0])
    repeat_df = dataset.merge(
        splits[["node_id", "split", "spatial_cell"]],
        on="node_id",
        how="inner",
    )
    if len(repeat_df) != len(dataset):
        raise ValueError("Spatial split does not cover all rows.")

    train = repeat_df[repeat_df["split"] == "train"].copy()
    model = XGBRegressor(**model_params(args, 0))
    model.fit(train[feature_columns], train[args.target_column], verbose=False)
    train_mean = float(train[args.target_column].mean())

    for split_name in ["train", "val", "test"]:
        split_df = repeat_df[repeat_df["split"] == split_name].copy()
        y_pred = model.predict(split_df[feature_columns]).astype(float)
        append_metrics(
            metric_rows,
            experiment="pooled_spatial_cv",
            target_view=target_view,
            feature_view=feature_view,
            feature_set=feature_set,
            repeat=0,
            seed=seed,
            split=split_name,
            city="__all__",
            y_true=split_df[args.target_column],
            y_pred=y_pred,
            train_mean=train_mean,
        )
        predictions.append(
            pd.DataFrame(
                {
                    "experiment": "pooled_spatial_cv",
                    "target_view": target_view,
                    "feature_view": feature_view,
                    "feature_set": feature_set,
                    "repeat": 0,
                    "seed": seed,
                    "city": split_df["city"].to_numpy(),
                    "block_id": split_df["block_id"].to_numpy(),
                    "split": split_name,
                    "y_true": split_df[args.target_column].to_numpy(dtype=float),
                    "y_pred": y_pred,
                }
            )
        )

        if split_name == "test":
            for city, city_df in split_df.groupby("city"):
                city_pred = y_pred[split_df["city"].to_numpy() == city]
                append_metrics(
                    metric_rows,
                    experiment="pooled_spatial_cv",
                    target_view=target_view,
                    feature_view=feature_view,
                    feature_set=feature_set,
                    repeat=0,
                    seed=seed,
                    split="test",
                    city=str(city),
                    y_true=city_df[args.target_column],
                    y_pred=city_pred,
                    train_mean=train_mean,
                )

    for feature, importance in zip(feature_columns, model.feature_importances_.astype(float)):
        importances.append(
            {
                "feature_view": feature_view,
                "target_view": target_view,
                "feature_set": feature_set,
                "experiment": "pooled_spatial_cv",
                "repeat": 0,
                "seed": seed,
                "feature": feature,
                "importance_gain": float(importance),
            }
        )

    return (
        pd.DataFrame(metric_rows),
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(importances),
    )


def run_leave_one_city_out(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    cities: list[str],
    args: argparse.Namespace,
    target_view: TargetView,
    feature_view: FeatureView,
    feature_set: FeatureSet,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, float | int | str]] = []
    predictions: list[pd.DataFrame] = []
    importances: list[dict[str, float | int | str]] = []

    for i, heldout_city in enumerate(cities):
        train = dataset[dataset["city"] != heldout_city].copy()
        test = dataset[dataset["city"] == heldout_city].copy()
        model = XGBRegressor(**model_params(args, i))
        model.fit(train[feature_columns], train[args.target_column], verbose=False)
        train_mean = float(train[args.target_column].mean())

        for split_name, split_df in [("train", train), ("test", test)]:
            y_pred = model.predict(split_df[feature_columns]).astype(float)
            append_metrics(
                metric_rows,
                experiment="leave_one_city_out",
                target_view=target_view,
                feature_view=feature_view,
                feature_set=feature_set,
                repeat=i,
                seed=int(args.random_state + i),
                split=split_name,
                city=heldout_city if split_name == "test" else "__train_cities__",
                y_true=split_df[args.target_column],
                y_pred=y_pred,
                train_mean=train_mean,
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "target_view": target_view,
                        "feature_view": feature_view,
                        "feature_set": feature_set,
                        "experiment": "leave_one_city_out",
                        "repeat": i,
                        "seed": int(args.random_state + i),
                        "heldout_city": heldout_city,
                        "city": split_df["city"].to_numpy(),
                        "block_id": split_df["block_id"].to_numpy(),
                        "split": split_name,
                        "y_true": split_df[args.target_column].to_numpy(dtype=float),
                        "y_pred": y_pred,
                    }
                )
            )

        for feature, importance in zip(feature_columns, model.feature_importances_.astype(float)):
            importances.append(
                {
                    "target_view": target_view,
                    "feature_view": feature_view,
                    "feature_set": feature_set,
                    "experiment": "leave_one_city_out",
                    "repeat": i,
                    "seed": int(args.random_state + i),
                    "heldout_city": heldout_city,
                    "feature": feature,
                    "importance_gain": float(importance),
                }
            )

    return (
        pd.DataFrame(metric_rows),
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(importances),
    )


def importance_summary(
    importance_df: pd.DataFrame,
    group_columns: list[str],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for group_key, group in importance_df.groupby(group_columns):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        label = " / ".join(str(part) for part in group_key)
        summary = (
            group.groupby("feature")["importance_gain"]
            .agg(["mean", "std", "min", "max"])
            .fillna(0.0)
            .reset_index()
            .sort_values("mean", ascending=False)
            .head(20)
        )
        out[label] = string_key_records(summary)
    return out


def feature_view_summary(
    metrics_df: pd.DataFrame,
    metric_cols: list[str],
) -> dict[str, dict[str, Any]]:
    """Summarize pooled spatial and LOCO metrics separately by feature view."""
    out: dict[str, dict[str, Any]] = {}
    for view, view_metrics in metrics_df.groupby("feature_view", sort=False):
        pooled_global = view_metrics[
            (view_metrics["experiment"] == "pooled_spatial_cv")
            & (view_metrics["city"] == "__all__")
        ]
        loco_test = view_metrics[
            (view_metrics["experiment"] == "leave_one_city_out")
            & (view_metrics["split"] == "test")
        ]
        out[str(view)] = {
            "pooled_spatial_cv_summary_by_split": summarize_columns(
                pooled_global,
                "split",
                metric_cols,
            ),
            "leave_one_city_out_summary": summarize_columns(
                loco_test,
                "split",
                metric_cols,
            ),
            "leave_one_city_out_by_city": string_key_records(
                loco_test.sort_values("city")
            ),
        }
    return out


def feature_config_summary(
    metrics_df: pd.DataFrame,
    metric_cols: list[str],
) -> dict[str, dict[str, Any]]:
    """Summarize results for every feature-set / feature-view combination."""
    out: dict[str, dict[str, Any]] = {}
    for (feature_set, view), group in metrics_df.groupby(
        ["feature_set", "feature_view"],
        sort=False,
    ):
        pooled_global = group[
            (group["experiment"] == "pooled_spatial_cv")
            & (group["city"] == "__all__")
        ]
        loco_test = group[
            (group["experiment"] == "leave_one_city_out")
            & (group["split"] == "test")
        ]
        label = f"{feature_set} / {view}"
        out[label] = {
            "feature_set": str(feature_set),
            "feature_view": str(view),
            "pooled_spatial_cv_summary_by_split": summarize_columns(
                pooled_global,
                "split",
                metric_cols,
            ),
            "leave_one_city_out_summary": summarize_columns(
                loco_test,
                "split",
                metric_cols,
            ),
            "leave_one_city_out_by_city": string_key_records(
                loco_test.sort_values("city")
            ),
        }
    return out


def string_key_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert pandas record dictionaries into string-keyed dictionaries."""
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append({str(key): value for key, value in row.items()})
    return records


def main() -> None:
    args = parse_args()
    cities = validate_cities([str(city) for city in args.cities])
    feature_views = resolve_feature_views([str(view) for view in args.feature_views])
    feature_sets = resolve_feature_sets([str(feature_set) for feature_set in args.feature_sets])
    dataset, feature_columns, coords, metadata_by_city = load_multicity_inputs(
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
    splits = build_single_city_splits(coords, args)

    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    feature_config_metadata: dict[str, Any] = {}

    for feature_set in feature_sets:
        selected_features, set_metadata = select_feature_set(
            feature_columns,
            feature_set,
            args.reduced_feature_set,
        )
        for view in feature_views:
            view_dataset, view_feature_columns, view_metadata = build_feature_view(
                dataset,
                selected_features,
                view,
            )
            config_key = f"{feature_set} / {view}"
            feature_config_metadata[config_key] = {
                **set_metadata,
                **view_metadata,
                "n_features": int(len(view_feature_columns)),
            }
            pooled_metrics, pooled_predictions, pooled_importance = run_pooled_spatial_cv(
                view_dataset,
                view_feature_columns,
                splits,
                args,
                target_view,
                view,
                feature_set,
            )
            loco_metrics, loco_predictions, loco_importance = run_leave_one_city_out(
                view_dataset,
                view_feature_columns,
                cities,
                args,
                target_view,
                view,
                feature_set,
            )
            metric_frames.extend([pooled_metrics, loco_metrics])
            prediction_frames.extend([pooled_predictions, loco_predictions])
            importance_frames.extend([pooled_importance, loco_importance])

    metrics_df = pd.concat(metric_frames, ignore_index=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    importance_df = pd.concat(importance_frames, ignore_index=True)

    output_root = Path(args.outputs_root)
    metrics_dir = output_root / "metrics"
    tables_dir = output_root / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    label = "xgboost"
    prefix = str(args.output_prefix) if args.output_prefix else "multicity_xgboost"
    metrics_path = tables_dir / f"{prefix}_metrics.csv"
    predictions_path = tables_dir / f"{prefix}_predictions.csv"
    splits_path = tables_dir / f"{prefix}_pooled_spatial_splits.csv"
    importance_path = tables_dir / f"{prefix}_feature_importance.csv"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    splits.to_csv(splits_path, index=False)
    importance_df.to_csv(importance_path, index=False)

    metric_cols = [
        "rmse",
        "mae",
        "r2",
        "pearson_r",
        "spearman_r",
        "bias_corrected_r2",
        "affine_corrected_r2",
        "baseline_rmse",
        "baseline_r2",
        "rmse_improvement_vs_baseline_pct",
        "prediction_bias",
    ]
    primary_view = feature_views[0]
    primary_set = feature_sets[0]
    primary_config_key = f"{primary_set} / {primary_view}"
    primary_metrics = metrics_df[
        (metrics_df["feature_view"] == primary_view)
        & (metrics_df["feature_set"] == primary_set)
    ]
    pooled_global = primary_metrics[
        (primary_metrics["experiment"] == "pooled_spatial_cv")
        & (primary_metrics["city"] == "__all__")
    ]
    loco_test = primary_metrics[
        (primary_metrics["experiment"] == "leave_one_city_out")
        & (primary_metrics["split"] == "test")
    ]

    summary = {
        "model": label,
        "cities": cities,
        "target_column": args.target_column,
        "target_view": target_view,
        "target_view_metadata": target_view_metadata,
        "n_rows": int(len(dataset)),
        "n_features": int(feature_config_metadata[primary_config_key]["n_features"]),
        "base_n_features": int(len(feature_columns)),
        "feature_views": feature_views,
        "feature_sets": feature_sets,
        "primary_feature_view": primary_view,
        "primary_feature_set": primary_set,
        "primary_feature_config": primary_config_key,
        "feature_config_metadata": feature_config_metadata,
        "n_spatial_splits": 1,
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
        "pooled_spatial_cv_summary_by_split": summarize_columns(
            pooled_global,
            "split",
            metric_cols,
        ),
        "leave_one_city_out_summary": summarize_columns(
            loco_test,
            "split",
            metric_cols,
        ),
        "leave_one_city_out_by_city": (
            string_key_records(loco_test.sort_values("city"))
        ),
        "feature_view_summary": feature_view_summary(metrics_df, metric_cols),
        "feature_config_summary": feature_config_summary(metrics_df, metric_cols),
        "top_features_by_experiment": importance_summary(
            importance_df[
                (importance_df["feature_view"] == primary_view)
                & (importance_df["feature_set"] == primary_set)
            ],
            ["experiment"],
        ),
        "top_features_by_feature_config": importance_summary(
            importance_df,
            ["feature_set", "feature_view", "experiment"],
        ),
        "artifacts": {
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
            "pooled_spatial_splits": str(splits_path),
            "feature_importance": str(importance_path),
        },
    }
    summary_path = metrics_dir / f"{prefix}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

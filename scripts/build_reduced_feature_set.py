"""Build an interpretable reduced feature set for thesis XGBoost experiments.

The goal is dimensionality reduction without PCA: keep original feature names,
remove redundant highly correlated predictors, and prioritize features that are
both domain-interpretable and empirically useful in the current XGBoost/SHAP
outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_multicity_xgboost import (  # noqa: E402
    build_feature_view,
    load_multicity_inputs,
)
from src.modeling.dataset import validate_model_feature_columns  # noqa: E402
from src.utils.cities import DEFAULT_CITIES, validate_cities  # noqa: E402


PROTECTED_FEATURES = (
    "pt_stop_density_per_km2",
    "pt_route_count",
    "pt_weekday_departures",
    "pt_peak_departures",
    "pt_peak_departure_share",
    "pt_avg_headway_min",
    "pt_nearest_stop_dist_m",
    "pt_stops_within_800m",
    "pt_routes_within_800m",
    "pt_departures_within_800m",
    "pt_stops_within_2000m",
    "pt_routes_within_2000m",
    "pt_departures_within_2000m",
    "pt_metro_peak_departures_within_2000m",
    "pt_metro_peak_departures_within_3000m",
    "pt_nearest_train_stop_dist_m",
    "pt_connected_blocks_same_route",
    "pt_service_span_hours",
    "be_block_area_m2",
    "be_compactness",
    "be_street_length_density_m_per_km2",
    "be_bikeable_street_share",
    "be_intersection_density_per_km2",
    "be_adjacency_degree",
    "be_major_road_density_m_per_km2",
    "be_estimated_avg_road_speed_kmh",
    "be_low_speed_street_share",
    "be_landuse_residential_share",
    "be_landuse_commercial_share",
    "be_landuse_green_share",
    "be_landuse_entropy",
    "be_building_footprint_share",
    "be_building_count_density_per_km2",
    "acs_population_density_per_km2",
    "acs_median_household_income",
    "acs_poverty_share",
    "acs_unemployment_share",
    "acs_zero_vehicle_household_share",
    "acs_bachelor_or_higher_share",
    "acs_age_under_18_share",
    "acs_age_65_plus_share",
    "acs_white_non_hispanic_share",
    "acs_black_share",
    "acs_hispanic_share",
    "acs_asian_share",
    "acs_avg_commute_time_min",
    "acs_car_commute_share",
    "acs_public_transit_commute_share",
    "acs_walk_commute_share",
    "acs_bike_commute_share",
    "acs_disability_share",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a correlation-pruned, interpretable reduced feature set."
    )
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    parser.add_argument("--interim-root", default="data/interim")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--target-column", default="Y_global_log_minmax")
    parser.add_argument(
        "--feature-view",
        default="log1p",
        choices=["raw", "log1p", "city_zscore"],
        help="Feature view used for variance/correlation pruning.",
    )
    parser.add_argument(
        "--shap-importance",
        default="outputs/tables/multicity_xgboost_shap_feature_importance.csv",
        help="Current SHAP feature importance CSV.",
    )
    parser.add_argument(
        "--gain-importance",
        default="outputs/tables/multicity_xgboost_main_baseline_feature_importance.csv",
        help="Current XGBoost feature importance CSV.",
    )
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--max-features", type=int, default=60)
    parser.add_argument("--min-pt", type=int, default=18)
    parser.add_argument("--max-pt", type=int, default=25)
    parser.add_argument("--min-be", type=int, default=10)
    parser.add_argument("--max-be", type=int, default=15)
    parser.add_argument("--min-acs", type=int, default=14)
    parser.add_argument("--max-acs", type=int, default=20)
    parser.add_argument(
        "--output-json",
        default="data/interim/modeling/reduced_feature_set.json",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/tables/reduced_feature_set.csv",
    )
    return parser.parse_args()


def feature_family(feature: str) -> str:
    if feature.startswith("pt_"):
        return "pt"
    if feature.startswith("be_"):
        return "be"
    if feature.startswith("acs_"):
        return "acs"
    return "other"


def minmax(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return pd.Series(0.0, index=values.index)
    return (values - lo) / (hi - lo)


def load_priority_scores(
    feature_columns: list[str],
    shap_path: str | Path,
    gain_path: str | Path,
) -> pd.DataFrame:
    base = pd.DataFrame({"feature": feature_columns})
    base["family"] = base["feature"].map(feature_family)
    base["protected"] = base["feature"].isin(PROTECTED_FEATURES)

    shap_file = Path(shap_path)
    if shap_file.exists():
        shap = pd.read_csv(shap_file)
        shap = shap[["feature", "mean_abs_shap"]].copy()
    else:
        shap = pd.DataFrame(columns=["feature", "mean_abs_shap"])

    gain_file = Path(gain_path)
    if gain_file.exists():
        gain = pd.read_csv(gain_file)
        gain = (
            gain.groupby("feature", as_index=False)["importance_gain"]
            .mean()
            .rename(columns={"importance_gain": "mean_gain"})
        )
    else:
        gain = pd.DataFrame(columns=["feature", "mean_gain"])

    out = base.merge(shap, on="feature", how="left").merge(gain, on="feature", how="left")
    out["mean_abs_shap"] = out["mean_abs_shap"].fillna(0.0).astype(float)
    out["mean_gain"] = out["mean_gain"].fillna(0.0).astype(float)
    out["shap_score"] = minmax(out["mean_abs_shap"])
    out["gain_score"] = minmax(out["mean_gain"])
    out["domain_bonus"] = out["protected"].astype(float) * 0.15
    out["priority"] = 0.75 * out["shap_score"] + 0.25 * out["gain_score"] + out["domain_bonus"]
    protected_order = {feature: i for i, feature in enumerate(PROTECTED_FEATURES)}
    out["protected_order"] = out["feature"].map(protected_order).fillna(1_000_000).astype(int)
    return out


def near_zero_variance_columns(dataset: pd.DataFrame, feature_columns: list[str]) -> set[str]:
    variances = dataset[feature_columns].astype(float).var(axis=0)
    return set(variances[variances <= 1e-12].index.astype(str))


def family_limits(args: argparse.Namespace) -> dict[str, tuple[int, int]]:
    return {
        "pt": (int(args.min_pt), int(args.max_pt)),
        "be": (int(args.min_be), int(args.max_be)),
        "acs": (int(args.min_acs), int(args.max_acs)),
    }


def select_uncorrelated_family(
    dataset: pd.DataFrame,
    priority: pd.DataFrame,
    family: str,
    min_count: int,
    max_count: int,
    threshold: float,
) -> list[str]:
    candidates = priority[priority["family"] == family].copy()
    candidates = candidates.sort_values(
        ["protected", "priority", "protected_order"],
        ascending=[False, False, True],
    )
    feature_names = candidates["feature"].tolist()
    if not feature_names:
        return []

    corr = dataset[feature_names].astype(float).corr().abs().fillna(0.0)
    selected: list[str] = []
    for feature in feature_names:
        if len(selected) >= max_count:
            break
        max_corr = float(corr.loc[feature, selected].max()) if selected else 0.0
        is_protected = bool(candidates.loc[candidates["feature"] == feature, "protected"].iloc[0])
        if max_corr < threshold or (is_protected and len(selected) < min_count):
            selected.append(feature)

    if len(selected) < min_count:
        for feature in feature_names:
            if feature not in selected:
                selected.append(feature)
            if len(selected) >= min_count:
                break

    return selected[:max_count]


def trim_to_max_features(
    selected: list[str],
    priority: pd.DataFrame,
    limits: dict[str, tuple[int, int]],
    max_features: int,
) -> list[str]:
    if len(selected) <= max_features:
        return selected

    priority_by_feature = priority.set_index("feature")
    work = selected.copy()
    while len(work) > max_features:
        removable: list[str] = []
        counts = pd.Series([feature_family(feature) for feature in work]).value_counts().to_dict()
        for feature in work:
            family = feature_family(feature)
            min_count, _ = limits.get(family, (0, len(work)))
            if counts.get(family, 0) > min_count and not bool(priority_by_feature.loc[feature, "protected"]):
                removable.append(feature)
        if not removable:
            break
        remove_feature = min(removable, key=lambda feature: float(priority_by_feature.loc[feature, "priority"]))
        work.remove(remove_feature)
    return work


def main() -> None:
    args = parse_args()
    cities = validate_cities([str(city) for city in args.cities])
    dataset, feature_columns, _, _ = load_multicity_inputs(
        cities=cities,
        interim_root=args.interim_root,
        target_column=args.target_column,
    )
    view_dataset, view_features, view_metadata = build_feature_view(
        dataset,
        feature_columns,
        str(args.feature_view),
    )
    validate_model_feature_columns(view_features)

    zero_variance = near_zero_variance_columns(view_dataset, view_features)
    eligible = [feature for feature in view_features if feature not in zero_variance]
    priority = load_priority_scores(eligible, args.shap_importance, args.gain_importance)

    limits = family_limits(args)
    selected: list[str] = []
    for family, (min_count, max_count) in limits.items():
        selected.extend(
            select_uncorrelated_family(
                dataset=view_dataset,
                priority=priority,
                family=family,
                min_count=min_count,
                max_count=max_count,
                threshold=float(args.correlation_threshold),
            )
        )

    selected = trim_to_max_features(
        selected=selected,
        priority=priority,
        limits=limits,
        max_features=int(args.max_features),
    )
    validate_model_feature_columns(selected)

    selected_set = set(selected)
    detail = priority.copy()
    detail["selected"] = detail["feature"].isin(selected_set)
    detail["selection_rank"] = detail["feature"].map(
        {feature: i + 1 for i, feature in enumerate(selected)}
    )
    detail = detail.sort_values(["selected", "family", "priority"], ascending=[False, True, False])

    payload: dict[str, Any] = {
        "feature_columns": selected,
        "n_features": int(len(selected)),
        "base_n_features": int(len(feature_columns)),
        "feature_view_for_pruning": str(args.feature_view),
        "feature_view_metadata": view_metadata,
        "correlation_threshold": float(args.correlation_threshold),
        "max_features": int(args.max_features),
        "family_counts": {
            family: int(sum(feature_family(feature) == family for feature in selected))
            for family in ["pt", "be", "acs"]
        },
        "zero_variance_excluded": sorted(zero_variance),
        "priority_sources": {
            "shap_importance": str(args.shap_importance),
            "gain_importance": str(args.gain_importance),
        },
        "method": (
            "Near-zero variance removal, family-wise correlation pruning, "
            "and SHAP/XGBoost/domain-prior ranking while preserving original feature names."
        ),
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    detail.to_csv(output_csv, index=False)

    summary = {
        **payload,
        "output_json": str(output_json),
        "output_csv": str(output_csv),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

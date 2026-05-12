import pandas as pd
import json

from scripts.evaluate_multicity_xgboost import (
    apply_target_view,
    build_global_target_view,
    build_feature_view,
    resolve_feature_sets,
    resolve_feature_views,
    select_feature_set,
)


def test_resolve_feature_views_expands_all_once() -> None:
    assert resolve_feature_views(["raw", "all"]) == [
        "raw",
        "log1p",
        "city_zscore",
        "raw_plus_city_context",
        "city_zscore_plus_city_context",
    ]


def test_resolve_feature_sets_expands_ablation_once() -> None:
    assert resolve_feature_sets(["pt", "all_ablation"]) == [
        "full",
        "pt",
        "be",
        "acs",
        "pt_be",
        "full_no_acs_commute",
    ]


def test_select_feature_set_excludes_acs_commute_behavior_only() -> None:
    columns = [
        "pt_stop_count",
        "be_street_length_density_m_per_km2",
        "acs_population_density_per_km2",
        "acs_zero_vehicle_household_share",
        "acs_public_transit_commute_share",
        "acs_avg_commute_time_min",
    ]

    selected, metadata = select_feature_set(columns, "full_no_acs_commute")

    assert "acs_public_transit_commute_share" not in selected
    assert "acs_avg_commute_time_min" not in selected
    assert "acs_zero_vehicle_household_share" in selected
    assert metadata["n_features"] == 4


def test_select_feature_set_loads_reduced_allowlist(tmp_path) -> None:
    path = tmp_path / "reduced_feature_set.json"
    path.write_text(
        json.dumps({"feature_columns": ["pt_stop_count", "be_block_area_m2"]}),
        encoding="utf-8",
    )
    selected, metadata = select_feature_set(
        ["pt_stop_count", "be_block_area_m2", "acs_population"],
        "reduced",
        path,
    )

    assert selected == ["pt_stop_count", "be_block_area_m2"]
    assert metadata["n_features"] == 2
    assert metadata["reduced_feature_set_path"] == str(path)


def test_build_global_target_view_uses_cross_city_scale(tmp_path) -> None:
    target_components = pd.DataFrame(
        {
            "node_id": ["a:1", "a:2", "b:1", "b:2"],
            "city": ["a", "a", "b", "b"],
            "block_id": ["1", "2", "1", "2"],
            "jobs_1km": [0.0, 10.0, 100.0, 1000.0],
            "amen_food_1km": [0.0, 1.0, 2.0, 10.0],
        }
    )

    out, metadata = build_global_target_view(
        target_components,
        cities=["a", "b"],
        interim_root=tmp_path,
        target_view="global_minmax",
    )

    assert metadata["target_normalization_scope"] == "selected_cities_global"
    assert out["target_value"].min() == 0.0
    assert out["target_value"].max() == 1.0
    assert out.loc[out["node_id"] == "b:2", "target_value"].iloc[0] == 1.0


def test_apply_target_view_city_rank_replaces_label_with_percentiles() -> None:
    dataset = pd.DataFrame(
        {
            "node_id": ["a:1", "a:2", "b:1", "b:2"],
            "city": ["a", "a", "b", "b"],
            "Y": [0.1, 0.9, 10.0, 20.0],
        }
    )

    out, metadata = apply_target_view(
        dataset,
        cities=["a", "b"],
        interim_root="unused",
        target_column="Y",
        target_view="city_rank",
    )

    assert metadata["target_view"] == "city_rank"
    assert out["Y"].tolist() == [0.5, 1.0, 0.5, 1.0]


def test_build_feature_view_city_zscore_normalizes_within_city() -> None:
    dataset = pd.DataFrame(
        {
            "city": ["a", "a", "b", "b"],
            "pt_stop_count": [1.0, 3.0, 10.0, 14.0],
            "acs_poverty_share": [0.1, 0.2, 0.3, 0.4],
        }
    )

    out, feature_columns, metadata = build_feature_view(
        dataset,
        ["pt_stop_count", "acs_poverty_share"],
        "city_zscore",
    )

    assert feature_columns == ["pt_stop_count", "acs_poverty_share"]
    assert metadata["feature_view"] == "city_zscore"
    means = out.groupby("city")["pt_stop_count"].mean().round(12)
    assert means.to_dict() == {"a": 0.0, "b": 0.0}


def test_build_feature_view_adds_city_context_without_target_columns() -> None:
    dataset = pd.DataFrame(
        {
            "city": ["a", "a", "b"],
            "pt_stop_density_per_km2": [1.0, 3.0, 10.0],
            "be_bikeable_street_share": [0.1, 0.2, 0.3],
            "Y": [0.2, 0.3, 0.4],
        }
    )

    out, feature_columns, metadata = build_feature_view(
        dataset,
        ["pt_stop_density_per_km2", "be_bikeable_street_share"],
        "raw_plus_city_context",
    )

    assert "Y" not in feature_columns
    assert "city_mean__pt_stop_density_per_km2" in feature_columns
    assert "city_std__pt_stop_density_per_km2" in feature_columns
    assert metadata["city_context_columns"]
    assert out.loc[0, "city_mean__pt_stop_density_per_km2"] == 2.0

import pytest
import pandas as pd

from scripts.build_city_features import validate_feature_columns
from src.features.gtfs_features import (
	PT_CATCHMENT_RADII_M,
	PT_MODE_NAMES,
	PT_NEAREST_FALLBACK_DIST_M,
	_empty_pt_features,
	_route_mode,
)


def test_validate_feature_columns_rejects_target_ingredients():
	with pytest.raises(ValueError):
		validate_feature_columns(["pt_stop_count", "jobs_1km"])

	with pytest.raises(ValueError):
		validate_feature_columns(["be_street_length_m", "amen_food_retail_1km"])

	with pytest.raises(ValueError):
		validate_feature_columns(["pt_route_count", "Y"])


def test_validate_feature_columns_accepts_pt_be_features():
	validate_feature_columns(
		[
			"pt_stop_count",
			"pt_weekday_departures",
			"be_street_length_m",
			"be_intersection_density_per_km2",
			"acs_median_household_income",
		]
	)


def test_empty_pt_features_include_wide_context_catchments():
	features = _empty_pt_features(pd.Series(["a"]))

	assert PT_CATCHMENT_RADII_M == (400, 800, 1200, 2000, 3000)
	for radius in [2000, 3000]:
		assert f"pt_stops_within_{radius}m" in features.columns
		assert f"pt_routes_within_{radius}m" in features.columns
		assert f"pt_departures_within_{radius}m" in features.columns
		assert f"pt_peak_departures_within_{radius}m" in features.columns
		for mode in PT_MODE_NAMES:
			assert f"pt_{mode}_stops_within_{radius}m" in features.columns
			assert f"pt_{mode}_routes_within_{radius}m" in features.columns
			assert f"pt_{mode}_departures_within_{radius}m" in features.columns

	for mode in PT_MODE_NAMES:
		assert f"pt_has_{mode}_service" in features.columns
		assert f"pt_{mode}_stop_count" in features.columns
		assert f"pt_{mode}_route_count" in features.columns
		assert f"pt_nearest_{mode}_stop_dist_m" in features.columns
		assert features.loc[0, f"pt_nearest_{mode}_stop_dist_m"] == PT_NEAREST_FALLBACK_DIST_M
	assert features.loc[0, "pt_nearest_stop_dist_m"] == PT_NEAREST_FALLBACK_DIST_M
	for column in [
		"pt_peak_departure_share",
		"pt_departures_per_stop",
		"pt_routes_per_stop",
		"pt_peak_departures_per_route",
		"pt_first_departure_min",
		"pt_last_departure_min",
		"pt_service_span_hours",
		"pt_nearest_stop_departures",
		"pt_nearest_stop_peak_departures",
		"pt_nearest_stop_route_count",
		"pt_connected_blocks_same_route",
	]:
		assert column in features.columns
	for radius in PT_CATCHMENT_RADII_M:
		assert f"pt_departures_per_stop_within_{radius}m" in features.columns
		assert f"pt_routes_per_stop_within_{radius}m" in features.columns
		assert f"pt_peak_departure_share_within_{radius}m" in features.columns


def test_route_mode_maps_basic_and_extended_gtfs_types():
	modes = _route_mode(pd.Series(["0", "1", "2", "3", "11", "700", "900", "9999"]))

	assert modes.tolist() == [
		"tram",
		"metro",
		"train",
		"bus",
		"bus",
		"bus",
		"tram",
		"other",
	]

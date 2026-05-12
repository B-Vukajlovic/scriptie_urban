"""Leakage-safe public transport feature construction from GTFS feeds."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


PEAK_WINDOWS_MINUTES = ((7 * 60, 9 * 60), (16 * 60, 18 * 60))
PT_CATCHMENT_RADII_M = (400, 800, 1200, 2000, 3000)
PT_MODE_NAMES = ("bus", "tram", "metro", "train")
PT_NEAREST_FALLBACK_DIST_M = 10_000.0


def _read_gtfs_table(
    feed_path: Path,
    table_name: str,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    with ZipFile(feed_path) as zf:
        if table_name not in zf.namelist():
            return pd.DataFrame()
        if usecols is not None:
            with zf.open(table_name) as f:
                available = pd.read_csv(f, nrows=0).columns.tolist()
            present_usecols = [col for col in usecols if col in available]
            if not present_usecols:
                return pd.DataFrame()
            usecols = present_usecols
        with zf.open(table_name) as f:
            return pd.read_csv(f, usecols=usecols, dtype="string")


def _parse_gtfs_time_to_minutes(times: pd.Series) -> pd.Series:
    parts = times.fillna("").str.split(":", expand=True)
    valid = parts.shape[1] >= 3
    if not valid:
        return pd.Series(np.nan, index=times.index, dtype=float)

    hours = pd.to_numeric(parts[0], errors="coerce")
    minutes = pd.to_numeric(parts[1], errors="coerce")
    seconds = pd.to_numeric(parts[2], errors="coerce")
    return hours * 60 + minutes + seconds / 60


def _active_weekday_service_ids(calendar: pd.DataFrame, trips: pd.DataFrame) -> set[str]:
    if calendar.empty:
        return set(trips["service_id"].dropna().astype(str))

    weekday_cols = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    missing = [col for col in weekday_cols if col not in calendar.columns]
    if missing:
        return set(trips["service_id"].dropna().astype(str))

    work = calendar.copy()
    for col in weekday_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)
    active = work[work[weekday_cols].sum(axis=1) > 0]
    return set(active["service_id"].dropna().astype(str))


def _is_peak(minutes: pd.Series) -> pd.Series:
    peak = pd.Series(False, index=minutes.index)
    minute_of_day = minutes % (24 * 60)
    for start, end in PEAK_WINDOWS_MINUTES:
        peak |= (minute_of_day >= start) & (minute_of_day < end)
    return peak


def _route_mode(route_type: pd.Series) -> pd.Series:
    """Map GTFS route_type values into thesis-facing transit mode groups."""
    numeric = pd.to_numeric(route_type, errors="coerce")
    mode = pd.Series("other", index=route_type.index, dtype="object")

    mode.loc[numeric.isin([3, 11]) | ((numeric >= 700) & (numeric < 800))] = "bus"
    mode.loc[numeric.isin([0, 5]) | ((numeric >= 900) & (numeric < 1000))] = "tram"
    mode.loc[numeric.eq(1) | ((numeric >= 400) & (numeric < 500))] = "metro"
    mode.loc[numeric.isin([2, 12]) | ((numeric >= 100) & (numeric < 200))] = "train"
    return mode


def _empty_pt_features(block_ids: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "block_id": block_ids.astype(str),
            "pt_stop_count": 0.0,
            "pt_stop_density_per_km2": 0.0,
            "pt_route_count": 0.0,
            "pt_weekday_departures": 0.0,
            "pt_peak_departures": 0.0,
            "pt_offpeak_departures": 0.0,
            "pt_peak_departure_share": 0.0,
            "pt_avg_headway_min": 1440.0,
            "pt_first_departure_min": 1440.0,
            "pt_last_departure_min": 0.0,
            "pt_service_span_hours": 0.0,
            "pt_modal_variety": 0.0,
            "pt_nearest_stop_dist_m": PT_NEAREST_FALLBACK_DIST_M,
            "pt_connected_blocks_same_route": 0.0,
            **{
                f"pt_has_{mode}_service": 0.0
                for mode in PT_MODE_NAMES
            },
            **{
                f"pt_{mode}_stop_count": 0.0
                for mode in PT_MODE_NAMES
            },
            **{
                f"pt_{mode}_stop_density_per_km2": 0.0
                for mode in PT_MODE_NAMES
            },
            **{
                f"pt_{mode}_route_count": 0.0
                for mode in PT_MODE_NAMES
            },
            **{
                f"pt_{mode}_weekday_departures": 0.0
                for mode in PT_MODE_NAMES
            },
            **{
                f"pt_{mode}_peak_departures": 0.0
                for mode in PT_MODE_NAMES
            },
            **{
                f"pt_nearest_{mode}_stop_dist_m": PT_NEAREST_FALLBACK_DIST_M
                for mode in PT_MODE_NAMES
            },
            **{
                f"pt_stops_within_{radius}m": 0.0
                for radius in PT_CATCHMENT_RADII_M
            },
            **{
                f"pt_routes_within_{radius}m": 0.0
                for radius in PT_CATCHMENT_RADII_M
            },
            **{
                f"pt_departures_within_{radius}m": 0.0
                for radius in PT_CATCHMENT_RADII_M
            },
            **{
                f"pt_peak_departures_within_{radius}m": 0.0
                for radius in PT_CATCHMENT_RADII_M
            },
            **{
                f"pt_{mode}_stops_within_{radius}m": 0.0
                for mode in PT_MODE_NAMES
                for radius in PT_CATCHMENT_RADII_M
            },
            **{
                f"pt_{mode}_routes_within_{radius}m": 0.0
                for mode in PT_MODE_NAMES
                for radius in PT_CATCHMENT_RADII_M
            },
            **{
                f"pt_{mode}_departures_within_{radius}m": 0.0
                for mode in PT_MODE_NAMES
                for radius in PT_CATCHMENT_RADII_M
            },
            **{
                f"pt_{mode}_peak_departures_within_{radius}m": 0.0
                for mode in PT_MODE_NAMES
                for radius in PT_CATCHMENT_RADII_M
            },
        }
    )


def build_gtfs_features(
    blocks: gpd.GeoDataFrame,
    centroids: gpd.GeoDataFrame,
    gtfs_city_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Build block-level PT supply features from all GTFS zips in a city folder."""
    required = {"block_id", "geometry"}
    missing = required - set(blocks.columns)
    if missing:
        raise KeyError(f"Missing required block columns: {sorted(missing)}")

    gtfs_files = sorted(Path(gtfs_city_dir).glob("*.zip"))
    block_ids = blocks["block_id"].astype(str)
    if not gtfs_files:
        return (
            _empty_pt_features(block_ids),
            {
                "n_gtfs_feeds": 0,
                "n_active_physical_stops": 0,
                "n_active_stops_inside_blocks": 0,
            },
        )

    active_stops_all: list[gpd.GeoDataFrame] = []
    stop_service_events_all: list[pd.DataFrame] = []
    stop_events_all: list[pd.DataFrame] = []
    stop_headways_all: list[pd.DataFrame] = []
    n_active_physical_stops = 0

    for feed_idx, feed_path in enumerate(gtfs_files):
        feed_prefix = f"feed{feed_idx}"
        stops = _read_gtfs_table(
            feed_path,
            "stops.txt",
            usecols=["stop_id", "stop_lat", "stop_lon", "location_type"],
        )
        routes = _read_gtfs_table(
            feed_path,
            "routes.txt",
            usecols=["route_id", "route_type"],
        )
        trips = _read_gtfs_table(
            feed_path,
            "trips.txt",
            usecols=["route_id", "service_id", "trip_id"],
        )
        stop_times = _read_gtfs_table(
            feed_path,
            "stop_times.txt",
            usecols=["trip_id", "departure_time", "stop_id"],
        )
        calendar = _read_gtfs_table(feed_path, "calendar.txt")
        if stops.empty or trips.empty or stop_times.empty:
            continue
        required_stop_cols = {"stop_id", "stop_lat", "stop_lon"}
        if required_stop_cols - set(stops.columns):
            continue

        active_service_ids = _active_weekday_service_ids(calendar, trips)
        active_trips = trips[trips["service_id"].astype(str).isin(active_service_ids)].copy()
        if active_trips.empty:
            continue

        active_trips["trip_key"] = feed_prefix + ":" + active_trips["trip_id"].astype(str)
        active_trips["route_key"] = feed_prefix + ":" + active_trips["route_id"].astype(str)
        routes["route_key"] = feed_prefix + ":" + routes["route_id"].astype(str)

        stop_times = stop_times.merge(
            active_trips[["trip_id", "route_key"]],
            on="trip_id",
            how="inner",
        )
        if stop_times.empty:
            continue

        stop_times["stop_key"] = feed_prefix + ":" + stop_times["stop_id"].astype(str)
        stop_times["departure_min"] = _parse_gtfs_time_to_minutes(stop_times["departure_time"])
        stop_times = stop_times.dropna(subset=["departure_min"])
        stop_times["is_peak"] = _is_peak(stop_times["departure_min"])

        stops["stop_key"] = feed_prefix + ":" + stops["stop_id"].astype(str)
        stops = stops[stops["stop_key"].isin(stop_times["stop_key"])].copy()
        stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
        stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
        stops = stops.dropna(subset=["stop_lat", "stop_lon"])
        if "location_type" in stops.columns:
            location_type = pd.to_numeric(stops["location_type"], errors="coerce").fillna(0)
            stops = stops[location_type == 0].copy()
        if stops.empty:
            continue

        stop_points = gpd.GeoDataFrame(
            stops[["stop_key", "stop_lat", "stop_lon"]].copy(),
            geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
            crs="EPSG:4326",
        ).to_crs(blocks.crs)
        stop_points["physical_stop_key"] = (
            stop_points.geometry.x.round(1).astype(str)
            + ":"
            + stop_points.geometry.y.round(1).astype(str)
        )
        active_stops_all.append(stop_points)

        service_events = stop_times.merge(
            stop_points[["stop_key", "physical_stop_key"]],
            on="stop_key",
            how="inner",
        )
        service_events = service_events.merge(
            routes[["route_key", "route_type"]],
            on="route_key",
            how="left",
        )
        service_events["route_mode"] = _route_mode(service_events["route_type"])
        if not service_events.empty:
            stop_service_events_all.append(service_events)

        stop_block = gpd.sjoin(
            stop_points[["stop_key", "physical_stop_key", "geometry"]],
            blocks[["block_id", "geometry"]],
            how="inner",
            predicate="within",
        )[["stop_key", "physical_stop_key", "block_id"]]

        events = stop_times.merge(stop_block, on="stop_key", how="inner")
        if events.empty:
            continue
        events = events.merge(routes[["route_key", "route_type"]], on="route_key", how="left")
        events["route_mode"] = _route_mode(events["route_type"])
        stop_events_all.append(events)

        headways = (
            events.sort_values(["physical_stop_key", "departure_min"])
            .groupby("physical_stop_key")["departure_min"]
            .diff()
        )
        headway_df = events[["physical_stop_key", "block_id"]].copy()
        headway_df["headway_min"] = headways
        headway_df = headway_df[
            (headway_df["headway_min"] > 0)
            & (headway_df["headway_min"] <= 240)
        ]
        stop_headways_all.append(headway_df)

    features = _empty_pt_features(block_ids)
    block_area_km2 = blocks.set_index("block_id").geometry.area / 1_000_000

    if active_stops_all:
        active_stops = pd.concat(active_stops_all, ignore_index=True)
        active_stops_gdf = gpd.GeoDataFrame(active_stops, geometry="geometry", crs=blocks.crs)
        stop_block = gpd.sjoin(
            active_stops_gdf[["physical_stop_key", "geometry"]],
            blocks[["block_id", "geometry"]],
            how="inner",
            predicate="within",
        )
        stop_counts = stop_block.groupby("block_id")["physical_stop_key"].nunique()
        features = features.set_index("block_id")
        features["pt_stop_count"] = stop_counts.reindex(features.index).fillna(0).astype(float)
        area = block_area_km2.reindex(features.index).replace(0, np.nan)
        features["pt_stop_density_per_km2"] = (features["pt_stop_count"] / area).fillna(0.0)

        centroid_xy = np.column_stack(
            [centroids.geometry.x.to_numpy(), centroids.geometry.y.to_numpy()]
        )
        stop_xy = np.column_stack(
            [
                active_stops_gdf.geometry.x.to_numpy(),
                active_stops_gdf.geometry.y.to_numpy(),
            ]
        )
        if len(stop_xy):
            dists, _ = KDTree(stop_xy).query(centroid_xy)
            nearest = pd.Series(
                np.minimum(dists, PT_NEAREST_FALLBACK_DIST_M),
                index=centroids["block_id"].astype(str),
            )
            features["pt_nearest_stop_dist_m"] = nearest.reindex(features.index)

            unique_stops = active_stops_gdf.drop_duplicates("physical_stop_key").copy()
            unique_stops = unique_stops.reset_index(drop=True)
            n_active_physical_stops = int(len(unique_stops))
            unique_stop_xy = np.column_stack(
                [unique_stops.geometry.x.to_numpy(), unique_stops.geometry.y.to_numpy()]
            )
            unique_stop_keys = unique_stops["physical_stop_key"].astype(str).tolist()
            service_by_stop = _stop_service_summary(
                stop_service_events_all,
                unique_stop_keys,
            )
            stop_route_sets = _stop_route_sets(stop_service_events_all, unique_stop_keys)
            service_by_stop_mode = _stop_service_summary_by_mode(
                stop_service_events_all,
                unique_stop_keys,
            )
            stop_route_sets_by_mode = _stop_route_sets_by_mode(
                stop_service_events_all,
                unique_stop_keys,
            )
            stop_tree = KDTree(unique_stop_xy)
            centroid_ids = centroids["block_id"].astype(str).tolist()

            for mode in PT_MODE_NAMES:
                mode_service = service_by_stop_mode[mode]
                mode_stop_mask = mode_service["departures"].to_numpy(dtype=float) > 0
                features[f"pt_has_{mode}_service"] = float(mode_stop_mask.any())
                if mode_stop_mask.any():
                    mode_tree = KDTree(unique_stop_xy[mode_stop_mask])
                    mode_dists, _ = mode_tree.query(centroid_xy)
                    nearest_mode = pd.Series(
                        np.minimum(mode_dists, PT_NEAREST_FALLBACK_DIST_M),
                        index=centroid_ids,
                    )
                    features[f"pt_nearest_{mode}_stop_dist_m"] = nearest_mode.reindex(
                        features.index
                    )

            for radius in PT_CATCHMENT_RADII_M:
                neighbor_lists = stop_tree.query_ball_point(centroid_xy, r=radius)
                catchment = _aggregate_catchment_service(
                    block_ids=centroid_ids,
                    neighbor_lists=neighbor_lists,
                    stop_service=service_by_stop,
                    stop_route_sets=stop_route_sets,
                )
                catchment = catchment.set_index("block_id").reindex(features.index)
                for metric in [
                    "stops",
                    "routes",
                    "departures",
                    "peak_departures",
                ]:
                    features[f"pt_{metric}_within_{radius}m"] = (
                        catchment[metric].fillna(0.0).astype(float)
                    )

                for mode in PT_MODE_NAMES:
                    mode_catchment = _aggregate_catchment_service(
                        block_ids=centroid_ids,
                        neighbor_lists=neighbor_lists,
                        stop_service=service_by_stop_mode[mode],
                        stop_route_sets=stop_route_sets_by_mode[mode],
                    )
                    mode_catchment = mode_catchment.set_index("block_id").reindex(
                        features.index
                    )
                    for metric in [
                        "stops",
                        "routes",
                        "departures",
                        "peak_departures",
                    ]:
                        features[f"pt_{mode}_{metric}_within_{radius}m"] = (
                            mode_catchment[metric].fillna(0.0).astype(float)
                        )
        features = features.copy().reset_index()

    if stop_events_all:
        events = pd.concat(stop_events_all, ignore_index=True)
        event_group = events.groupby("block_id")
        features = features.set_index("block_id")
        features["pt_route_count"] = (
            event_group["route_key"].nunique().reindex(features.index).fillna(0).astype(float)
        )
        features["pt_weekday_departures"] = (
            event_group.size().reindex(features.index).fillna(0).astype(float)
        )
        peak = events[events["is_peak"]].groupby("block_id").size()
        features["pt_peak_departures"] = peak.reindex(features.index).fillna(0).astype(float)
        features["pt_offpeak_departures"] = (
            features["pt_weekday_departures"] - features["pt_peak_departures"]
        )
        features["pt_peak_departure_share"] = (
            features["pt_peak_departures"]
            / features["pt_weekday_departures"].replace(0, np.nan)
        ).fillna(0.0)
        first_departure = event_group["departure_min"].min()
        last_departure = event_group["departure_min"].max()
        features["pt_first_departure_min"] = (
            first_departure.reindex(features.index).fillna(1440.0).astype(float)
        )
        features["pt_last_departure_min"] = (
            last_departure.reindex(features.index).fillna(0.0).astype(float)
        )
        features["pt_service_span_hours"] = (
            (features["pt_last_departure_min"] - features["pt_first_departure_min"])
            .clip(lower=0.0)
            / 60.0
        )
        features["pt_modal_variety"] = (
            event_group["route_type"].nunique().reindex(features.index).fillna(0).astype(float)
        )
        features["pt_connected_blocks_same_route"] = _connected_blocks_same_route(
            events=events,
            block_ids=features.index.astype(str).tolist(),
        )
        area = block_area_km2.reindex(features.index).replace(0, np.nan)
        for mode in PT_MODE_NAMES:
            mode_events = events[events["route_mode"] == mode].copy()
            if mode_events.empty:
                continue
            mode_group = mode_events.groupby("block_id")
            stop_count = (
                mode_group["physical_stop_key"]
                .nunique()
                .reindex(features.index)
                .fillna(0)
                .astype(float)
            )
            features[f"pt_{mode}_stop_count"] = stop_count
            features[f"pt_{mode}_stop_density_per_km2"] = (stop_count / area).fillna(0.0)
            features[f"pt_{mode}_route_count"] = (
                mode_group["route_key"].nunique().reindex(features.index).fillna(0).astype(float)
            )
            features[f"pt_{mode}_weekday_departures"] = (
                mode_group.size().reindex(features.index).fillna(0).astype(float)
            )
            mode_peak = mode_events[mode_events["is_peak"]].groupby("block_id").size()
            features[f"pt_{mode}_peak_departures"] = (
                mode_peak.reindex(features.index).fillna(0).astype(float)
            )
        features = features.copy().reset_index()

    if stop_headways_all:
        headways = pd.concat(stop_headways_all, ignore_index=True)
        if not headways.empty:
            stop_median = (
                headways.groupby(["block_id", "physical_stop_key"])["headway_min"]
                .median()
                .reset_index()
            )
            block_headway = stop_median.groupby("block_id")["headway_min"].mean()
            features = features.set_index("block_id")
            features["pt_avg_headway_min"] = (
                block_headway.reindex(features.index).fillna(1440.0).astype(float)
            )
            features = features.copy().reset_index()

    metadata = {
        "n_gtfs_feeds": len(gtfs_files),
        "n_active_physical_stops": n_active_physical_stops,
        "n_active_stops_inside_blocks": int(features["pt_stop_count"].sum()),
        "n_blocks_with_stops": int((features["pt_stop_count"] > 0).sum()),
        "n_blocks_with_departures": int((features["pt_weekday_departures"] > 0).sum()),
        "median_nearest_stop_dist_m": float(features["pt_nearest_stop_dist_m"].median()),
        "catchment_radii_m": list(PT_CATCHMENT_RADII_M),
        "n_blocks_with_stop_within_800m": int(
            (features["pt_stops_within_800m"] > 0).sum()
        ),
        "mode_groups": list(PT_MODE_NAMES),
        "nearest_stop_fallback_dist_m": PT_NEAREST_FALLBACK_DIST_M,
        "n_blocks_with_bus_stop_within_800m": int(
            (features["pt_bus_stops_within_800m"] > 0).sum()
        ),
        "n_blocks_with_rail_mode_stop_within_800m": int(
            (
                features["pt_tram_stops_within_800m"]
                + features["pt_metro_stops_within_800m"]
                + features["pt_train_stops_within_800m"]
            ).gt(0).sum()
        ),
    }
    return features, metadata


def _stop_service_summary(
    stop_service_events_all: list[pd.DataFrame],
    physical_stop_keys: list[str],
) -> pd.DataFrame:
    """Aggregate GTFS service intensity at deduplicated physical stop locations."""
    index = pd.Index(physical_stop_keys, name="physical_stop_key")
    summary = pd.DataFrame(index=index)
    summary["departures"] = 0.0
    summary["peak_departures"] = 0.0

    if not stop_service_events_all:
        return summary

    events = pd.concat(stop_service_events_all, ignore_index=True)
    events["physical_stop_key"] = events["physical_stop_key"].astype(str)
    departures = events.groupby("physical_stop_key").size()
    peak = events[events["is_peak"]].groupby("physical_stop_key").size()
    summary["departures"] = departures.reindex(index).fillna(0.0).astype(float)
    summary["peak_departures"] = peak.reindex(index).fillna(0.0).astype(float)
    return summary


def _stop_service_summary_by_mode(
    stop_service_events_all: list[pd.DataFrame],
    physical_stop_keys: list[str],
) -> dict[str, pd.DataFrame]:
    """Aggregate stop service intensity separately for each route mode."""
    if not stop_service_events_all:
        return {
            mode: _stop_service_summary([], physical_stop_keys)
            for mode in PT_MODE_NAMES
        }

    events = pd.concat(stop_service_events_all, ignore_index=True)
    out: dict[str, pd.DataFrame] = {}
    for mode in PT_MODE_NAMES:
        out[mode] = _stop_service_summary(
            [events[events["route_mode"] == mode].copy()],
            physical_stop_keys,
        )
    return out


def _stop_route_sets(
    stop_service_events_all: list[pd.DataFrame],
    physical_stop_keys: list[str],
) -> list[set[str]]:
    """Build route sets per physical stop for catchment-level unique route counts."""
    route_sets = {key: set() for key in physical_stop_keys}
    if stop_service_events_all:
        events = pd.concat(stop_service_events_all, ignore_index=True)
        events["physical_stop_key"] = events["physical_stop_key"].astype(str)
        events["route_key"] = events["route_key"].astype(str)
        grouped = events.groupby("physical_stop_key")["route_key"].unique()
        for stop_key, route_keys in grouped.items():
            if stop_key in route_sets:
                route_sets[stop_key] = set(route_keys)
    return [route_sets[key] for key in physical_stop_keys]


def _stop_route_sets_by_mode(
    stop_service_events_all: list[pd.DataFrame],
    physical_stop_keys: list[str],
) -> dict[str, list[set[str]]]:
    """Build mode-specific route sets per physical stop."""
    if not stop_service_events_all:
        return {
            mode: [set() for _ in physical_stop_keys]
            for mode in PT_MODE_NAMES
        }

    events = pd.concat(stop_service_events_all, ignore_index=True)
    return {
        mode: _stop_route_sets(
            [events[events["route_mode"] == mode].copy()],
            physical_stop_keys,
        )
        for mode in PT_MODE_NAMES
    }


def _aggregate_catchment_service(
    block_ids: list[str],
    neighbor_lists: list[list[int]],
    stop_service: pd.DataFrame,
    stop_route_sets: list[set[str]],
) -> pd.DataFrame:
    """Summarize stop and service supply reachable within one catchment radius."""
    rows: list[dict[str, float | str]] = []
    departures = stop_service["departures"].to_numpy(dtype=float)
    peak_departures = stop_service["peak_departures"].to_numpy(dtype=float)

    for block_id, stop_idxs in zip(block_ids, neighbor_lists):
        if not stop_idxs:
            rows.append(
                {
                    "block_id": block_id,
                    "stops": 0.0,
                    "routes": 0.0,
                    "departures": 0.0,
                    "peak_departures": 0.0,
                }
            )
            continue

        route_union: set[str] = set()
        for stop_idx in stop_idxs:
            route_union.update(stop_route_sets[stop_idx])

        idx = np.array(stop_idxs, dtype=int)
        rows.append(
            {
                "block_id": block_id,
                "stops": float((departures[idx] > 0).sum()),
                "routes": float(len(route_union)),
                "departures": float(departures[idx].sum()),
                "peak_departures": float(peak_departures[idx].sum()),
            }
        )

    return pd.DataFrame(rows)


def _connected_blocks_same_route(events: pd.DataFrame, block_ids: list[str]) -> pd.Series:
    """Count blocks sharing at least one direct transit route with each source block."""
    if events.empty:
        return pd.Series(0.0, index=pd.Index(block_ids, dtype="object"))

    route_blocks = events.groupby("route_key")["block_id"].agg(lambda values: set(values.astype(str)))
    block_routes = events.groupby("block_id")["route_key"].agg(lambda values: set(values.astype(str)))
    route_block_map = route_blocks.to_dict()
    counts: dict[str, float] = {}
    for block_id in block_ids:
        connected: set[str] = set()
        for route in block_routes.get(block_id, set()):
            connected.update(route_block_map.get(route, set()))
        connected.discard(block_id)
        counts[block_id] = float(len(connected))
    return pd.Series(counts).reindex(block_ids).fillna(0.0).astype(float)

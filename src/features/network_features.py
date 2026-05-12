"""Leakage-safe built-environment feature construction from block and street geometry."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src.data.load_osm import extract_osm_tag
from src.target.osm_network_graph import build_road_graph


STREET_HIGHWAY_TYPES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "residential",
    "living_street",
    "unclassified",
    "service",
    "cycleway",
    "path",
}

BIKEABLE_HIGHWAY_TYPES = {
    "residential",
    "living_street",
    "unclassified",
    "service",
    "tertiary",
    "tertiary_link",
    "cycleway",
    "path",
}

MAJOR_HIGHWAY_TYPES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
}

LOCAL_HIGHWAY_TYPES = {
    "residential",
    "living_street",
    "unclassified",
    "service",
}

LANDUSE_GROUPS = {
    "residential": {"residential"},
    "commercial": {"commercial", "office"},
    "retail": {"retail"},
    "industrial": {"industrial", "railway", "quarry", "landfill"},
    "civic": {"education", "religious", "cemetery", "military", "government"},
    "green": {"grass", "forest", "meadow", "recreation_ground", "park", "village_green"},
}

DEFAULT_SPEED_KMH = {
    "motorway": 105.0,
    "motorway_link": 65.0,
    "trunk": 90.0,
    "trunk_link": 55.0,
    "primary": 65.0,
    "primary_link": 45.0,
    "secondary": 55.0,
    "secondary_link": 40.0,
    "tertiary": 45.0,
    "tertiary_link": 35.0,
    "residential": 30.0,
    "living_street": 15.0,
    "unclassified": 35.0,
    "service": 20.0,
    "cycleway": 15.0,
    "path": 10.0,
}


def _primary_highway(highway: pd.Series) -> pd.Series:
    return highway.fillna("").astype(str).str.split(";").str[0].str.lower().str.strip()


def load_street_lines_in_bbox(
    osm_pbf_path: str | Path,
    bbox_wgs84: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    """Load street-like OSM lines within a bbox from a local PBF file."""
    gdf = gpd.read_file(osm_pbf_path, layer="lines", bbox=bbox_wgs84)
    if gdf.empty:
        return gpd.GeoDataFrame(
            columns=["highway", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
    if "highway" not in gdf.columns:
        return gpd.GeoDataFrame(columns=["highway", "geometry"], geometry="geometry", crs=gdf.crs)

    keep_cols = ["highway", "geometry"]
    if "maxspeed" in gdf.columns:
        keep_cols.append("maxspeed")
    if "other_tags" in gdf.columns:
        keep_cols.append("other_tags")
    out = gdf[keep_cols].copy()
    out["highway_type"] = _primary_highway(out["highway"])
    if "maxspeed" not in out.columns:
        other_tags = (
            out["other_tags"]
            if "other_tags" in out.columns
            else pd.Series("", index=out.index, dtype="object")
        )
        out["maxspeed"] = extract_osm_tag(other_tags, "maxspeed")
    out = out[out["highway_type"].isin(STREET_HIGHWAY_TYPES)].copy()
    return out


def _parse_speed_kmh(maxspeed: pd.Series, highway_type: pd.Series) -> pd.Series:
    raw = maxspeed.fillna("").astype(str).str.lower()
    number = pd.to_numeric(raw.str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce")
    mph_mask = raw.str.contains("mph", na=False)
    speed = number.where(~mph_mask, number * 1.60934)
    defaults = highway_type.map(DEFAULT_SPEED_KMH).astype(float)
    return speed.fillna(defaults).fillna(30.0).astype(float)


def _landuse_group(landuse: pd.Series) -> pd.Series:
    raw = landuse.fillna("").astype(str).str.split(";").str[0].str.lower().str.strip()
    out = pd.Series("other", index=raw.index, dtype="object")
    for group, values in LANDUSE_GROUPS.items():
        out.loc[raw.isin(values)] = group
    return out


def load_landuse_building_polygons_in_bbox(
    osm_pbf_path: str | Path,
    bbox_wgs84: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    """Load OSM landuse and building polygons within a city bbox."""
    gdf = gpd.read_file(
        osm_pbf_path,
        layer="multipolygons",
        bbox=bbox_wgs84,
        columns=["landuse", "building", "other_tags"],
    )
    if gdf.empty:
        return gpd.GeoDataFrame(
            columns=["landuse", "building", "landuse_group", "is_building", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    if "other_tags" not in gdf.columns:
        gdf["other_tags"] = pd.Series(dtype="object", index=gdf.index)
    if "landuse" not in gdf.columns:
        gdf["landuse"] = extract_osm_tag(gdf["other_tags"], "landuse")
    if "building" not in gdf.columns:
        gdf["building"] = extract_osm_tag(gdf["other_tags"], "building")

    out = gdf[["landuse", "building", "geometry"]].copy()
    out["landuse_group"] = _landuse_group(out["landuse"])
    building = out["building"].fillna("").astype(str).str.lower().str.strip()
    out["is_building"] = (building != "") & (~building.isin({"no", "none", "false"}))
    out = out[(out["landuse_group"] != "other") | out["is_building"]].copy()
    return out


def _length_by_block(
    blocks: gpd.GeoDataFrame,
    lines: gpd.GeoDataFrame,
    length_col: str,
    highway_filter: set[str] | None = None,
) -> pd.Series:
    if lines.empty:
        return pd.Series(dtype=float)

    work = lines
    if highway_filter is not None:
        work = lines[lines["highway_type"].isin(highway_filter)].copy()
    if work.empty:
        return pd.Series(dtype=float)

    clipped = gpd.overlay(
        work[["highway_type", "geometry"]],
        blocks[["block_id", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    if clipped.empty:
        return pd.Series(dtype=float)
    clipped[length_col] = clipped.geometry.length
    return clipped.groupby("block_id")[length_col].sum()


def _length_weighted_value_by_block(
    blocks: gpd.GeoDataFrame,
    lines: gpd.GeoDataFrame,
    value_col: str,
) -> pd.Series:
    if lines.empty:
        return pd.Series(dtype=float)
    clipped = gpd.overlay(
        lines[[value_col, "geometry"]],
        blocks[["block_id", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    if clipped.empty:
        return pd.Series(dtype=float)
    clipped["weighted_length"] = clipped.geometry.length * clipped[value_col].astype(float)
    return clipped.groupby("block_id")["weighted_length"].sum()


def _road_node_features(blocks: gpd.GeoDataFrame, lines: gpd.GeoDataFrame) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame(
            columns=[
                "block_id",
                "be_road_node_count",
                "be_intersection_count",
                "be_avg_node_degree",
            ]
        )

    graph, node_coords = build_road_graph(lines)
    degrees = np.diff(graph.indptr)
    nodes = gpd.GeoDataFrame(
        {
            "degree": degrees.astype(float),
            "is_intersection": (degrees >= 3).astype(float),
        },
        geometry=gpd.points_from_xy(node_coords[:, 0], node_coords[:, 1]),
        crs=blocks.crs,
    )

    joined = gpd.sjoin(
        nodes[["degree", "is_intersection", "geometry"]],
        blocks[["block_id", "geometry"]],
        how="inner",
        predicate="within",
    )
    if joined.empty:
        return pd.DataFrame(
            columns=[
                "block_id",
                "be_road_node_count",
                "be_intersection_count",
                "be_avg_node_degree",
            ]
        )

    return (
        joined.groupby("block_id")
        .agg(
            be_road_node_count=("degree", "size"),
            be_intersection_count=("is_intersection", "sum"),
            be_avg_node_degree=("degree", "mean"),
        )
        .reset_index()
    )


def _landuse_building_features(blocks: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame) -> pd.DataFrame:
    """Build exact block-level OSM land-use and building polygon features."""
    index = pd.Index(blocks["block_id"].astype(str), name="block_id")
    out = pd.DataFrame(index=index)
    block_area = blocks.set_index("block_id").geometry.area.astype(float).reindex(index)
    area_km2 = (block_area / 1_000_000).replace(0, np.nan)

    for group in LANDUSE_GROUPS:
        out[f"be_landuse_{group}_share"] = 0.0
    out["be_landuse_entropy"] = 0.0
    out["be_building_footprint_share"] = 0.0
    out["be_building_count_density_per_km2"] = 0.0

    if polygons.empty:
        return out.reset_index()

    work = polygons
    if polygons.crs is not None and polygons.crs != blocks.crs:
        work = polygons.to_crs(blocks.crs)

    work = work[["landuse_group", "is_building", "geometry"]].copy()
    work = work[work.geometry.notna() & ~work.geometry.is_empty].copy()
    if work.empty:
        return out.reset_index()

    clipped = gpd.overlay(
        work,
        blocks[["block_id", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    if clipped.empty:
        return out.reset_index()

    clipped["area_m2"] = clipped.geometry.area.astype(float)
    landuse = clipped[clipped["landuse_group"] != "other"].copy()
    if not landuse.empty:
        landuse_area = (
            landuse.groupby(["block_id", "landuse_group"])["area_m2"]
            .sum()
            .unstack(fill_value=0.0)
        )
        shares = landuse_area.div(block_area.reindex(landuse_area.index), axis=0).fillna(0.0)
        for group in LANDUSE_GROUPS:
            if group in shares.columns:
                out.loc[shares.index, f"be_landuse_{group}_share"] = shares[group].clip(0.0, 1.0)
        share_values = shares[[col for col in shares.columns if col in LANDUSE_GROUPS]].to_numpy()
        share_values = np.clip(share_values, 0.0, 1.0)
        positive_shares = np.where(share_values > 0, share_values, 1.0)
        entropy = -np.sum(
            np.where(share_values > 0, share_values * np.log(positive_shares), 0.0),
            axis=1,
        )
        max_entropy = np.log(max(1, len(LANDUSE_GROUPS)))
        if max_entropy > 0:
            entropy = entropy / max_entropy
        out.loc[shares.index, "be_landuse_entropy"] = entropy

    buildings = clipped[clipped["is_building"]].copy()
    if not buildings.empty:
        building_area = buildings.groupby("block_id")["area_m2"].sum()
        building_count = buildings.groupby("block_id").size()
        out["be_building_footprint_share"] = (
            building_area.reindex(index).fillna(0.0) / block_area.replace(0, np.nan)
        ).fillna(0.0).clip(0.0, 1.0)
        out["be_building_count_density_per_km2"] = (
            building_count.reindex(index).fillna(0.0) / area_km2
        ).fillna(0.0)

    return out.reset_index()


def build_network_features(
    blocks: gpd.GeoDataFrame,
    adjacency_edges: pd.DataFrame,
    osm_pbf_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Build block-level BE predictors from street geometry and block shape."""
    required = {"block_id", "geometry"}
    missing = required - set(blocks.columns)
    if missing:
        raise KeyError(f"Missing required block columns: {sorted(missing)}")

    if blocks.crs is None:
        raise ValueError("Blocks GeoDataFrame must have a CRS.")

    blocks = blocks[["block_id", "geometry"]].copy()
    blocks["block_id"] = blocks["block_id"].astype(str)
    blocks_wgs84 = blocks.to_crs(4326)
    minx, miny, maxx, maxy = map(float, blocks_wgs84.total_bounds.tolist())
    bbox_wgs84 = (minx, miny, maxx, maxy)

    lines = load_street_lines_in_bbox(osm_pbf_path, bbox_wgs84)
    if lines.crs is not None and lines.crs != blocks.crs:
        lines = lines.to_crs(blocks.crs)
    polygons = load_landuse_building_polygons_in_bbox(osm_pbf_path, bbox_wgs84)

    out = pd.DataFrame({"block_id": blocks["block_id"]})
    out["be_block_area_m2"] = blocks.geometry.area.astype(float).to_numpy()
    perimeter = pd.Series(blocks.geometry.length.astype(float).to_numpy()).replace(0, np.nan)
    out["be_compactness"] = (4 * np.pi * out["be_block_area_m2"] / (perimeter**2)).fillna(0.0)

    out = out.set_index("block_id")
    area_km2 = (out["be_block_area_m2"] / 1_000_000).replace(0, np.nan)

    total_len = _length_by_block(blocks, lines, "length_m")
    bike_len = _length_by_block(blocks, lines, "bike_length_m", BIKEABLE_HIGHWAY_TYPES)
    major_len = _length_by_block(blocks, lines, "major_length_m", MAJOR_HIGHWAY_TYPES)
    local_len = _length_by_block(blocks, lines, "local_length_m", LOCAL_HIGHWAY_TYPES)

    out["be_street_length_m"] = total_len.reindex(out.index).fillna(0.0)
    out["be_bikeable_street_length_m"] = bike_len.reindex(out.index).fillna(0.0)
    out["be_street_length_density_m_per_km2"] = (out["be_street_length_m"] / area_km2).fillna(0.0)
    out["be_bikeable_street_length_density_m_per_km2"] = (
        out["be_bikeable_street_length_m"] / area_km2
    ).fillna(0.0)
    out["be_bikeable_street_share"] = (
        out["be_bikeable_street_length_m"] / out["be_street_length_m"].replace(0, np.nan)
    ).fillna(0.0)
    out["be_major_road_share"] = (
        major_len.reindex(out.index).fillna(0.0) / out["be_street_length_m"].replace(0, np.nan)
    ).fillna(0.0)
    out["be_major_road_density_m_per_km2"] = (
        major_len.reindex(out.index).fillna(0.0) / area_km2
    ).fillna(0.0)
    out["be_local_road_share"] = (
        local_len.reindex(out.index).fillna(0.0) / out["be_street_length_m"].replace(0, np.nan)
    ).fillna(0.0)
    if not lines.empty:
        lines = lines.copy()
        lines["speed_kmh"] = _parse_speed_kmh(lines["maxspeed"], lines["highway_type"])
        speed_length = _length_weighted_value_by_block(blocks, lines, "speed_kmh")
        low_speed_len = _length_by_block(
            blocks,
            lines[lines["speed_kmh"] <= 35.0].copy(),
            "low_speed_length_m",
        )
        out["be_estimated_avg_road_speed_kmh"] = (
            speed_length.reindex(out.index).fillna(0.0)
            / out["be_street_length_m"].replace(0, np.nan)
        ).fillna(0.0)
        out["be_low_speed_street_share"] = (
            low_speed_len.reindex(out.index).fillna(0.0)
            / out["be_street_length_m"].replace(0, np.nan)
        ).fillna(0.0)
    else:
        out["be_estimated_avg_road_speed_kmh"] = 0.0
        out["be_low_speed_street_share"] = 0.0

    node_features = _road_node_features(blocks, lines)
    if not node_features.empty:
        node_features = node_features.set_index("block_id")
        for col in ["be_road_node_count", "be_intersection_count", "be_avg_node_degree"]:
            out[col] = node_features[col].reindex(out.index).fillna(0.0)
    else:
        out["be_road_node_count"] = 0.0
        out["be_intersection_count"] = 0.0
        out["be_avg_node_degree"] = 0.0

    out["be_intersection_density_per_km2"] = (out["be_intersection_count"] / area_km2).fillna(0.0)

    degree_counts = pd.concat(
        [
            adjacency_edges["src_block_id"].astype(str),
            adjacency_edges["dst_block_id"].astype(str),
        ]
    ).value_counts()
    out["be_adjacency_degree"] = degree_counts.reindex(out.index).fillna(0.0)

    landuse_features = _landuse_building_features(blocks, polygons)
    if not landuse_features.empty:
        landuse_features = landuse_features.set_index("block_id")
        for col in landuse_features.columns:
            out[col] = landuse_features[col].reindex(out.index).fillna(0.0)

    metadata = {
        "n_street_lines": int(len(lines)),
        "n_landuse_building_polygons": int(len(polygons)),
        "landuse_building_assignment": "exact_polygon_overlay",
        "n_blocks_with_streets": int((out["be_street_length_m"] > 0).sum()),
        "n_blocks_with_intersections": int((out["be_intersection_count"] > 0).sum()),
        "n_blocks_with_building_footprints": int((out["be_building_footprint_share"] > 0).sum()),
    }
    return out.reset_index(), metadata

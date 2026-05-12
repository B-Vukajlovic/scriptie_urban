"""Load OSM amenity features and extract category-related tags from local PBF files."""

from __future__ import annotations

from pathlib import Path
import re

import geopandas as gpd
import pandas as pd


_TAG_PATTERN_TEMPLATE = r'"{tag}"=>"([^\"]+)"'


def find_state_osm_pbf(state_code: str, osm_root: str | Path) -> Path:
	"""Return the first OSM PBF file for a state folder."""
	state_dir = Path(osm_root) / state_code.lower()
	if not state_dir.exists():
		raise FileNotFoundError(f"OSM state folder not found: {state_dir}")

	matches = sorted(state_dir.glob("*.osm.pbf"))
	if not matches:
		raise FileNotFoundError(f"No .osm.pbf files found in: {state_dir}")
	return matches[0]


def _ensure_tag_column(gdf: gpd.GeoDataFrame, tag_name: str) -> pd.Series:
	"""Return a tag column, preferring the explicit field and falling back to other_tags."""
	if tag_name in gdf.columns:
		return gdf[tag_name].astype("object")

	other_tags = gdf.get("other_tags")
	if other_tags is None:
		other_tags = pd.Series(dtype="object", index=gdf.index)
	return extract_osm_tag(other_tags.astype("object"), tag_name)


def load_osm_amenities_in_bbox(
    osm_pbf_path: str | Path,
    bbox_wgs84: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    """Load amenity-like OSM features from points and multipolygons.

    Polygon features are converted to representative points so accessibility
    counts treat all amenities as point destinations.
    """
    pbf = Path(osm_pbf_path)
    if not pbf.exists():
        raise FileNotFoundError(f"OSM PBF not found: {pbf}")

    cols = [
        "osm_id",
        "name",
        "amenity",
        "shop",
        "leisure",
        "other_tags",
        "source_layer",
        "geometry",
    ]
    frames: list[gpd.GeoDataFrame] = []
    for layer_name in ["points", "multipolygons"]:
        gdf = gpd.read_file(pbf, layer=layer_name, bbox=bbox_wgs84)
        if gdf.empty:
            continue

        if "other_tags" not in gdf.columns:
            gdf["other_tags"] = pd.Series(dtype="object", index=gdf.index)
        gdf["amenity"] = _ensure_tag_column(gdf, "amenity")
        gdf["shop"] = _ensure_tag_column(gdf, "shop")
        gdf["leisure"] = _ensure_tag_column(gdf, "leisure")
        gdf["source_layer"] = layer_name

        out = gdf[cols].copy()
        if layer_name == "multipolygons":
            out["geometry"] = out.geometry.representative_point()
        frames.append(out)

    if not frames:
        return gpd.GeoDataFrame(
            columns=cols,
            geometry="geometry",
            crs="EPSG:4326",
        )

    combined = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=frames[0].crs)


def extract_osm_tag(other_tags: pd.Series, tag_name: str) -> pd.Series:
	"""Extract one OSM tag value from the OGR other_tags field."""
	pattern = _TAG_PATTERN_TEMPLATE.format(tag=re.escape(tag_name))
	return other_tags.fillna("").str.extract(pattern, expand=False)

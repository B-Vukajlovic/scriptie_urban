"""Load and prepare census blocks from TIGER raw inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd  # type: ignore[import-not-found]


STATE_ABBREV_TO_FIPS = {
	"al": "01",
	"ak": "02",
	"az": "04",
	"ar": "05",
	"ca": "06",
	"co": "08",
	"ct": "09",
	"de": "10",
	"dc": "11",
	"fl": "12",
	"ga": "13",
	"hi": "15",
	"id": "16",
	"il": "17",
	"in": "18",
	"ia": "19",
	"ks": "20",
	"ky": "21",
	"la": "22",
	"me": "23",
	"md": "24",
	"ma": "25",
	"mi": "26",
	"mn": "27",
	"ms": "28",
	"mo": "29",
	"mt": "30",
	"ne": "31",
	"nv": "32",
	"nh": "33",
	"nj": "34",
	"nm": "35",
	"ny": "36",
	"nc": "37",
	"nd": "38",
	"oh": "39",
	"ok": "40",
	"or": "41",
	"pa": "42",
	"ri": "44",
	"sc": "45",
	"sd": "46",
	"tn": "47",
	"tx": "48",
	"ut": "49",
	"vt": "50",
	"va": "51",
	"wa": "53",
	"wv": "54",
	"wi": "55",
	"wy": "56",
	"pr": "72",
}


def _zip_uri(path: Path) -> str:
	return f"zip://{path.resolve()}"


def _find_column(columns: Iterable[str], candidates: list[str]) -> str:
	column_set = {c.upper(): c for c in columns}
	for candidate in candidates:
		resolved = column_set.get(candidate.upper())
		if resolved:
			return resolved
	raise KeyError(f"None of the expected columns were found: {candidates}")


def _state_fips(state_abbrev: str) -> str:
	fips = STATE_ABBREV_TO_FIPS.get(state_abbrev.lower())
	if not fips:
		raise ValueError(f"Unsupported state abbreviation: {state_abbrev}")
	return fips


def load_place_boundary(tiger_root: Path, state_abbrev: str, place_geoid: str) -> gpd.GeoDataFrame:
	"""Load one place polygon from TIGER place shapefiles."""
	state_fips = _state_fips(state_abbrev)
	path = tiger_root / "place" / state_abbrev.lower() / f"tl_2023_{state_fips}_place.zip"
	if not path.exists():
		raise FileNotFoundError(f"Place file not found: {path}")

	gdf = gpd.read_file(_zip_uri(path))
	geoid_col = _find_column(gdf.columns, ["GEOID", "GEOID20", "GEOID10"])
	out = gdf[gdf[geoid_col] == place_geoid].copy()
	if out.empty:
		raise ValueError(f"Place GEOID {place_geoid} not found in {path.name}.")
	return out[[geoid_col, "geometry"]].rename(columns={geoid_col: "place_geoid"})


def load_blocks_for_state(
	tiger_root: Path,
	state_abbrev: str,
	bbox: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
	"""Load TIGER blocks for one state, pre-filtered by bounding box."""
	state_fips = _state_fips(state_abbrev)
	path = tiger_root / "blocks" / state_abbrev.lower() / f"tl_2023_{state_fips}_tabblock20.zip"
	if not path.exists():
		raise FileNotFoundError(f"Block file not found: {path}")
	return gpd.read_file(_zip_uri(path), bbox=bbox)


def clip_blocks_to_place(blocks: gpd.GeoDataFrame, place: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
	"""Clip blocks to a place polygon using intersection."""
	if blocks.crs is None:
		raise ValueError("Blocks GeoDataFrame has no CRS.")
	if place.crs is None:
		raise ValueError("Place GeoDataFrame has no CRS.")
	if blocks.crs != place.crs:
		place = place.to_crs(blocks.crs)

	place_geom = gpd.GeoDataFrame(place[["geometry"]].copy(), geometry="geometry", crs=place.crs)
	clipped = gpd.overlay(blocks, place_geom, how="intersection", keep_geom_type=True)
	if clipped.empty:
		raise ValueError("No blocks intersected the selected place boundary.")
	return clipped


def select_blocks_for_place(
	blocks: gpd.GeoDataFrame,
	place: gpd.GeoDataFrame,
	mode: str = "representative_point",
) -> gpd.GeoDataFrame:
	"""Select city blocks using a configurable place-membership rule.

	``intersection`` clips geometries exactly to the place boundary but can be
	very slow for large cities. ``representative_point`` keeps full census block
	geometries whose representative point lies inside the place; this is much
	faster and aligns with full-block LEHD job counts.
	"""
	if mode == "intersection":
		return clip_blocks_to_place(blocks, place)
	if blocks.crs is None:
		raise ValueError("Blocks GeoDataFrame has no CRS.")
	if place.crs is None:
		raise ValueError("Place GeoDataFrame has no CRS.")
	if blocks.crs != place.crs:
		place = place.to_crs(blocks.crs)

	place_geom = place.geometry.union_all()
	if mode == "representative_point":
		mask = blocks.geometry.representative_point().within(place_geom)
	elif mode == "intersects":
		mask = blocks.geometry.intersects(place_geom)
	else:
		raise ValueError(
			"mode must be one of: intersection, representative_point, intersects"
		)

	out = blocks.loc[mask].copy()
	if out.empty:
		raise ValueError("No blocks matched the selected place boundary.")
	return out


def standardize_block_columns(blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
	"""Rename key TIGER columns to stable names used in the pipeline."""
	geoid_col = _find_column(blocks.columns, ["GEOID20", "GEOID10", "GEOID"])
	state_col = _find_column(blocks.columns, ["STATEFP20", "STATEFP"])
	county_col = _find_column(blocks.columns, ["COUNTYFP20", "COUNTYFP"])
	tract_col = _find_column(blocks.columns, ["TRACTCE20", "TRACTCE"])
	block_col = _find_column(blocks.columns, ["BLOCKCE20", "BLOCKCE"])

	out = blocks.rename(
		columns={
			geoid_col: "block_id",
			state_col: "state_fips",
			county_col: "county_fips",
			tract_col: "tract_code",
			block_col: "block_code",
		}
	).copy()

	out = out[["block_id", "state_fips", "county_fips", "tract_code", "block_code", "geometry"]]
	out = out.loc[~out["block_id"].duplicated()].reset_index(drop=True)
	return out

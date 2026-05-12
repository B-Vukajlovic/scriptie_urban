"""Projection helpers for city-scale geospatial preprocessing."""

from __future__ import annotations

import math

import geopandas as gpd


def estimate_utm_epsg(gdf: gpd.GeoDataFrame) -> int:
	"""Estimate a UTM EPSG code from the dataset centroid.

	The function assumes data is in a geographic CRS and reprojects to EPSG:4326
	if needed before estimating the UTM zone.
	"""
	if gdf.empty:
		raise ValueError("Cannot estimate UTM CRS for an empty GeoDataFrame.")

	if gdf.crs is None:
		raise ValueError("Input GeoDataFrame has no CRS.")

	geo = gdf.to_crs(4326) if gdf.crs.to_epsg() != 4326 else gdf
	centroid = geo.geometry.union_all().centroid
	lon, lat = float(centroid.x), float(centroid.y)

	zone = int(math.floor((lon + 180.0) / 6.0) + 1)
	is_northern = lat >= 0
	return 32600 + zone if is_northern else 32700 + zone

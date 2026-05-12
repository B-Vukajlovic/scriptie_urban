"""Centroid generation utilities."""

from __future__ import annotations

import geopandas as gpd


def build_block_centroids(blocks_projected: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
	"""Create representative point centroids for each block.

	Representative points are guaranteed to lie within polygons, which avoids
	centroid-outside-polygon artifacts for irregular shapes.
	"""
	if "block_id" not in blocks_projected.columns:
		raise KeyError("Expected column 'block_id' in blocks GeoDataFrame.")

	centroids = blocks_projected[["block_id", "geometry"]].copy()
	centroids["geometry"] = centroids.geometry.representative_point()
	centroids["x_m"] = centroids.geometry.x
	centroids["y_m"] = centroids.geometry.y

	centroids_ll = centroids.to_crs(4326)
	centroids["lon"] = centroids_ll.geometry.x
	centroids["lat"] = centroids_ll.geometry.y
	return centroids

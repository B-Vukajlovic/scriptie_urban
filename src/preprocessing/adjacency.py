"""Adjacency construction for block-level spatial graphs."""

from __future__ import annotations

import pandas as pd


def build_adjacency_edges(blocks_gdf, id_col: str = "block_id") -> pd.DataFrame:
	"""Build undirected rook-style adjacency edges for polygons.

	Only polygon pairs with a shared boundary segment are kept. Corner-touching
	polygons are excluded because they do not represent a meaningful shared edge
	for the block graph.
	"""
	if id_col not in blocks_gdf.columns:
		raise KeyError(f"Expected id column '{id_col}' in blocks GeoDataFrame.")

	left = blocks_gdf[[id_col, "geometry"]].copy()
	right = blocks_gdf[[id_col, "geometry"]].copy()

	joined = left.sjoin(right, how="inner", predicate="touches", lsuffix="src", rsuffix="dst")
	src_col = f"{id_col}_src"
	dst_col = f"{id_col}_dst"
	right_index_col = next((col for col in joined.columns if col.startswith("index_")), None)
	if right_index_col is None:
		raise KeyError("Spatial join did not expose the right-side index column.")

	right_geometry = right.geometry.reset_index(drop=True)
	joined["dst_geometry"] = joined[right_index_col].map(right_geometry)
	shared = joined.geometry.intersection(joined["dst_geometry"])
	joined = joined.loc[shared.length > 0].copy()

	edges = joined[[src_col, dst_col]].rename(
		columns={src_col: "src_block_id", dst_col: "dst_block_id"}
	)
	edges = edges[edges["src_block_id"] != edges["dst_block_id"]].copy()

	edges[["a", "b"]] = pd.DataFrame(
		{
			"a": edges[["src_block_id", "dst_block_id"]].min(axis=1),
			"b": edges[["src_block_id", "dst_block_id"]].max(axis=1),
		}
	)
	edges = edges[["a", "b"]].drop_duplicates().rename(
		columns={"a": "src_block_id", "b": "dst_block_id"}
	)
	return edges.reset_index(drop=True)

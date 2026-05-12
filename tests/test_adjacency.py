import geopandas as gpd
from shapely.geometry import Polygon

from src.preprocessing.adjacency import build_adjacency_edges


def test_build_adjacency_edges_excludes_corner_only_neighbors():
	blocks = gpd.GeoDataFrame(
		{
			"block_id": ["a", "b", "c"],
			"geometry": [
				Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
				Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
				Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
			],
		},
		crs="EPSG:3857",
	)

	edges = build_adjacency_edges(blocks)
	edge_pairs = {tuple(row) for row in edges[["src_block_id", "dst_block_id"]].to_numpy()}

	assert ("a", "b") in edge_pairs
	assert ("b", "c") in edge_pairs
	assert ("a", "c") not in edge_pairs

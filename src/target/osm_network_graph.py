"""Utilities for building an OSM road graph, snapping points, and attaching destinations."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import KDTree


DEFAULT_HIGHWAY_TYPES = {
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
}


def load_osm_road_lines_in_bbox(
    osm_pbf_path: str,
    bbox_wgs84: tuple[float, float, float, float],
    allowed_highways: set[str] | None = None,
) -> gpd.GeoDataFrame:
    """Load road-like OSM lines within a bbox from a local PBF file."""
    gdf = gpd.read_file(osm_pbf_path, layer="lines", bbox=bbox_wgs84)
    if gdf.empty:
        raise ValueError("No OSM lines found in bbox.")

    if "highway" not in gdf.columns:
        raise ValueError("OSM lines layer has no 'highway' column.")

    allowed = allowed_highways or DEFAULT_HIGHWAY_TYPES
    highway = gdf["highway"].fillna("").astype(str)
    primary_tag = highway.str.split(";").str[0].str.lower().str.strip()
    out = gdf[primary_tag.isin(allowed)].copy()
    if out.empty:
        raise ValueError("No OSM road lines matched allowed highway types.")
    return out[["highway", "geometry"]]


def _iter_line_coordinates(geometry):
    geom_type = geometry.geom_type
    if geom_type == "LineString":
        yield np.asarray(geometry.coords, dtype=float)
    elif geom_type == "MultiLineString":
        for part in geometry.geoms:
            yield np.asarray(part.coords, dtype=float)


def build_road_graph(
    lines_projected: gpd.GeoDataFrame,
    round_decimals: int = 3,
) -> tuple[csr_matrix, np.ndarray]:
    """Build an undirected weighted sparse graph from projected road lines.

    Returns
    -------
    graph:
        CSR adjacency matrix with edge lengths in meters.
    node_coords:
        Array shape (n_nodes, 2) with node coordinates in projected meters.
    """
    if lines_projected.crs is None:
        raise ValueError("Road lines GeoDataFrame has no CRS.")

    node_index: dict[tuple[float, float], int] = {}
    coords_list: list[tuple[float, float]] = []
    edge_weights: dict[tuple[int, int], float] = {}

    def get_node_id(x: float, y: float) -> int:
        key = (round(float(x), round_decimals), round(float(y), round_decimals))
        idx = node_index.get(key)
        if idx is None:
            idx = len(coords_list)
            node_index[key] = idx
            coords_list.append(key)
        return idx

    for geom in lines_projected.geometry:
        if geom is None or geom.is_empty:
            continue
        for coords in _iter_line_coordinates(geom):
            if len(coords) < 2:
                continue
            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]
                u = get_node_id(x1, y1)
                v = get_node_id(x2, y2)
                if u == v:
                    continue
                w = float(np.hypot(x1 - x2, y1 - y2))

                uv = (u, v)
                vu = (v, u)
                edge_weights[uv] = min(w, edge_weights.get(uv, w))
                edge_weights[vu] = min(w, edge_weights.get(vu, w))

    if not coords_list or not edge_weights:
        raise ValueError("Failed to build a non-empty road graph from OSM lines.")

    rows, cols, data = [], [], []
    for (u, v), w in edge_weights.items():
        rows.append(u)
        cols.append(v)
        data.append(w)

    n_nodes = len(coords_list)
    graph = coo_matrix(
        (np.asarray(data), (np.asarray(rows), np.asarray(cols))),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    graph = csr_matrix(graph)
    node_coords = np.asarray(coords_list, dtype=float)
    return graph, node_coords


def snap_points_to_nodes(
    points_xy: np.ndarray,
    node_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map XY points to nearest graph node and return (node_idx, snap_distance_m)."""
    if len(node_coords) == 0:
        raise ValueError("Cannot snap points: node_coords is empty")
    tree = KDTree(node_coords)
    dists, node_idx = tree.query(points_xy)
    return np.asarray(node_idx, dtype=int), np.asarray(dists, dtype=float)


def attach_destination_nodes(
	graph: csr_matrix,
	destination_node_idx: np.ndarray,
	destination_offsets_m: np.ndarray,
	destination_values: np.ndarray,
) -> tuple[csr_matrix, np.ndarray]:
	"""Attach destination pseudo-nodes to preserve destination snap offsets.

	Each destination becomes a new leaf node connected to its snapped road node
	with an edge weighted by the destination snap distance. Feature values live
	on the pseudo-nodes only, which makes shortest-path reachability account for
	both network distance and destination offset.
	"""
	graph_shape = graph.shape
	if graph_shape is None:
		raise ValueError("graph must have a valid shape")

	destination_node_idx = np.asarray(destination_node_idx, dtype=int)
	destination_offsets_m = np.asarray(destination_offsets_m, dtype=float)
	destination_values = np.asarray(destination_values, dtype=float)

	if destination_values.ndim != 2:
		raise ValueError("destination_values must be a 2D array")
	if len(destination_node_idx) != len(destination_offsets_m):
		raise ValueError("destination_node_idx and destination_offsets_m must have the same length")
	if len(destination_node_idx) != destination_values.shape[0]:
		raise ValueError("destination_values must have one row per destination")

	n_base_nodes = int(graph_shape[0])
	n_destinations = int(len(destination_node_idx))
	if n_destinations == 0:
		zero_features = np.zeros((n_base_nodes, destination_values.shape[1]), dtype=float)
		return csr_matrix(graph), zero_features

	graph_coo = graph.tocoo()
	pseudo_idx = np.arange(n_base_nodes, n_base_nodes + n_destinations, dtype=int)

	rows = np.concatenate([graph_coo.row, destination_node_idx, pseudo_idx])
	cols = np.concatenate([graph_coo.col, pseudo_idx, destination_node_idx])
	data = np.concatenate([graph_coo.data, destination_offsets_m, destination_offsets_m])

	augmented = coo_matrix(
		(data, (rows, cols)),
		shape=(n_base_nodes + n_destinations, n_base_nodes + n_destinations),
	).tocsr()
	augmented = csr_matrix(augmented)

	feature_matrix = np.zeros(
		(n_base_nodes + n_destinations, destination_values.shape[1]),
		dtype=float,
	)
	feature_matrix[n_base_nodes:, :] = destination_values
	return augmented, feature_matrix

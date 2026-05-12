"""General shortest-path reachability utilities for graph-based accessibility."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra


def build_block_graph_csr(
    block_ids: pd.Series,
    x_m: np.ndarray,
    y_m: np.ndarray,
    adjacency_edges: pd.DataFrame,
) -> tuple[csr_matrix, dict[str, int]]:
    """Build an undirected weighted CSR graph from adjacency edges."""
    ids = block_ids.astype(str).to_list()
    id_to_idx = {block_id: i for i, block_id in enumerate(ids)}
    n = len(ids)

    src = adjacency_edges["src_block_id"].astype(str)
    dst = adjacency_edges["dst_block_id"].astype(str)
    keep = src.isin(id_to_idx) & dst.isin(id_to_idx)
    src = src[keep]
    dst = dst[keep]

    src_idx = src.map(id_to_idx).to_numpy(dtype=int)
    dst_idx = dst.map(id_to_idx).to_numpy(dtype=int)

    dx = x_m[src_idx] - x_m[dst_idx]
    dy = y_m[src_idx] - y_m[dst_idx]
    w = np.sqrt(dx * dx + dy * dy)

    rows = np.concatenate([src_idx, dst_idx])
    cols = np.concatenate([dst_idx, src_idx])
    data = np.concatenate([w, w])

    graph = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    graph = csr_matrix(graph)
    return graph, id_to_idx


def batched_reachability_sums(
    graph: csr_matrix,
    destination_matrix: np.ndarray,
    radii_m: list[float],
    batch_size: int = 128,
    source_indices: np.ndarray | None = None,
    source_offsets_m: np.ndarray | None = None,
) -> dict[float, np.ndarray]:
    """Compute sums of destination attributes reachable within each radius."""
    graph_shape = graph.shape
    if graph_shape is None:
        raise ValueError("graph must have a valid shape")

    if graph_shape[0] != destination_matrix.shape[0]:
        raise ValueError("graph and destination_matrix must have the same number of nodes")

    n_nodes = int(graph_shape[0])
    n_features = destination_matrix.shape[1]
    radii = [float(r) for r in radii_m]
    max_radius = max(radii)

    if source_indices is None:
        source_idx: np.ndarray = np.arange(n_nodes, dtype=int)
    else:
        source_idx = np.asarray(source_indices, dtype=int)

    if source_offsets_m is None:
        source_off: np.ndarray = np.zeros(len(source_idx), dtype=float)
    else:
        source_off = np.asarray(source_offsets_m, dtype=float)
        if len(source_off) != len(source_idx):
            raise ValueError("source_offsets_m must have the same length as source_indices")

    n_sources = len(source_idx)
    sums = {r: np.zeros((n_sources, n_features), dtype=float) for r in radii}

    for start in range(0, n_sources, batch_size):
        end = min(start + batch_size, n_sources)
        indices = source_idx[start:end]
        offsets = source_off[start:end]

        dist = dijkstra(graph, directed=False, indices=indices, limit=max_radius)
        if dist.ndim == 1:
            dist = dist[np.newaxis, :]

        for r in radii:
            threshold = np.maximum(float(r) - offsets, 0.0)[:, np.newaxis]
            reachable = dist <= threshold
            sums[r][start:end, :] = reachable.astype(float) @ destination_matrix

    return sums
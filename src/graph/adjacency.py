"""Lightweight graph adjacency utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def build_undirected_adjacency(
    block_ids: pd.Series,
    edges: pd.DataFrame,
    src_col: str = "src_block_id",
    dst_col: str = "dst_block_id",
) -> sparse.csr_matrix:
    """Build a CSR adjacency matrix aligned to ``block_ids``."""
    required = {src_col, dst_col}
    missing = required - set(edges.columns)
    if missing:
        raise KeyError(f"Adjacency edge list is missing columns: {sorted(missing)}")

    index = pd.Index(block_ids.astype(str))
    src = index.get_indexer(edges[src_col].astype(str))
    dst = index.get_indexer(edges[dst_col].astype(str))
    valid = (src >= 0) & (dst >= 0) & (src != dst)
    src = src[valid]
    dst = dst[valid]

    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(len(rows), dtype=float)
    adjacency = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(index), len(index)),
    ).tocsr()
    adjacency.data[:] = 1.0
    adjacency.eliminate_zeros()
    return adjacency

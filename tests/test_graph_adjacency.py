import pandas as pd

from src.graph.adjacency import build_undirected_adjacency


def test_build_undirected_adjacency_ignores_edges_outside_block_index() -> None:
    block_ids = pd.Series(["a", "b"])
    edges = pd.DataFrame(
        {
            "src_block_id": ["a", "a"],
            "dst_block_id": ["b", "missing"],
        }
    )

    adjacency = build_undirected_adjacency(block_ids, edges)

    assert adjacency.shape == (2, 2)
    assert adjacency.nnz == 2

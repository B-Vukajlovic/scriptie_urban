import torch
import warnings

from src.models.gnn import GCNRegressor, GraphSAGERegressor

warnings.filterwarnings(
    "ignore",
    message="Sparse invariant checks are implicitly disabled.*",
    category=UserWarning,
)


def _toy_sparse_adjacency() -> torch.Tensor:
    indices = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    values = torch.ones(4, dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.sparse_coo_tensor(
            indices,
            values,
            (3, 3),
            check_invariants=False,
        ).coalesce()


def test_gnn_regressors_return_one_prediction_per_node() -> None:
    x = torch.randn(3, 4)
    adjacency = _toy_sparse_adjacency()

    for model in [
        GCNRegressor(in_dim=4, hidden_dim=8, dropout=0.0),
        GraphSAGERegressor(in_dim=4, hidden_dim=8, dropout=0.0),
    ]:
        out = model(x, adjacency)
        assert out.shape == (3,)
        assert torch.isfinite(out).all()

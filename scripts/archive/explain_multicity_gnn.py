"""GNNExplainer-style local explanations for custom GCN/GraphSAGE regressors.

This project uses lightweight PyTorch GNNs rather than PyTorch Geometric models,
so this script implements the core GNNExplainer idea directly:

- train the selected GNN on the configured graph;
- sample held-out spatial-test nodes;
- build a k-hop ego graph around each explained node;
- optimize differentiable feature and edge masks that preserve the original
  prediction for that node;
- aggregate local masks into report-ready model-level feature importance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_multicity_gnn import (  # noqa: E402
    GraphInputs,
    ModelName,
    _row_normalize,
    _symmetric_normalize_with_self_loops,
    _torch_sparse,
    apply_gnn_feature_config,
    build_single_city_splits,
    load_multicity_graph,
    make_model,
    standardize_train_only,
)
from src.models.metrics import regression_metrics  # noqa: E402
from src.utils.cities import DEFAULT_CITIES, validate_cities  # noqa: E402


@dataclass(frozen=True)
class TrainedGNN:
    model_name: ModelName
    model: nn.Module
    x: torch.Tensor
    y_scaled: torch.Tensor
    y_true: np.ndarray
    y_pred: np.ndarray
    y_mean: float
    y_std: float
    labels: pd.Series
    adjacency_tensor: torch.Tensor
    best_epoch: int
    best_val_loss: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain reduced GCN/GraphSAGE predictions with learned feature/edge masks."
    )
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    parser.add_argument("--interim-root", default="data/interim")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--target-column", default="Y_global_log_minmax")
    parser.add_argument("--feature-set", choices=["full", "reduced"], default="reduced")
    parser.add_argument("--feature-view", choices=["raw", "log1p"], default="log1p")
    parser.add_argument(
        "--reduced-feature-set",
        default="data/interim/modeling/reduced_feature_set.json",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["gcn", "graphsage"],
        default=["gcn", "graphsage"],
    )
    parser.add_argument("--split-seed", type=int, default=1000)
    parser.add_argument("--split-val-frac", type=float, default=0.15)
    parser.add_argument("--split-test-frac", type=float, default=0.15)
    parser.add_argument("--split-grid-bins-x", type=int, default=8)
    parser.add_argument("--split-grid-bins-y", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-test-nodes", type=int, default=200)
    parser.add_argument(
        "--nodes-per-city",
        type=int,
        default=25,
        help="Maximum explained test nodes sampled per city before applying --max-test-nodes.",
    )
    parser.add_argument("--khop", type=int, default=2)
    parser.add_argument("--explain-steps", type=int, default=80)
    parser.add_argument("--explain-learning-rate", type=float, default=0.08)
    parser.add_argument("--feature-size-penalty", type=float, default=0.01)
    parser.add_argument("--edge-size-penalty", type=float, default=0.003)
    parser.add_argument("--mask-entropy-penalty", type=float, default=0.001)
    parser.add_argument("--top-n-features", type=int, default=20)
    parser.add_argument("--top-n-edges", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output-prefix", default="multicity_gnn_reduced_explainer")
    return parser.parse_args()


def single_spatial_labels(
    graph_inputs: GraphInputs,
    cities: list[str],
    args: argparse.Namespace,
) -> tuple[pd.Series, pd.DataFrame]:
    splits = build_single_city_splits(cities, args.interim_root, args)
    nodes = graph_inputs.frame[["node_id"]].copy()
    merged = nodes.merge(
        splits[["node_id", "split"]],
        on="node_id",
        how="left",
        validate="one_to_one",
    )
    if merged["split"].isna().any():
        raise ValueError("Spatial split does not cover every graph node.")
    return merged["split"], splits


def train_model(
    model_name: ModelName,
    graph_inputs: GraphInputs,
    labels: pd.Series,
    args: argparse.Namespace,
) -> TrainedGNN:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    frame = graph_inputs.frame
    train_mask = labels.eq("train").to_numpy()
    val_mask = labels.eq("val").to_numpy()
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Train and validation masks must be non-empty.")

    data = standardize_train_only(
        frame,
        graph_inputs.feature_columns,
        args.target_column,
        train_mask,
    )
    x = torch.tensor(data.x, dtype=torch.float32)
    y = torch.tensor(data.y_scaled, dtype=torch.float32)
    train_idx = torch.tensor(np.flatnonzero(train_mask), dtype=torch.long)
    val_idx = torch.tensor(np.flatnonzero(val_mask), dtype=torch.long)

    if model_name == "gcn":
        adjacency_tensor = _torch_sparse(_symmetric_normalize_with_self_loops(graph_inputs.adjacency))
    else:
        adjacency_tensor = _torch_sparse(_row_normalize(graph_inputs.adjacency, self_loops=False))

    model = make_model(
        model_name=model_name,
        in_dim=len(graph_inputs.feature_columns),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    loss_fn = nn.MSELoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = int(args.patience)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, adjacency_tensor)
        loss = loss_fn(pred[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_eval = model(x, adjacency_tensor)
            val_loss = float(loss_fn(pred_eval[val_idx], y[val_idx]).item())
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_left = int(args.patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x, adjacency_tensor).detach().cpu().numpy()
    y_pred = pred_scaled * data.y_std + data.y_mean
    y_true = frame[args.target_column].astype(float).to_numpy()
    return TrainedGNN(
        model_name=model_name,
        model=model,
        x=x,
        y_scaled=y,
        y_true=y_true,
        y_pred=y_pred,
        y_mean=data.y_mean,
        y_std=data.y_std,
        labels=labels.reset_index(drop=True),
        adjacency_tensor=adjacency_tensor,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
    )


def sample_test_nodes(
    frame: pd.DataFrame,
    labels: pd.Series,
    max_nodes: int,
    nodes_per_city: int,
    seed: int,
) -> pd.DataFrame:
    candidates = frame.loc[labels.eq("test").to_numpy(), ["node_id", "city", "block_id"]].copy()
    candidates["node_index"] = candidates.index.astype(int)
    sampled_parts: list[pd.DataFrame] = []
    for _city, group in candidates.groupby("city", sort=False):
        n = min(len(group), nodes_per_city)
        sampled_parts.append(group.sample(n=n, random_state=seed) if len(group) > n else group)
    sampled = pd.concat(sampled_parts, ignore_index=True)
    if max_nodes > 0 and len(sampled) > max_nodes:
        sampled = sampled.sample(n=max_nodes, random_state=seed)
    return sampled.sort_values(["city", "node_index"]).reset_index(drop=True)


def khop_node_indices(adjacency: sparse.csr_matrix, node_index: int, khop: int) -> np.ndarray:
    visited = {int(node_index)}
    frontier = {int(node_index)}
    for _ in range(khop):
        if not frontier:
            break
        rows = np.array(sorted(frontier), dtype=int)
        neighbors = set(adjacency[rows].nonzero()[1].astype(int).tolist())
        frontier = neighbors - visited
        visited.update(neighbors)
    return np.array(sorted(visited), dtype=int)


def undirected_edges_from_dense(adjacency_dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.triu(adjacency_dense > 0, k=1).nonzero()
    return rows.astype(int), cols.astype(int)


def dense_weighted_adjacency(
    n_nodes: int,
    edge_rows: torch.Tensor,
    edge_cols: torch.Tensor,
    edge_mask: torch.Tensor,
    model_name: ModelName,
) -> torch.Tensor:
    adjacency = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    if len(edge_rows) > 0:
        adjacency[edge_rows, edge_cols] = edge_mask
        adjacency[edge_cols, edge_rows] = edge_mask
    if model_name == "gcn":
        graph = adjacency + torch.eye(n_nodes, dtype=torch.float32)
        degree = graph.sum(dim=1).clamp(min=1e-8)
        inv_sqrt = degree.pow(-0.5)
        return inv_sqrt[:, None] * graph * inv_sqrt[None, :]
    degree = adjacency.sum(dim=1).clamp(min=1e-8)
    return adjacency / degree[:, None]


def masked_forward(
    model_name: ModelName,
    model: nn.Module,
    x_sub: torch.Tensor,
    normalized_adjacency: torch.Tensor,
) -> torch.Tensor:
    if model_name == "gcn":
        # Same computation as GCNRegressor, but using dense differentiable adjacency.
        h = model.activation(model.conv1.linear(normalized_adjacency @ x_sub))
        h = model.activation(model.conv2.linear(normalized_adjacency @ h))
        return model.output(h).squeeze(-1)

    h1_neighbor = normalized_adjacency @ x_sub
    h = model.activation(model.sage1.linear(torch.cat([x_sub, h1_neighbor], dim=1)))
    h2_neighbor = normalized_adjacency @ h
    h = model.activation(model.sage2.linear(torch.cat([h, h2_neighbor], dim=1)))
    return model.output(h).squeeze(-1)


def mask_entropy(mask: torch.Tensor) -> torch.Tensor:
    clipped = mask.clamp(min=1e-8, max=1 - 1e-8)
    return (-(clipped * torch.log(clipped) + (1 - clipped) * torch.log(1 - clipped))).mean()


def explain_one_node(
    trained: TrainedGNN,
    graph_inputs: GraphInputs,
    node_index: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    sub_nodes = khop_node_indices(graph_inputs.adjacency, node_index, int(args.khop))
    local_target_idx = int(np.flatnonzero(sub_nodes == node_index)[0])
    sub_adj = graph_inputs.adjacency[sub_nodes][:, sub_nodes].toarray().astype(float)
    edge_rows_np, edge_cols_np = undirected_edges_from_dense(sub_adj)
    edge_rows = torch.tensor(edge_rows_np, dtype=torch.long)
    edge_cols = torch.tensor(edge_cols_np, dtype=torch.long)
    x_sub_base = trained.x[sub_nodes].detach()
    target_prediction = torch.tensor(
        float((trained.y_pred[node_index] - trained.y_mean) / trained.y_std),
        dtype=torch.float32,
    )

    feature_logits = nn.Parameter(torch.full((x_sub_base.shape[1],), 2.0))
    edge_logits = nn.Parameter(torch.full((len(edge_rows_np),), 2.0))
    optimizer = torch.optim.Adam(
        [feature_logits, edge_logits],
        lr=float(args.explain_learning_rate),
    )

    last_loss = 0.0
    for _step in range(int(args.explain_steps)):
        optimizer.zero_grad(set_to_none=True)
        feature_mask = torch.sigmoid(feature_logits)
        edge_mask = torch.sigmoid(edge_logits)
        normalized = dense_weighted_adjacency(
            n_nodes=len(sub_nodes),
            edge_rows=edge_rows,
            edge_cols=edge_cols,
            edge_mask=edge_mask,
            model_name=trained.model_name,
        )
        pred = masked_forward(
            model_name=trained.model_name,
            model=trained.model,
            x_sub=x_sub_base * feature_mask,
            normalized_adjacency=normalized,
        )[local_target_idx]
        fidelity_loss = (pred - target_prediction).pow(2)
        loss = (
            fidelity_loss
            + float(args.feature_size_penalty) * feature_mask.mean()
            + float(args.edge_size_penalty) * edge_mask.mean()
            + float(args.mask_entropy_penalty) * (mask_entropy(feature_mask) + mask_entropy(edge_mask))
        )
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())

    with torch.no_grad():
        feature_mask = torch.sigmoid(feature_logits).detach().cpu().numpy()
        edge_mask = torch.sigmoid(edge_logits).detach().cpu().numpy()
        normalized = dense_weighted_adjacency(
            n_nodes=len(sub_nodes),
            edge_rows=edge_rows,
            edge_cols=edge_cols,
            edge_mask=torch.tensor(edge_mask, dtype=torch.float32),
            model_name=trained.model_name,
        )
        masked_pred_scaled = float(
            masked_forward(
                model_name=trained.model_name,
                model=trained.model,
                x_sub=x_sub_base * torch.tensor(feature_mask, dtype=torch.float32),
                normalized_adjacency=normalized,
            )[local_target_idx].detach().cpu().item()
        )
    masked_pred = masked_pred_scaled * trained.y_std + trained.y_mean
    original_pred = float(trained.y_pred[node_index])

    node = graph_inputs.frame.iloc[node_index]
    node_row = {
        "model": trained.model_name,
        "node_index": int(node_index),
        "node_id": str(node["node_id"]),
        "city": str(node["city"]),
        "block_id": str(node["block_id"]),
        "y_true": float(trained.y_true[node_index]),
        "y_pred": original_pred,
        "masked_y_pred": float(masked_pred),
        "fidelity_abs_error": float(abs(masked_pred - original_pred)),
        "subgraph_nodes": int(len(sub_nodes)),
        "subgraph_edges": int(len(edge_rows_np)),
        "explainer_loss": last_loss,
    }
    feature_rows = pd.DataFrame(
        {
            "model": trained.model_name,
            "node_id": str(node["node_id"]),
            "city": str(node["city"]),
            "block_id": str(node["block_id"]),
            "feature": graph_inputs.feature_columns,
            "importance": feature_mask.astype(float),
        }
    )
    feature_rows["rank_within_node"] = (
        feature_rows["importance"].rank(method="first", ascending=False).astype(int)
    )
    edge_table = pd.DataFrame(
        {
            "model": trained.model_name,
            "node_id": str(node["node_id"]),
            "city": str(node["city"]),
            "block_id": str(node["block_id"]),
            "src_node_id": graph_inputs.frame.iloc[sub_nodes[edge_rows_np]]["node_id"].to_numpy()
            if len(edge_rows_np)
            else [],
            "dst_node_id": graph_inputs.frame.iloc[sub_nodes[edge_cols_np]]["node_id"].to_numpy()
            if len(edge_cols_np)
            else [],
            "edge_importance": edge_mask.astype(float),
        }
    )
    if not edge_table.empty:
        edge_table = edge_table.sort_values("edge_importance", ascending=False).head(
            int(args.top_n_edges)
        )
    return node_row, feature_rows, edge_table


def plot_top_features(global_importance: pd.DataFrame, output_path: Path, top_n: int) -> None:
    models = global_importance["model"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(10, max(4.5, 0.36 * top_n * len(models))),
        squeeze=False,
    )
    for ax, model_name in zip(axes.ravel(), models):
        top = (
            global_importance[global_importance["model"] == model_name]
            .sort_values("mean_importance", ascending=False)
            .head(top_n)
            .iloc[::-1]
        )
        labels = (
            top["feature"]
            .str.removeprefix("pt_")
            .str.removeprefix("be_")
            .str.removeprefix("acs_")
            .str.replace("_", " ")
        )
        ax.barh(labels, top["mean_importance"], color="#2a6f8f")
        ax.set_title(f"{model_name.upper()} GNNExplainer-style feature masks")
        ax.set_xlabel("Mean learned feature-mask importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_city_heatmap(city_importance: pd.DataFrame, output_path: Path, top_n: int) -> None:
    top_features = (
        city_importance.groupby("feature")["mean_importance"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )
    for model_name, group in city_importance.groupby("model", sort=False):
        heatmap = (
            group[group["feature"].isin(top_features)]
            .pivot(index="feature", columns="city", values="mean_importance")
            .reindex(index=top_features)
            .fillna(0.0)
        )
        fig, ax = plt.subplots(figsize=(12, max(5, 0.4 * len(top_features))))
        image = ax.imshow(heatmap.to_numpy(dtype=float), cmap="YlGnBu")
        ax.set_title(f"{model_name.upper()} GNN explanation importance by city")
        ax.set_xticks(np.arange(len(heatmap.columns)))
        ax.set_xticklabels(heatmap.columns.tolist(), rotation=45, ha="right")
        ax.set_yticks(np.arange(len(heatmap.index)))
        ax.set_yticklabels(
            [
                feature.removeprefix("pt_").removeprefix("be_").removeprefix("acs_").replace("_", " ")
                for feature in heatmap.index
            ]
        )
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("Mean feature-mask importance")
        fig.tight_layout()
        fig.savefig(output_path.with_name(f"{output_path.stem}_{model_name}{output_path.suffix}"), dpi=180)
        plt.close(fig)


def write_report(
    report_path: Path,
    global_importance: pd.DataFrame,
    node_summary: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    lines = [
        "# GNNExplainer-Style GNN Explanations",
        "",
        "This report summarizes learned local feature masks for held-out spatial-test nodes.",
        "The implementation follows the GNNExplainer principle of optimizing feature and edge masks to preserve each node prediction on its k-hop ego graph.",
        "",
        "## Run Metadata",
        "",
        f"- Target: {metadata['target_column']}",
        f"- Feature set/view: {metadata['feature_set']} / {metadata['feature_view']}",
        f"- Explained nodes: {metadata['n_explained_nodes']}",
        f"- k-hop ego graph: {metadata['khop']}",
        f"- Explanation steps: {metadata['explain_steps']}",
        "",
        "## Fidelity",
        "",
    ]
    fidelity = (
        node_summary.groupby("model")
        .agg(
            mean_fidelity_abs_error=("fidelity_abs_error", "mean"),
            median_fidelity_abs_error=("fidelity_abs_error", "median"),
            mean_subgraph_nodes=("subgraph_nodes", "mean"),
            mean_subgraph_edges=("subgraph_edges", "mean"),
        )
        .reset_index()
    )
    lines.extend(
        [
            "| model | mean_fidelity_abs_error | median_fidelity_abs_error | mean_subgraph_nodes | mean_subgraph_edges |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in fidelity.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.mean_fidelity_abs_error:.4f} | "
            f"{row.median_fidelity_abs_error:.4f} | {row.mean_subgraph_nodes:.1f} | "
            f"{row.mean_subgraph_edges:.1f} |"
        )
    lines.extend(["", "## Top Features", ""])
    for model_name, group in global_importance.groupby("model", sort=False):
        lines.append(f"### {str(model_name).upper()}")
        lines.append("")
        lines.extend(["| rank | feature | mean_importance |", "| --- | --- | --- |"])
        for i, row in enumerate(
            group.sort_values("mean_importance", ascending=False).head(12).itertuples(index=False),
            start=1,
        ):
            lines.append(f"| {i} | {row.feature} | {row.mean_importance:.4f} |")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(int(args.torch_threads))
    cities = validate_cities([str(city) for city in args.cities])
    graph_inputs = load_multicity_graph(cities, args.interim_root, args.target_column)
    graph_inputs, feature_config_metadata = apply_gnn_feature_config(graph_inputs, args)
    labels, splits = single_spatial_labels(graph_inputs, cities, args)
    sampled_nodes = sample_test_nodes(
        graph_inputs.frame,
        labels,
        max_nodes=int(args.max_test_nodes),
        nodes_per_city=int(args.nodes_per_city),
        seed=int(args.seed),
    )

    node_rows: list[dict[str, Any]] = []
    feature_frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    model_metrics: dict[str, Any] = {}

    for raw_model_name in args.models:
        model_name: ModelName = raw_model_name  # type: ignore[assignment]
        trained = train_model(model_name, graph_inputs, labels, args)
        test_mask = labels.eq("test").to_numpy()
        model_metrics[model_name] = {
            "best_epoch": int(trained.best_epoch),
            "best_val_loss_scaled": float(trained.best_val_loss),
            "test_metrics": regression_metrics(trained.y_true[test_mask], trained.y_pred[test_mask]),
        }
        for i, row in enumerate(sampled_nodes.itertuples(index=False), start=1):
            node_row, feature_rows, edge_rows = explain_one_node(
                trained=trained,
                graph_inputs=graph_inputs,
                node_index=int(row.node_index),
                args=args,
            )
            node_rows.append(node_row)
            feature_frames.append(feature_rows)
            if not edge_rows.empty:
                edge_frames.append(edge_rows)
            if int(args.progress_every) > 0 and (
                i == 1 or i % int(args.progress_every) == 0 or i == len(sampled_nodes)
            ):
                print(
                    json.dumps(
                        {
                            "model": model_name,
                            "explained_nodes": i,
                            "total_nodes": int(len(sampled_nodes)),
                            "last_city": str(row.city),
                            "last_fidelity_abs_error": round(
                                float(node_row["fidelity_abs_error"]),
                                6,
                            ),
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    node_summary = pd.DataFrame(node_rows)
    feature_importance = pd.concat(feature_frames, ignore_index=True)
    edge_importance = (
        pd.concat(edge_frames, ignore_index=True)
        if edge_frames
        else pd.DataFrame()
    )
    global_importance = (
        feature_importance.groupby(["model", "feature"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            median_importance=("importance", "median"),
            std_importance=("importance", "std"),
            n_explained_nodes=("node_id", "nunique"),
        )
        .fillna(0.0)
        .sort_values(["model", "mean_importance"], ascending=[True, False])
    )
    city_importance = (
        feature_importance.groupby(["model", "city", "feature"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            n_explained_nodes=("node_id", "nunique"),
        )
        .sort_values(["model", "city", "mean_importance"], ascending=[True, True, False])
    )

    output_root = Path(args.outputs_root)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    metrics_dir = output_root / "metrics"
    reports_dir = output_root / "reports"
    for path in [tables_dir, figures_dir, metrics_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    node_path = tables_dir / f"{args.output_prefix}_node_explanations.csv"
    feature_path = tables_dir / f"{args.output_prefix}_feature_masks.csv"
    global_path = tables_dir / f"{args.output_prefix}_global_feature_importance.csv"
    city_path = tables_dir / f"{args.output_prefix}_city_feature_importance.csv"
    edge_path = tables_dir / f"{args.output_prefix}_edge_importance.csv"
    node_summary.to_csv(node_path, index=False)
    feature_importance.to_csv(feature_path, index=False)
    global_importance.to_csv(global_path, index=False)
    city_importance.to_csv(city_path, index=False)
    edge_importance.to_csv(edge_path, index=False)

    top_features_path = figures_dir / f"{args.output_prefix}_top_features.png"
    city_heatmap_path = figures_dir / f"{args.output_prefix}_city_heatmap.png"
    plot_top_features(global_importance, top_features_path, top_n=int(args.top_n_features))
    plot_city_heatmap(city_importance, city_heatmap_path, top_n=12)

    metadata: dict[str, Any] = {
        "models": list(args.models),
        "cities": cities,
        "target_column": args.target_column,
        "feature_set": str(args.feature_set),
        "feature_view": str(args.feature_view),
        "feature_config_metadata": feature_config_metadata,
        "n_rows": int(len(graph_inputs.frame)),
        "n_edges": int(graph_inputs.adjacency.nnz // 2),
        "n_features": int(len(graph_inputs.feature_columns)),
        "n_explained_nodes": int(len(sampled_nodes)),
        "nodes_per_city": int(args.nodes_per_city),
        "khop": int(args.khop),
        "explain_steps": int(args.explain_steps),
        "explainer_hyperparameters": {
            "explain_learning_rate": float(args.explain_learning_rate),
            "feature_size_penalty": float(args.feature_size_penalty),
            "edge_size_penalty": float(args.edge_size_penalty),
            "mask_entropy_penalty": float(args.mask_entropy_penalty),
        },
        "training_hyperparameters": {
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
        },
        "model_metrics": model_metrics,
        "artifacts": {
            "node_explanations": str(node_path),
            "feature_masks": str(feature_path),
            "global_feature_importance": str(global_path),
            "city_feature_importance": str(city_path),
            "edge_importance": str(edge_path),
            "top_features_figure": str(top_features_path),
            "city_heatmap_prefix": str(city_heatmap_path),
            "report": str(reports_dir / f"{args.output_prefix}_report.md"),
            "spatial_splits": str(tables_dir / f"{args.output_prefix}_spatial_splits.csv"),
        },
    }
    splits.to_csv(tables_dir / f"{args.output_prefix}_spatial_splits.csv", index=False)
    summary_path = metrics_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(
        reports_dir / f"{args.output_prefix}_report.md",
        global_importance,
        node_summary,
        metadata,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

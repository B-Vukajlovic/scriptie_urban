"""Evaluate PyTorch Geometric GNNs and official GNNExplainer attributions."""

from __future__ import annotations

import argparse
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
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_multicity_gnn import (  # noqa: E402
    GraphInputs,
    apply_gnn_feature_config,
    build_single_city_splits,
    load_multicity_graph,
    standardize_train_only,
)
from src.models.metrics import regression_metrics  # noqa: E402
from src.models.pyg_gnn import PyGGCNRegressor, PyGGraphSAGERegressor  # noqa: E402
from src.utils.cities import DEFAULT_CITIES, validate_cities  # noqa: E402

PyGModelName = Literal["gcn", "graphsage"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official PyTorch Geometric GCN/GraphSAGE and GNNExplainer."
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--max-explain-nodes", type=int, default=100)
    parser.add_argument("--explain-nodes-per-city", type=int, default=10)
    parser.add_argument("--explainer-epochs", type=int, default=60)
    parser.add_argument("--explainer-lr", type=float, default=0.01)
    parser.add_argument("--top-n-features", type=int, default=20)
    parser.add_argument("--output-prefix", default="multicity_pyg_gnn_reduced")
    return parser.parse_args()


def build_edge_index(adjacency: sparse.csr_matrix) -> torch.Tensor:
    coo = adjacency.tocoo()
    return torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)


def single_spatial_labels(
    graph_inputs: GraphInputs,
    cities: list[str],
    args: argparse.Namespace,
) -> tuple[pd.Series, pd.DataFrame]:
    splits = build_single_city_splits(cities, args.interim_root, args)
    merged = graph_inputs.frame[["node_id"]].merge(
        splits[["node_id", "split"]],
        on="node_id",
        how="left",
        validate="one_to_one",
    )
    if merged["split"].isna().any():
        raise ValueError("Spatial split does not cover every graph node.")
    return merged["split"], splits


def make_model(
    model_name: PyGModelName,
    in_dim: int,
    hidden_dim: int,
    dropout: float,
) -> nn.Module:
    if model_name == "gcn":
        return PyGGCNRegressor(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    if model_name == "graphsage":
        return PyGGraphSAGERegressor(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unknown PyG model: {model_name}")


def train_pyg_model(
    model_name: PyGModelName,
    graph_inputs: GraphInputs,
    labels: pd.Series,
    edge_index: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    frame = graph_inputs.frame
    train_mask = labels.eq("train").to_numpy()
    val_mask = labels.eq("val").to_numpy()
    test_mask = labels.eq("test").to_numpy()
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Train, val, and test splits must all be non-empty.")
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

    model = make_model(
        model_name,
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
        pred = model(x, edge_index)
        loss = loss_fn(pred[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_eval = model(x, edge_index)
            val_loss = float(loss_fn(pred_eval[val_idx], y[val_idx]).item())
        if int(args.progress_every) > 0 and (epoch == 1 or epoch % int(args.progress_every) == 0):
            print(
                json.dumps(
                    {
                        "backend": "pyg",
                        "model": model_name,
                        "epoch": epoch,
                        "train_loss": round(float(loss.item()), 6),
                        "val_loss": round(val_loss, 6),
                        "best_epoch": best_epoch,
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
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
        pred_scaled = model(x, edge_index).detach().cpu().numpy()
    y_pred = pred_scaled * data.y_std + data.y_mean
    y_true = frame[args.target_column].astype(float).to_numpy()

    predictions = frame[["city", "block_id", args.target_column]].copy()
    predictions = predictions.rename(columns={args.target_column: "y_true"})
    predictions.insert(0, "model", model_name)
    predictions.insert(0, "backend", "pyg")
    predictions["split"] = labels.to_numpy()
    predictions["y_pred"] = y_pred.astype(float)
    predictions["node_index"] = np.arange(len(predictions), dtype=int)

    train_mean = float(y_true[train_mask].mean())
    metric_rows: list[dict[str, Any]] = []
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        metrics = regression_metrics(y_true[mask], y_pred[mask])
        baseline = regression_metrics(y_true[mask], np.full(mask.sum(), train_mean, dtype=float))
        metric_rows.append(
            {
                "backend": "pyg",
                "model": model_name,
                "split": split_name,
                "city": "__all__",
                **metrics,
                "baseline_rmse": baseline["rmse"],
                "baseline_r2": baseline["r2"],
                "n_rows": int(mask.sum()),
            }
        )
    for city, group in predictions[predictions["split"] == "test"].groupby("city", sort=True):
        metric_rows.append(
            {
                "backend": "pyg",
                "model": model_name,
                "split": "test",
                "city": str(city),
                **regression_metrics(
                    group["y_true"].to_numpy(dtype=float),
                    group["y_pred"].to_numpy(dtype=float),
                ),
                "baseline_rmse": np.nan,
                "baseline_r2": np.nan,
                "n_rows": int(len(group)),
            }
        )
    summary = {
        "best_epoch": int(best_epoch),
        "best_val_loss_scaled": float(best_val_loss),
    }
    return model, summary, pd.DataFrame(metric_rows), predictions


def sample_explain_nodes(
    predictions: pd.DataFrame,
    max_nodes: int,
    nodes_per_city: int,
    seed: int,
) -> pd.DataFrame:
    candidates = predictions[predictions["split"] == "test"].copy()
    if candidates.empty:
        return predictions.iloc[0:0].copy()
    parts: list[pd.DataFrame] = []
    for _city, group in candidates.groupby("city", sort=False):
        n = min(len(group), nodes_per_city)
        parts.append(group.sample(n=n, random_state=seed) if len(group) > n else group)
    sampled = pd.concat(parts, ignore_index=True)
    if max_nodes > 0 and len(sampled) > max_nodes:
        sampled = sampled.sample(n=max_nodes, random_state=seed)
    return sampled.sort_values(["city", "node_index"]).reset_index(drop=True)


def explain_model(
    model: nn.Module,
    model_name: str,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    predictions: pd.DataFrame,
    feature_columns: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=int(args.explainer_epochs), lr=float(args.explainer_lr)),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="regression", task_level="node", return_type="raw"),
    )
    sampled = sample_explain_nodes(
        predictions,
        max_nodes=int(args.max_explain_nodes),
        nodes_per_city=int(args.explain_nodes_per_city),
        seed=int(args.seed),
    )
    if sampled.empty:
        return (
            pd.DataFrame(
                columns=[
                    "backend",
                    "model",
                    "node_index",
                    "city",
                    "block_id",
                    "y_true",
                    "y_pred",
                    "subgraph_nodes",
                    "subgraph_edges",
                    "mean_edge_importance",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "backend",
                    "model",
                    "node_index",
                    "city",
                    "block_id",
                    "feature",
                    "importance",
                ]
            ),
        )
    node_rows: list[dict[str, Any]] = []
    feature_rows: list[pd.DataFrame] = []
    for i, row in enumerate(sampled.itertuples(index=False), start=1):
        node_index = int(np.asarray(row.node_index).item())
        subset, sub_edge_index, mapping, _edge_mask = k_hop_subgraph(
            node_idx=node_index,
            num_hops=2,
            edge_index=edge_index,
            relabel_nodes=True,
        )
        sub_x = x[subset]
        local_index = int(mapping.item())
        explanation = explainer(sub_x, sub_edge_index, index=local_index)
        node_mask = explanation.node_mask.detach().cpu()
        if node_mask.ndim == 2:
            feature_importance = node_mask.abs().mean(dim=0).numpy()
        else:
            feature_importance = node_mask.abs().numpy()
        edge_importance = (
            explanation.edge_mask.detach().cpu().abs().numpy()
            if explanation.edge_mask is not None
            else np.array([], dtype=float)
        )
        node_rows.append(
            {
                "backend": "pyg",
                "model": model_name,
                "node_index": node_index,
                "city": str(row.city),
                "block_id": str(row.block_id),
                "y_true": float(np.asarray(row.y_true).item()),
                "y_pred": float(np.asarray(row.y_pred).item()),
                "subgraph_nodes": int(len(subset)),
                "subgraph_edges": int(sub_edge_index.shape[1]),
                "mean_edge_importance": float(edge_importance.mean()) if len(edge_importance) else 0.0,
            }
        )
        feature_rows.append(
            pd.DataFrame(
                {
                    "backend": "pyg",
                    "model": model_name,
                    "node_index": node_index,
                    "city": str(row.city),
                    "block_id": str(row.block_id),
                    "feature": feature_columns,
                    "importance": feature_importance.astype(float),
                }
            )
        )
        if int(args.progress_every) > 0 and (
            i == 1 or i % int(args.progress_every) == 0 or i == len(sampled)
        ):
            print(
                json.dumps(
                    {
                        "backend": "pyg",
                        "model": model_name,
                        "explained_nodes": i,
                        "total_nodes": int(len(sampled)),
                        "last_city": str(row.city),
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
    return pd.DataFrame(node_rows), pd.concat(feature_rows, ignore_index=True)


def plot_feature_importance(global_importance: pd.DataFrame, path: Path, top_n: int) -> None:
    models = global_importance["model"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 5 * len(models)), squeeze=False)
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
        ax.barh(labels, top["mean_importance"], color="#355f8c")
        ax.set_title(f"Official PyG GNNExplainer: {model_name}")
        ax.set_xlabel("Mean node-mask importance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(int(args.torch_threads))
    cities = validate_cities([str(city) for city in args.cities])
    graph_inputs = load_multicity_graph(cities, args.interim_root, args.target_column)
    graph_inputs, feature_config_metadata = apply_gnn_feature_config(graph_inputs, args)
    labels, splits = single_spatial_labels(graph_inputs, cities, args)
    edge_index = build_edge_index(graph_inputs.adjacency)
    train_mask = labels.eq("train").to_numpy()
    data = standardize_train_only(
        graph_inputs.frame,
        graph_inputs.feature_columns,
        args.target_column,
        train_mask,
    )
    x = torch.tensor(data.x, dtype=torch.float32)

    model_summaries: dict[str, Any] = {}
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    explanation_node_frames: list[pd.DataFrame] = []
    explanation_feature_frames: list[pd.DataFrame] = []

    for raw_model in args.models:
        model_name: PyGModelName = raw_model  # type: ignore[assignment]
        model, train_summary, metrics, predictions = train_pyg_model(
            model_name,
            graph_inputs,
            labels,
            edge_index,
            args,
        )
        model_summaries[model_name] = train_summary
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
        if bool(args.explain):
            nodes, features = explain_model(
                model,
                model_name,
                x,
                edge_index,
                predictions,
                graph_inputs.feature_columns,
                args,
            )
            explanation_node_frames.append(nodes)
            explanation_feature_frames.append(features)

    metrics_df = pd.concat(metric_frames, ignore_index=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    output_root = Path(args.outputs_root)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    metrics_dir = output_root / "metrics"
    reports_dir = output_root / "reports"
    for path in [tables_dir, figures_dir, metrics_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    predictions_path = tables_dir / f"{args.output_prefix}_predictions.csv"
    splits_path = tables_dir / f"{args.output_prefix}_pooled_spatial_splits.csv"
    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    splits.to_csv(splits_path, index=False)

    artifacts: dict[str, str] = {
        "metrics": str(metrics_path),
        "predictions": str(predictions_path),
        "pooled_spatial_splits": str(splits_path),
    }
    explanation_summary: dict[str, Any] = {}
    if explanation_feature_frames:
        node_explanations = pd.concat(explanation_node_frames, ignore_index=True)
        feature_masks = pd.concat(explanation_feature_frames, ignore_index=True)
        global_importance = (
            feature_masks.groupby(["backend", "model", "feature"], as_index=False)
            .agg(
                mean_importance=("importance", "mean"),
                median_importance=("importance", "median"),
                n_explained_nodes=("node_index", "nunique"),
            )
            .sort_values(["model", "mean_importance"], ascending=[True, False])
        )
        node_path = tables_dir / f"{args.output_prefix}_gnnexplainer_nodes.csv"
        masks_path = tables_dir / f"{args.output_prefix}_gnnexplainer_feature_masks.csv"
        importance_path = tables_dir / f"{args.output_prefix}_gnnexplainer_global_feature_importance.csv"
        fig_path = figures_dir / f"{args.output_prefix}_gnnexplainer_top_features.png"
        node_explanations.to_csv(node_path, index=False)
        feature_masks.to_csv(masks_path, index=False)
        global_importance.to_csv(importance_path, index=False)
        plot_feature_importance(global_importance, fig_path, top_n=int(args.top_n_features))
        artifacts.update(
            {
                "gnnexplainer_nodes": str(node_path),
                "gnnexplainer_feature_masks": str(masks_path),
                "gnnexplainer_global_feature_importance": str(importance_path),
                "gnnexplainer_top_features": str(fig_path),
            }
        )
        explanation_summary = {
            "n_explained_nodes_per_model": int(args.max_explain_nodes),
            "explainer": "torch_geometric.explain.GNNExplainer",
        }

    summary = {
        "backend": "pyg",
        "models": list(args.models),
        "target_column": args.target_column,
        "feature_set": str(args.feature_set),
        "feature_view": str(args.feature_view),
        "feature_config_metadata": feature_config_metadata,
        "n_rows": int(len(graph_inputs.frame)),
        "n_edges": int(graph_inputs.adjacency.nnz // 2),
        "n_features": int(len(graph_inputs.feature_columns)),
        "hyperparameters": {
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
        },
        "model_summaries": model_summaries,
        "test_metrics": metrics_df[
            (metrics_df["split"] == "test") & (metrics_df["city"] == "__all__")
        ].to_dict(orient="records"),
        "explanation_summary": explanation_summary,
        "artifacts": artifacts,
    }
    summary_path = metrics_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = reports_dir / f"{args.output_prefix}_report.md"
    lines = [
        "# Official PyTorch Geometric GNN Evaluation",
        "",
        f"- Target: `{args.target_column}`",
        f"- Features: `{args.feature_set}` / `{args.feature_view}`",
        f"- Nodes: {len(graph_inputs.frame)}",
        f"- Undirected edges: {graph_inputs.adjacency.nnz // 2}",
        "",
        "## Test Metrics",
        "",
        "| model | rmse | mae | r2 |",
        "| --- | --- | --- | --- |",
    ]
    for row in summary["test_metrics"]:
        lines.append(
            f"| {row['model']} | {float(row['rmse']):.4f} | "
            f"{float(row['mae']):.4f} | {float(row['r2']):.4f} |"
        )
    if explanation_summary:
        lines.extend(
            [
                "",
                "## Explainability",
                "",
                "Official `torch_geometric.explain.GNNExplainer` was run on sampled held-out test nodes using 2-hop subgraphs.",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

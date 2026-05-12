"""Train true GCN and GraphSAGE regressors on the multi-city block graph."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Literal
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph.adjacency import build_undirected_adjacency  # noqa: E402
from src.evaluation.spatial_splits import build_spatial_train_val_test_splits  # noqa: E402
from src.modeling.dataset import load_modeling_table  # noqa: E402
from src.models.gnn import GCNRegressor, GraphSAGERegressor  # noqa: E402
from src.models.metrics import regression_metrics  # noqa: E402
from src.utils.cities import DEFAULT_CITIES, validate_cities  # noqa: E402
from scripts.evaluate_multicity_xgboost import (  # noqa: E402
    build_feature_view,
    select_feature_set,
)

warnings.filterwarnings(
    "ignore",
    message="Sparse invariant checks are implicitly disabled.*",
    category=UserWarning,
)

ModelName = Literal["gcn", "graphsage"]


@dataclass(frozen=True)
class GraphInputs:
    frame: pd.DataFrame
    feature_columns: list[str]
    adjacency: sparse.csr_matrix
    metadata_by_city: dict[str, Any]


@dataclass(frozen=True)
class StandardizedData:
    x: np.ndarray
    y_scaled: np.ndarray
    y_mean: float
    y_std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate true GCN/GraphSAGE node regressors on the multi-city graph."
    )
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    parser.add_argument("--interim-root", default="data/interim")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument(
        "--feature-set",
        choices=["full", "reduced"],
        default="reduced",
        help="Feature set used as GNN node attributes.",
    )
    parser.add_argument(
        "--feature-view",
        choices=["raw", "log1p"],
        default="log1p",
        help="Feature transformation applied before train-only standardization.",
    )
    parser.add_argument(
        "--reduced-feature-set",
        default="data/interim/modeling/reduced_feature_set.json",
        help="JSON feature list used when --feature-set reduced.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["gcn", "graphsage"],
        default=["gcn", "graphsage"],
    )
    parser.add_argument("--target-column", default="Y_global_log_minmax")
    parser.add_argument(
        "--evaluation-mode",
        choices=["single_spatial_cv", "leave_one_city_out"],
        default="single_spatial_cv",
        help=(
            "Use one generated spatial split or hold out each whole city."
        ),
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
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print training progress every N epochs; 0 disables progress logs.",
    )
    parser.add_argument(
        "--output-prefix",
        default="multicity_gnn",
        help="Prefix for metrics/prediction artifacts.",
    )
    return parser.parse_args()


def apply_gnn_feature_config(
    graph_inputs: GraphInputs,
    args: argparse.Namespace,
) -> tuple[GraphInputs, dict[str, Any]]:
    """Apply the thesis feature set/view before train-only GNN standardization."""
    selected_features, feature_set_metadata = select_feature_set(
        graph_inputs.feature_columns,
        str(args.feature_set),
        args.reduced_feature_set,
    )
    frame, view_features, feature_view_metadata = build_feature_view(
        graph_inputs.frame,
        selected_features,
        str(args.feature_view),
    )
    configured = GraphInputs(
        frame=frame,
        feature_columns=view_features,
        adjacency=graph_inputs.adjacency,
        metadata_by_city=graph_inputs.metadata_by_city,
    )
    metadata = {
        **feature_set_metadata,
        **feature_view_metadata,
        "n_features": int(len(view_features)),
    }
    return configured, metadata


def _std(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def string_key_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append({str(key): value for key, value in row.items()})
    return records


def summarize_columns(
    df: pd.DataFrame,
    group_col: str,
    value_cols: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for group_value, group_df in df.groupby(group_col):
        group_summary: dict[str, dict[str, float]] = {}
        for col in value_cols:
            values = group_df[col].astype(float)
            group_summary[col] = {
                "mean": float(values.mean()),
                "std": _std(values),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        summary[str(group_value)] = group_summary
    return summary


def _row_normalize(adjacency: sparse.csr_matrix, self_loops: bool) -> sparse.csr_matrix:
    graph = adjacency.astype(np.float32)
    if self_loops:
        graph = graph + sparse.eye(graph.shape[0], dtype=np.float32, format="csr")
    row_sums = np.asarray(graph.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sums, dtype=np.float32)
    nonzero = row_sums > 0
    inv[nonzero] = 1.0 / row_sums[nonzero]
    return (sparse.diags(inv, format="csr") @ graph).tocsr()


def _symmetric_normalize_with_self_loops(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    graph = adjacency.astype(np.float32) + sparse.eye(
        adjacency.shape[0],
        dtype=np.float32,
        format="csr",
    )
    degree = np.asarray(graph.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    normalizer = sparse.diags(inv_sqrt, format="csr")
    return (normalizer @ graph @ normalizer).tocsr()


def _torch_sparse(matrix: sparse.csr_matrix) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(
        indices,
        values,
        coo.shape,
        check_invariants=False,
    ).coalesce()


def load_city_inputs(
    city: str,
    interim_root: str | Path,
    expected_features: list[str] | None,
    target_column: str,
) -> tuple[pd.DataFrame, list[str], sparse.csr_matrix, dict[str, Any]]:
    dataset, feature_columns, metadata = load_modeling_table(city, interim_root)
    if expected_features is not None and feature_columns != expected_features:
        raise ValueError(f"{city} feature columns do not match previous cities.")
    required = {"block_id", "split", target_column, *feature_columns}
    missing = required - set(dataset.columns)
    if missing:
        raise KeyError(f"{city} modeling table is missing columns: {sorted(missing)}")

    frame = dataset[["block_id", "split", target_column, *feature_columns]].copy()
    frame["block_id"] = frame["block_id"].astype(str)
    frame.insert(0, "city", city)
    frame.insert(0, "node_id", city + ":" + frame["block_id"].astype(str))
    edges_path = Path(interim_root) / city / "backbone" / "adjacency.csv"
    if not edges_path.exists():
        raise FileNotFoundError(f"Adjacency not found: {edges_path}")
    edges = pd.read_csv(edges_path, dtype={"src_block_id": "string", "dst_block_id": "string"})
    adjacency = build_undirected_adjacency(frame["block_id"], edges)
    city_metadata = {
        "n_rows": int(len(frame)),
        "n_edges": int(adjacency.nnz // 2),
        "split_counts": frame["split"].value_counts().to_dict(),
        "feature_metadata": metadata.get("inputs", {}),
        "adjacency": str(edges_path),
    }
    return frame, feature_columns, adjacency, city_metadata


def load_target_coordinates(city: str, interim_root: str | Path) -> pd.DataFrame:
    target_path = Path(interim_root) / city / "target" / "target_table.parquet"
    if not target_path.exists():
        raise FileNotFoundError(f"Target table not found: {target_path}")
    coords = pd.read_parquet(target_path, columns=["block_id", "x_m", "y_m"])
    coords["block_id"] = coords["block_id"].astype(str)
    coords.insert(0, "city", city)
    coords.insert(0, "node_id", city + ":" + coords["block_id"].astype(str))
    return coords


def build_single_city_splits(
    cities: list[str],
    interim_root: str | Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    splits: list[pd.DataFrame] = []
    for city in cities:
        coords = load_target_coordinates(city, interim_root)
        split_df = build_spatial_train_val_test_splits(
            coords[["block_id", "x_m", "y_m"]],
            seed=args.split_seed,
            val_frac=args.split_val_frac,
            test_frac=args.split_test_frac,
            grid_bins_x=args.split_grid_bins_x,
            grid_bins_y=args.split_grid_bins_y,
        )
        split_df.insert(0, "seed", int(args.split_seed))
        split_df.insert(0, "repeat", 0)
        split_df.insert(0, "city", city)
        split_df.insert(0, "node_id", city + ":" + split_df["block_id"].astype(str))
        splits.append(split_df)
    return pd.concat(splits, ignore_index=True)


def load_multicity_graph(
    cities: list[str],
    interim_root: str | Path,
    target_column: str,
) -> GraphInputs:
    frames: list[pd.DataFrame] = []
    adjacencies: list[sparse.csr_matrix] = []
    expected_features: list[str] | None = None
    metadata_by_city: dict[str, Any] = {}

    for city in cities:
        frame, feature_columns, adjacency, city_metadata = load_city_inputs(
            city,
            interim_root,
            expected_features,
            target_column,
        )
        if expected_features is None:
            expected_features = feature_columns
        frames.append(frame)
        adjacencies.append(adjacency)
        metadata_by_city[city] = city_metadata

    if expected_features is None:
        raise ValueError("No cities provided.")
    graph = sparse.block_diag(adjacencies, format="csr", dtype=np.float32)
    return GraphInputs(
        frame=pd.concat(frames, ignore_index=True),
        feature_columns=expected_features,
        adjacency=graph,
        metadata_by_city=metadata_by_city,
    )


def standardize_train_only(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    train_mask: np.ndarray,
) -> StandardizedData:
    x_raw = frame[feature_columns].astype(float).to_numpy(dtype=np.float32)
    y_raw = frame[target_column].astype(float).to_numpy(dtype=np.float32)

    x_mean = x_raw[train_mask].mean(axis=0)
    x_std = x_raw[train_mask].std(axis=0)
    x_std[x_std == 0] = 1.0
    x = (x_raw - x_mean) / x_std

    y_mean = float(y_raw[train_mask].mean())
    y_std = float(y_raw[train_mask].std())
    if y_std == 0:
        y_std = 1.0
    y_scaled = (y_raw - y_mean) / y_std
    return StandardizedData(x=x, y_scaled=y_scaled, y_mean=y_mean, y_std=y_std)


def make_model(
    model_name: ModelName,
    in_dim: int,
    hidden_dim: int,
    dropout: float,
) -> nn.Module:
    if model_name == "gcn":
        return GCNRegressor(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    if model_name == "graphsage":
        return GraphSAGERegressor(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unknown GNN model: {model_name}")


def train_one_model(
    model_name: ModelName,
    graph_inputs: GraphInputs,
    args: argparse.Namespace,
    *,
    split_labels: pd.Series,
    repeat: int,
    split_seed: int,
    model_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)

    frame = graph_inputs.frame
    labels = split_labels.reset_index(drop=True)
    if len(labels) != len(frame):
        raise ValueError("Split labels must have one value per graph node.")
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

    if model_name == "gcn":
        adjacency = _torch_sparse(_symmetric_normalize_with_self_loops(graph_inputs.adjacency))
    else:
        adjacency = _torch_sparse(_row_normalize(graph_inputs.adjacency, self_loops=False))

    model = make_model(
        model_name=model_name,
        in_dim=len(graph_inputs.feature_columns),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = args.patience
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, adjacency)
        loss = loss_fn(pred[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_eval = model(x, adjacency)
            val_loss = float(loss_fn(pred_eval[val_idx], y[val_idx]).item())
            train_loss = float(loss.item())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if args.progress_every > 0 and (epoch == 1 or epoch % args.progress_every == 0):
            print(
                json.dumps(
                    {
                        "model": model_name,
                        "repeat": repeat,
                        "epoch": epoch,
                        "train_loss": round(train_loss, 6),
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
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x, adjacency).detach().cpu().numpy()
    y_pred = pred_scaled * data.y_std + data.y_mean
    y_true = frame[args.target_column].astype(float).to_numpy()

    prediction_df = frame[["city", "block_id", args.target_column]].copy()
    prediction_df = prediction_df.rename(columns={args.target_column: "y_true"})
    prediction_df["model"] = model_name
    prediction_df["repeat"] = int(repeat)
    prediction_df["seed"] = int(split_seed)
    prediction_df["model_seed"] = int(model_seed)
    prediction_df["split"] = labels.to_numpy()
    prediction_df["y_pred"] = y_pred.astype(float)

    metrics_by_split: dict[str, dict[str, float]] = {}
    baseline_by_split: dict[str, dict[str, float]] = {}
    train_mean = float(y_true[train_mask].mean())
    for split_name, mask in [
        ("train", train_mask),
        ("val", val_mask),
        ("test", test_mask),
    ]:
        metrics_by_split[split_name] = regression_metrics(y_true[mask], y_pred[mask])
        baseline_pred = np.full(mask.sum(), train_mean, dtype=float)
        baseline_by_split[split_name] = regression_metrics(y_true[mask], baseline_pred)

    city_rows: list[dict[str, Any]] = []
    for city, group in prediction_df.groupby("city", sort=True):
        mask = group["split"].eq("test").to_numpy()
        if not mask.any():
            continue
        city_rows.append(
            {
                "city": city,
                **regression_metrics(
                    group.loc[mask, "y_true"].to_numpy(),
                    group.loc[mask, "y_pred"].to_numpy(),
                ),
                "n_test_rows": int(mask.sum()),
            }
        )

    summary = {
        "model": model_name,
        "repeat": int(repeat),
        "seed": int(split_seed),
        "model_seed": int(model_seed),
        "n_epochs_run": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_val_loss_scaled": float(best_val_loss),
        "metrics_by_split": metrics_by_split,
        "train_mean_baseline_metrics": baseline_by_split,
        "test_metrics_by_city": city_rows,
        "training_history_tail": history[-10:],
    }
    return summary, prediction_df


def append_metrics(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    experiment: str,
    repeat: int,
    seed: int,
    split: str,
    city: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    train_mean: float,
) -> None:
    metrics = regression_metrics(y_true.to_numpy(), y_pred)
    baseline_pred = np.full(len(y_true), train_mean, dtype=float)
    baseline = regression_metrics(y_true.to_numpy(), baseline_pred)
    rows.append(
        {
            "model": model_name,
            "experiment": experiment,
            "repeat": int(repeat),
            "seed": int(seed),
            "split": split,
            "city": city,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "baseline_rmse": baseline["rmse"],
            "baseline_mae": baseline["mae"],
            "baseline_r2": baseline["r2"],
            "rmse_improvement_vs_baseline_pct": float(
                100 * (baseline["rmse"] - metrics["rmse"]) / baseline["rmse"]
            ),
            "n_rows": int(len(y_true)),
            "target_mean": float(y_true.mean()),
            "target_std": _std(y_true),
            "prediction_mean": float(np.mean(y_pred)),
            "prediction_bias": float(np.mean(y_pred) - y_true.mean()),
        }
    )


def build_metric_rows(
    predictions: pd.DataFrame,
    experiment: str,
    target_column: str = "y_true",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_name, repeat), group in predictions.groupby(["model", "repeat"], sort=True):
        seed = int(group["seed"].iloc[0])
        train = group[group["split"] == "train"]
        train_mean = float(train[target_column].mean())
        for split_name in ["train", "val", "test"]:
            split_df = group[group["split"] == split_name]
            append_metrics(
                rows,
                model_name=str(model_name),
                experiment=experiment,
                repeat=int(repeat),
                seed=seed,
                split=split_name,
                city="__all__",
                y_true=split_df[target_column],
                y_pred=split_df["y_pred"].to_numpy(dtype=float),
                train_mean=train_mean,
            )
            if split_name == "test":
                for city, city_df in split_df.groupby("city", sort=True):
                    append_metrics(
                        rows,
                        model_name=str(model_name),
                        experiment=experiment,
                        repeat=int(repeat),
                        seed=seed,
                        split="test",
                        city=str(city),
                        y_true=city_df[target_column],
                        y_pred=city_df["y_pred"].to_numpy(dtype=float),
                        train_mean=train_mean,
                    )
    return pd.DataFrame(rows)


def single_spatial_split_labels(
    graph_inputs: GraphInputs,
    cities: list[str],
    args: argparse.Namespace,
) -> tuple[list[tuple[int, int, pd.Series]], pd.DataFrame]:
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
    return [(0, int(args.split_seed), merged["split"])], splits


def leave_one_city_out_labels(
    graph_inputs: GraphInputs,
    cities: list[str],
    args: argparse.Namespace,
) -> list[tuple[int, int, pd.Series]]:
    """Build LOCO masks while preserving held-in validation rows for early stopping."""
    labels_by_city: list[tuple[int, int, pd.Series]] = []
    frame = graph_inputs.frame
    for repeat, heldout_city in enumerate(cities):
        labels = pd.Series("train", index=frame.index, dtype="object")
        heldout_mask = frame["city"].eq(heldout_city)
        labels.loc[heldout_mask] = "test"
        labels.loc[~heldout_mask & frame["split"].eq("val")] = "val"
        seed = int(args.split_seed + repeat)
        labels_by_city.append((repeat, seed, labels))
    return labels_by_city


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    cities = validate_cities(args.cities)
    graph_inputs = load_multicity_graph(cities, args.interim_root, args.target_column)
    graph_inputs, feature_config_metadata = apply_gnn_feature_config(graph_inputs, args)

    if args.evaluation_mode == "leave_one_city_out":
        split_jobs = leave_one_city_out_labels(graph_inputs, cities, args)
        splits_artifact: pd.DataFrame | None = None
        experiment = "leave_one_city_out"
    else:
        split_jobs, splits_artifact = single_spatial_split_labels(graph_inputs, cities, args)
        experiment = "single_spatial_cv"

    summaries: dict[str, list[dict[str, Any]]] = {model_name: [] for model_name in args.models}
    prediction_frames: list[pd.DataFrame] = []
    for repeat, split_seed, labels in split_jobs:
        for model_name in args.models:
            summary, predictions = train_one_model(
                model_name,
                graph_inputs,
                args,
                split_labels=labels,
                repeat=repeat,
                split_seed=int(split_seed),
                model_seed=int(args.seed + repeat),
            )
            summaries[model_name].append(summary)
            prediction_frames.append(predictions)

    outputs_root = Path(args.outputs_root)
    metrics_dir = outputs_root / "metrics"
    tables_dir = outputs_root / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    predictions_path = tables_dir / f"{args.output_prefix}_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    metrics_df = build_metric_rows(predictions_df, experiment)
    metrics_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    splits_path: Path | None = None
    if splits_artifact is not None:
        splits_path = tables_dir / f"{args.output_prefix}_pooled_spatial_splits.csv"
        splits_artifact.to_csv(splits_path, index=False)

    metric_cols = [
        "rmse",
        "mae",
        "r2",
        "baseline_rmse",
        "baseline_r2",
        "rmse_improvement_vs_baseline_pct",
        "prediction_bias",
    ]
    global_metrics = metrics_df[metrics_df["city"] == "__all__"]
    summary_by_model: dict[str, Any] = {}
    for model_name, model_metrics in global_metrics.groupby("model", sort=True):
        summary_by_model[str(model_name)] = summarize_columns(
            model_metrics,
            "split",
            metric_cols,
        )

    payload = {
        "models": list(args.models),
        "cities": cities,
        "target_column": args.target_column,
        "evaluation_mode": args.evaluation_mode,
        "feature_set": str(args.feature_set),
        "feature_view": str(args.feature_view),
        "feature_config_metadata": feature_config_metadata,
        "n_rows": int(len(graph_inputs.frame)),
        "n_features": int(len(graph_inputs.feature_columns)),
        "n_edges": int(graph_inputs.adjacency.nnz // 2),
        "n_spatial_splits": int(len(split_jobs)) if args.evaluation_mode != "leave_one_city_out" else 0,
        "split_seed": int(args.split_seed),
        "split_config": {
            "val_frac": float(args.split_val_frac),
            "test_frac": float(args.split_test_frac),
            "grid_bins_x": int(args.split_grid_bins_x),
            "grid_bins_y": int(args.split_grid_bins_y),
        },
        "feature_columns": graph_inputs.feature_columns,
        "city_inputs": graph_inputs.metadata_by_city,
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "torch_threads": args.torch_threads,
            "progress_every": args.progress_every,
        },
        "summary_by_model": summary_by_model,
        "test_metrics_by_city": {
            str(model_name): string_key_records(
                metrics_df[
                    (metrics_df["model"] == model_name)
                    & (metrics_df["split"] == "test")
                    & (metrics_df["city"] != "__all__")
                ].sort_values(["repeat", "city"])
            )
            for model_name in args.models
        },
        "results": summaries,
        "artifacts": {
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
            **(
                {"pooled_spatial_splits": str(splits_path)}
                if splits_path is not None
                else {}
            ),
        },
    }
    summary_path = metrics_dir / f"{args.output_prefix}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

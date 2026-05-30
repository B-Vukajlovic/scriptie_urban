"""Cluster GNNExplainer feature-mask signatures into explanation groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_multicity_xgboost import string_key_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster official PyG GNNExplainer feature-mask vectors. "
            "Cluster IDs are ordered by mean predicted accessibility."
        )
    )
    parser.add_argument(
        "--feature-masks",
        default=(
            "outputs/tables/"
            "multicity_pyg_gnn_expanded_block_only_pt_be_gnnexplainer_feature_masks.csv"
        ),
    )
    parser.add_argument(
        "--nodes",
        default=(
            "outputs/tables/"
            "multicity_pyg_gnn_expanded_block_only_pt_be_gnnexplainer_nodes.csv"
        ),
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--top-n-features", type=int, default=14)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-prefix",
        default="multicity_pyg_gnn_expanded_block_only_pt_be_gnnexplainer_clusters",
    )
    return parser.parse_args()


def _short_label(feature: str) -> str:
    return (
        feature.removeprefix("pt_")
        .removeprefix("be_")
        .removeprefix("acs_")
        .replace("_", " ")
    )


def _family(feature: str) -> str:
    if feature.startswith("pt_"):
        return "PT"
    if feature.startswith("be_"):
        return "BE"
    if feature.startswith("acs_"):
        return "ACS"
    return "other"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{float(value):.4f}")
        else:
            display[column] = display[column].astype(str)
    lines = [
        "| " + " | ".join(display.columns.astype(str)) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def load_inputs(
    feature_masks_path: str | Path,
    nodes_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    masks = pd.read_csv(feature_masks_path, dtype={"block_id": str})
    nodes = pd.read_csv(nodes_path, dtype={"block_id": str})
    required_masks = {
        "model",
        "node_index",
        "city",
        "block_id",
        "feature",
        "importance",
    }
    required_nodes = {
        "model",
        "node_index",
        "city",
        "block_id",
        "y_true",
        "y_pred",
    }
    missing_masks = required_masks - set(masks.columns)
    missing_nodes = required_nodes - set(nodes.columns)
    if missing_masks:
        raise KeyError(f"Feature masks missing columns: {sorted(missing_masks)}")
    if missing_nodes:
        raise KeyError(f"Node explanations missing columns: {sorted(missing_nodes)}")
    return masks, nodes


def wide_masks(masks: pd.DataFrame) -> pd.DataFrame:
    wide = masks.pivot_table(
        index=["model", "node_index", "city", "block_id"],
        columns="feature",
        values="importance",
        aggfunc="mean",
        fill_value=0.0,
    ).reset_index()
    wide.columns.name = None
    return wide


def build_model_clusters(
    model_wide: pd.DataFrame,
    nodes: pd.DataFrame,
    feature_columns: list[str],
    n_clusters: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    if len(model_wide) <= n_clusters:
        raise ValueError("Need more explained nodes than clusters.")
    matrix = model_wide[feature_columns].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(matrix)
    raw_labels = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=50,
    ).fit_predict(scaled)
    silhouette = float(silhouette_score(scaled, raw_labels))
    assignments = model_wide[["model", "node_index", "city", "block_id"]].copy()
    assignments["raw_cluster_id"] = raw_labels.astype(int)
    assignments = assignments.merge(
        nodes[
            [
                "model",
                "node_index",
                "y_true",
                "y_pred",
                "subgraph_nodes",
                "subgraph_edges",
                "mean_edge_importance",
            ]
        ],
        on=["model", "node_index"],
        how="left",
        validate="one_to_one",
    )
    if assignments[["y_true", "y_pred"]].isna().any().any():
        raise ValueError("Failed to align GNN cluster assignments with node predictions.")
    order = (
        assignments.groupby("raw_cluster_id")["y_pred"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )
    remap = {int(raw): int(new) for new, raw in enumerate(order)}
    assignments["cluster_id"] = assignments["raw_cluster_id"].map(remap).astype(int)
    assignments = assignments.drop(columns=["raw_cluster_id"])
    assignments["cluster_label"] = "C" + assignments["cluster_id"].astype(str)
    return assignments, matrix, silhouette


def feature_profiles(
    model_wide: pd.DataFrame,
    assignments: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    joined = model_wide[["model", "node_index", *feature_columns]].merge(
        assignments[["model", "node_index", "cluster_id", "cluster_label"]],
        on=["model", "node_index"],
        how="inner",
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for cluster_id, group in joined.groupby("cluster_id", sort=True):
        for feature in feature_columns:
            values = group[feature].astype(float).to_numpy()
            rows.append(
                {
                    "model": str(group["model"].iloc[0]),
                    "cluster_id": int(cluster_id),
                    "cluster_label": f"C{int(cluster_id)}",
                    "feature": feature,
                    "family": _family(feature),
                    "mean_importance": float(values.mean()),
                    "median_importance": float(np.median(values)),
                    "std_importance": float(values.std(ddof=0)),
                }
            )
    profiles = pd.DataFrame(rows)
    profiles["rank_within_cluster"] = (
        profiles.groupby(["model", "cluster_id"])["mean_importance"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return profiles.sort_values(["model", "cluster_id", "rank_within_cluster"])


def family_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    out = (
        profiles.groupby(["model", "cluster_id", "cluster_label", "family"], as_index=False)
        .agg(total_importance=("mean_importance", "sum"), n_features=("feature", "size"))
        .sort_values(["model", "cluster_id", "family"])
    )
    totals = out.groupby(["model", "cluster_id"])["total_importance"].transform("sum")
    out["family_share"] = np.where(totals > 0, out["total_importance"] / totals, 0.0)
    return out


def cluster_summary(
    assignments: pd.DataFrame,
    profiles: pd.DataFrame,
    family: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(assignments)
    for (model, cluster_id), group in assignments.groupby(["model", "cluster_id"], sort=True):
        profile = profiles[
            profiles["model"].eq(model) & profiles["cluster_id"].eq(cluster_id)
        ]
        family_group = family[
            family["model"].eq(model) & family["cluster_id"].eq(cluster_id)
        ].copy()
        city_share = (
            group["city"]
            .value_counts(normalize=True)
            .sort_values(ascending=False)
            .head(5)
            .round(4)
            .to_dict()
        )
        family_share = {
            str(row.family): float(row.family_share)
            for row in family_group.itertuples(index=False)
        }
        rows.append(
            {
                "model": str(model),
                "cluster_id": int(cluster_id),
                "cluster_label": f"C{int(cluster_id)}",
                "n_rows": int(len(group)),
                "row_share_within_model": float(len(group) / total),
                "mean_y_true": float(group["y_true"].mean()),
                "mean_y_pred": float(group["y_pred"].mean()),
                "mean_prediction_error": float((group["y_pred"] - group["y_true"]).mean()),
                "mean_subgraph_nodes": float(group["subgraph_nodes"].mean()),
                "mean_subgraph_edges": float(group["subgraph_edges"].mean()),
                "mean_edge_importance": float(group["mean_edge_importance"].mean()),
                "top_cities": json.dumps({str(k): float(v) for k, v in city_share.items()}),
                "family_shares": json.dumps(family_share),
                "top_features": "; ".join(
                    profile.sort_values("mean_importance", ascending=False)
                    .head(top_n)["feature"]
                    .astype(str)
                    .tolist()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_feature_heatmap(profiles: pd.DataFrame, model: str, top_features: list[str], path: Path) -> None:
    work = profiles[profiles["model"].eq(model) & profiles["feature"].isin(top_features)]
    heatmap = (
        work.pivot(index="cluster_label", columns="feature", values="mean_importance")
        .reindex(columns=top_features)
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(top_features)), 5.5))
    image = ax.imshow(heatmap.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(top_features)))
    ax.set_xticklabels([_short_label(feature) for feature in top_features], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index.tolist())
    ax.set_title(f"{model.upper()} GNNExplainer cluster feature importance")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Explanation cluster")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean feature-mask importance")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_family_heatmap(family: pd.DataFrame, model: str, path: Path) -> None:
    work = family[family["model"].eq(model)]
    heatmap = (
        work.pivot(index="cluster_label", columns="family", values="family_share")
        .reindex(columns=["PT", "BE", "ACS", "other"])
        .dropna(axis=1, how="all")
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    image = ax.imshow(heatmap.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns.tolist())
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index.tolist())
    ax.set_title(f"{model.upper()} PT/BE explanation-family balance")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Share of cluster feature-mask importance")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_city_composition(assignments: pd.DataFrame, model: str, path: Path) -> None:
    work = assignments[assignments["model"].eq(model)]
    composition = pd.crosstab(
        work["cluster_label"],
        work["city"],
        normalize="index",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros(len(composition), dtype=float)
    for city in composition.columns:
        values = composition[city].to_numpy(dtype=float)
        ax.bar(composition.index, values, bottom=bottom, label=str(city))
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of cluster")
    ax.set_xlabel("Explanation cluster")
    ax.set_title(f"{model.upper()} city composition of GNNExplainer clusters")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pca(
    assignments: pd.DataFrame,
    matrix: np.ndarray,
    model: str,
    path: Path,
) -> pd.DataFrame:
    work = assignments[assignments["model"].eq(model)].copy()
    scaled = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(scaled)
    work["pc1"] = coords[:, 0]
    work["pc2"] = coords[:, 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster_label, group in work.groupby("cluster_label", sort=True):
        ax.scatter(group["pc1"], group["pc2"], s=28, alpha=0.75, label=cluster_label)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% of mask variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% of mask variance)")
    ax.set_title(f"{model.upper()} GNNExplainer signature clusters")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return work[
        [
            "model",
            "node_index",
            "city",
            "block_id",
            "cluster_id",
            "cluster_label",
            "y_true",
            "y_pred",
            "pc1",
            "pc2",
        ]
    ]


def plot_pt_be_balance(family: pd.DataFrame, assignments: pd.DataFrame, path: Path) -> None:
    pivot = (
        family.pivot_table(
            index=["model", "cluster_id", "cluster_label"],
            columns="family",
            values="family_share",
            aggfunc="mean",
            fill_value=0.0,
        )
        .reset_index()
    )
    mean_pred = (
        assignments.groupby(["model", "cluster_id"], as_index=False)["y_pred"]
        .mean()
        .rename(columns={"y_pred": "mean_y_pred"})
    )
    pivot = pivot.merge(mean_pred, on=["model", "cluster_id"], how="left")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for model, group in pivot.groupby("model", sort=True):
        ax.scatter(
            group.get("PT", pd.Series(0.0, index=group.index)),
            group.get("BE", pd.Series(0.0, index=group.index)),
            s=220,
            alpha=0.78,
            label=str(model).upper(),
        )
        for row in group.itertuples(index=False):
            ax.text(
                float(getattr(row, "PT", 0.0)) + 0.004,
                float(getattr(row, "BE", 0.0)) + 0.004,
                f"{row.cluster_label}",
                fontsize=9,
            )
    ax.set_xlabel("PT share of explanation importance")
    ax.set_ylabel("BE share of explanation importance")
    ax.set_title("GNNExplainer PT/BE balance by cluster")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(
    report_path: Path,
    summary: pd.DataFrame,
    profiles: pd.DataFrame,
    family: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    lines = [
        "# GNNExplainer Signature Clusters",
        "",
        "This report clusters official PyG GNNExplainer feature-mask vectors for sampled held-out test nodes.",
        "Unlike SHAP values, feature masks are non-directional: larger values mean a feature was more important for preserving the node prediction, not that it raised or lowered the prediction.",
        "",
        "## Run Metadata",
        "",
        f"- Explained nodes: {metadata['n_nodes']}",
        f"- Features: {metadata['n_features']}",
        f"- Clusters per model: {metadata['n_clusters']}",
        "",
        "## Cluster Summary",
        "",
        dataframe_to_markdown(
            summary[
                [
                    "model",
                    "cluster_label",
                    "n_rows",
                    "mean_y_true",
                    "mean_y_pred",
                    "family_shares",
                    "top_features",
                ]
            ]
        ),
        "",
        "## Top Features By Model",
        "",
    ]
    for model, group in profiles.groupby("model", sort=True):
        lines.append(f"### {str(model).upper()}")
        lines.append("")
        top = (
            group.groupby("feature", as_index=False)["mean_importance"]
            .mean()
            .sort_values("mean_importance", ascending=False)
            .head(12)
        )
        lines.append(dataframe_to_markdown(top))
        lines.append("")
    lines.append("## PT/BE Family Balance")
    lines.append("")
    lines.append(
        dataframe_to_markdown(
            family[
                [
                    "model",
                    "cluster_label",
                    "family",
                    "family_share",
                ]
            ]
        )
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    masks, nodes = load_inputs(args.feature_masks, args.nodes)
    wide = wide_masks(masks)
    feature_columns = [
        col
        for col in wide.columns
        if col not in {"model", "node_index", "city", "block_id"}
    ]
    output_root = Path(args.outputs_root)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    metrics_dir = output_root / "metrics"
    reports_dir = output_root / "reports"
    for path in [tables_dir, figures_dir, metrics_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    assignment_frames: list[pd.DataFrame] = []
    profile_frames: list[pd.DataFrame] = []
    pca_frames: list[pd.DataFrame] = []
    silhouettes: dict[str, float] = {}
    matrices_by_model: dict[str, np.ndarray] = {}

    for model, model_wide in wide.groupby("model", sort=True):
        assignments, matrix, silhouette = build_model_clusters(
            model_wide=model_wide,
            nodes=nodes[nodes["model"].eq(model)].copy(),
            feature_columns=feature_columns,
            n_clusters=int(args.n_clusters),
            random_state=int(args.random_state),
        )
        profiles = feature_profiles(model_wide, assignments, feature_columns)
        assignment_frames.append(assignments)
        profile_frames.append(profiles)
        silhouettes[str(model)] = float(silhouette)
        matrices_by_model[str(model)] = matrix

    assignments = pd.concat(assignment_frames, ignore_index=True)
    profiles = pd.concat(profile_frames, ignore_index=True)
    family = family_profiles(profiles)
    summary = cluster_summary(assignments, profiles, family, top_n=int(args.top_n_features))

    top_features = (
        profiles.groupby("feature")["mean_importance"]
        .mean()
        .sort_values(ascending=False)
        .head(int(args.top_n_features))
        .index.astype(str)
        .tolist()
    )

    for model in sorted(assignments["model"].unique()):
        model_assignments = assignments[assignments["model"].eq(model)]
        model_profiles = profiles[profiles["model"].eq(model)]
        plot_feature_heatmap(
            model_profiles,
            str(model),
            top_features,
            figures_dir / f"{args.output_prefix}_{model}_feature_heatmap.png",
        )
        plot_family_heatmap(
            family,
            str(model),
            figures_dir / f"{args.output_prefix}_{model}_family_heatmap.png",
        )
        plot_city_composition(
            model_assignments,
            str(model),
            figures_dir / f"{args.output_prefix}_{model}_city_composition.png",
        )
        pca_frames.append(
            plot_pca(
                assignments,
                matrices_by_model[str(model)],
                str(model),
                figures_dir / f"{args.output_prefix}_{model}_pca.png",
            )
        )

    plot_pt_be_balance(
        family,
        assignments,
        figures_dir / f"{args.output_prefix}_pt_be_balance.png",
    )
    pca_projection = pd.concat(pca_frames, ignore_index=True)

    assignments_path = tables_dir / f"{args.output_prefix}_assignments.csv"
    profiles_path = tables_dir / f"{args.output_prefix}_feature_profiles.csv"
    family_path = tables_dir / f"{args.output_prefix}_family_profiles.csv"
    summary_path = tables_dir / f"{args.output_prefix}_summary.csv"
    pca_path = tables_dir / f"{args.output_prefix}_pca_projection.csv"
    assignments.to_csv(assignments_path, index=False)
    profiles.to_csv(profiles_path, index=False)
    family.to_csv(family_path, index=False)
    summary.to_csv(summary_path, index=False)
    pca_projection.to_csv(pca_path, index=False)

    metadata: dict[str, Any] = {
        "feature_masks": str(args.feature_masks),
        "nodes": str(args.nodes),
        "n_nodes": int(len(assignments)),
        "n_features": int(len(feature_columns)),
        "n_clusters": int(args.n_clusters),
        "models": sorted(str(model) for model in assignments["model"].unique()),
        "silhouette_scores": silhouettes,
        "top_features": top_features,
        "artifacts": {
            "assignments": str(assignments_path),
            "feature_profiles": str(profiles_path),
            "family_profiles": str(family_path),
            "summary": str(summary_path),
            "pca_projection": str(pca_path),
            "report": str(reports_dir / f"{args.output_prefix}_report.md"),
            "pt_be_balance": str(figures_dir / f"{args.output_prefix}_pt_be_balance.png"),
        },
        "clusters": string_key_records(summary),
    }
    metadata_path = metrics_dir / f"{args.output_prefix}_summary.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(
        reports_dir / f"{args.output_prefix}_report.md",
        summary,
        profiles,
        family,
        metadata,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

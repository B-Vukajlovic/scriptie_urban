"""Cluster XGBoost SHAP signatures into explanation-based access zones."""

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
            "Cluster held-out XGBoost SHAP vectors so blocks with similar model "
            "explanations can be interpreted as access-pattern zones."
        )
    )
    parser.add_argument(
        "--shap-values",
        default="outputs/tables/multicity_xgboost_shap_reduced_values.parquet",
        help="Parquet file produced by explain_multicity_xgboost_shap.py.",
    )
    parser.add_argument(
        "--predictions",
        default="outputs/tables/multicity_xgboost_shap_reduced_predictions.csv",
        help="Prediction file produced with the same SHAP run.",
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--top-n-features", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-prefix",
        default="multicity_xgboost_shap_reduced_clusters",
        help="Artifact prefix used under outputs/tables, outputs/figures, etc.",
    )
    return parser.parse_args()


def _feature_from_shap_column(column: str) -> str:
    if not column.startswith("shap__"):
        raise ValueError(f"Not a SHAP column: {column}")
    return column.removeprefix("shap__")


def _short_feature_label(feature: str) -> str:
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
    """Render a small markdown table without requiring pandas' tabulate extra."""
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{float(value):.4f}")
        else:
            display[column] = display[column].astype(str)
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def load_inputs(shap_path: str | Path, predictions_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    shap_df = pd.read_parquet(shap_path)
    predictions = pd.read_csv(predictions_path, dtype={"block_id": str})
    required_shap = {"repeat", "seed", "node_id", "city", "block_id"}
    required_pred = {"repeat", "seed", "node_id", "city", "block_id", "y_true", "y_pred"}
    missing_shap = required_shap - set(shap_df.columns)
    missing_pred = required_pred - set(predictions.columns)
    if missing_shap:
        raise KeyError(f"SHAP table is missing columns: {sorted(missing_shap)}")
    if missing_pred:
        raise KeyError(f"Prediction table is missing columns: {sorted(missing_pred)}")
    return shap_df, predictions


def build_assignments(
    shap_df: pd.DataFrame,
    predictions: pd.DataFrame,
    n_clusters: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    shap_cols = [col for col in shap_df.columns if col.startswith("shap__")]
    if len(shap_cols) < 2:
        raise ValueError("Need at least two SHAP columns for clustering.")
    if n_clusters < 2:
        raise ValueError("--n-clusters must be at least 2.")
    if n_clusters >= len(shap_df):
        raise ValueError("--n-clusters must be smaller than the number of explained rows.")

    shap_matrix = shap_df[shap_cols].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(shap_matrix)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=50)
    raw_labels = kmeans.fit_predict(scaled)
    silhouette = float(silhouette_score(scaled, raw_labels)) if n_clusters > 1 else float("nan")

    assignments = shap_df[["repeat", "seed", "node_id", "city", "block_id"]].copy()
    assignments["raw_cluster_id"] = raw_labels.astype(int)
    assignments = assignments.merge(
        predictions[["repeat", "seed", "node_id", "y_true", "y_pred", "shap_base_value"]],
        on=["repeat", "seed", "node_id"],
        how="left",
        validate="one_to_one",
    )
    if assignments[["y_true", "y_pred"]].isna().any().any():
        raise ValueError("Failed to align SHAP cluster assignments with predictions.")

    # Make cluster IDs interpretable and stable: 0 is lowest predicted accessibility.
    order = (
        assignments.groupby("raw_cluster_id", sort=False)["y_pred"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )
    remap = {int(raw): int(new) for new, raw in enumerate(order)}
    assignments["cluster_id"] = assignments["raw_cluster_id"].map(remap).astype(int)
    assignments = assignments.drop(columns=["raw_cluster_id"])
    assignments["cluster_label"] = "C" + assignments["cluster_id"].astype(str)
    return assignments, shap_matrix, silhouette


def cluster_feature_profiles(
    shap_df: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    shap_cols = [col for col in shap_df.columns if col.startswith("shap__")]
    shap_with_cluster = shap_df[["node_id", *shap_cols]].merge(
        assignments[["node_id", "cluster_id"]],
        on="node_id",
        how="inner",
        validate="one_to_one",
    )
    rows: list[dict[str, float | int | str]] = []
    for cluster_id, group in shap_with_cluster.groupby("cluster_id", sort=True):
        for col in shap_cols:
            values = group[col].to_numpy(dtype=float)
            feature = _feature_from_shap_column(col)
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "cluster_label": f"C{int(cluster_id)}",
                    "feature": feature,
                    "family": _family(feature),
                    "mean_shap": float(np.mean(values)),
                    "mean_abs_shap": float(np.mean(np.abs(values))),
                }
            )
    profiles = pd.DataFrame(rows)
    profiles["abs_rank_within_cluster"] = (
        profiles.groupby("cluster_id")["mean_abs_shap"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return profiles.sort_values(["cluster_id", "abs_rank_within_cluster"])


def cluster_summary(assignments: pd.DataFrame, profiles: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(assignments)
    for cluster_id, group in assignments.groupby("cluster_id", sort=True):
        profile = profiles[profiles["cluster_id"] == cluster_id]
        top_abs = profile.sort_values("mean_abs_shap", ascending=False).head(top_n)
        top_pos = profile.sort_values("mean_shap", ascending=False).head(top_n)
        top_neg = profile.sort_values("mean_shap", ascending=True).head(top_n)
        city_share = (
            group["city"]
            .value_counts(normalize=True)
            .sort_values(ascending=False)
            .head(5)
            .round(4)
            .to_dict()
        )
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_label": f"C{int(cluster_id)}",
                "n_rows": int(len(group)),
                "row_share": float(len(group) / total),
                "mean_y_true": float(group["y_true"].mean()),
                "mean_y_pred": float(group["y_pred"].mean()),
                "mean_prediction_error": float((group["y_pred"] - group["y_true"]).mean()),
                "top_cities": json.dumps({str(k): float(v) for k, v in city_share.items()}),
                "top_abs_features": "; ".join(top_abs["feature"].astype(str).tolist()),
                "top_positive_features": "; ".join(top_pos["feature"].astype(str).tolist()),
                "top_negative_features": "; ".join(top_neg["feature"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows)


def plot_cluster_heatmap(
    profiles: pd.DataFrame,
    top_features: list[str],
    output_path: Path,
) -> None:
    heatmap = (
        profiles[profiles["feature"].isin(top_features)]
        .pivot(index="cluster_label", columns="feature", values="mean_shap")
        .reindex(columns=top_features)
        .fillna(0.0)
    )
    fig_width = max(10.0, 0.45 * len(top_features))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    vmax = float(np.nanmax(np.abs(heatmap.to_numpy(dtype=float))))
    image = ax.imshow(heatmap.to_numpy(dtype=float), cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(top_features)))
    ax.set_xticklabels([_short_feature_label(feature) for feature in top_features], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index.tolist())
    ax.set_title("Mean SHAP contribution by explanation cluster")
    ax.set_xlabel("Feature")
    ax.set_ylabel("SHAP cluster")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean SHAP value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_city_composition(assignments: pd.DataFrame, output_path: Path) -> None:
    composition = pd.crosstab(
        assignments["cluster_label"],
        assignments["city"],
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
    ax.set_xlabel("SHAP cluster")
    ax.set_title("City composition of SHAP explanation clusters")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_cluster_pca(assignments: pd.DataFrame, shap_matrix: np.ndarray, output_path: Path) -> pd.DataFrame:
    scaled = StandardScaler().fit_transform(shap_matrix)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(scaled)
    plot_df = assignments[["node_id", "city", "cluster_id", "cluster_label", "y_pred"]].copy()
    plot_df["pc1"] = coords[:, 0]
    plot_df["pc2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster_label, group in plot_df.groupby("cluster_label", sort=True):
        ax.scatter(group["pc1"], group["pc2"], s=14, alpha=0.65, label=cluster_label)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% of SHAP variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% of SHAP variance)")
    ax.set_title("SHAP signature clusters projected to two dimensions")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return plot_df


def write_report(
    report_path: Path,
    summary: pd.DataFrame,
    profiles: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    lines = [
        "# XGBoost SHAP Signature Clusters",
        "",
        "This report clusters held-out block-level SHAP vectors from the reduced XGBoost model.",
        "Cluster IDs are ordered by mean predicted accessibility, so C0 is the lowest-access explanation cluster.",
        "",
        "## Run Metadata",
        "",
        f"- Explained rows: {metadata['n_rows']}",
        f"- SHAP features: {metadata['n_features']}",
        f"- Clusters: {metadata['n_clusters']}",
        f"- Silhouette score: {metadata['silhouette_score']:.4f}",
        "",
        "## Cluster Summary",
        "",
        summary[
            [
                "cluster_label",
                "n_rows",
                "row_share",
                "mean_y_true",
                "mean_y_pred",
                "top_abs_features",
            ]
        ].pipe(dataframe_to_markdown),
        "",
        "## Dominant Feature Contributions",
        "",
    ]
    for cluster_id, group in profiles.groupby("cluster_id", sort=True):
        top_pos = group.sort_values("mean_shap", ascending=False).head(5)
        top_neg = group.sort_values("mean_shap", ascending=True).head(5)
        lines.append(f"### C{int(cluster_id)}")
        lines.append("")
        lines.append("Positive contributors:")
        for row in top_pos.itertuples(index=False):
            lines.append(f"- {row.feature}: {row.mean_shap:.4f}")
        lines.append("")
        lines.append("Negative contributors:")
        for row in top_neg.itertuples(index=False):
            lines.append(f"- {row.feature}: {row.mean_shap:.4f}")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.outputs_root)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    reports_dir = output_root / "reports"
    metrics_dir = output_root / "metrics"
    for path in [tables_dir, figures_dir, reports_dir, metrics_dir]:
        path.mkdir(parents=True, exist_ok=True)

    shap_df, predictions = load_inputs(args.shap_values, args.predictions)
    assignments, shap_matrix, silhouette = build_assignments(
        shap_df=shap_df,
        predictions=predictions,
        n_clusters=int(args.n_clusters),
        random_state=int(args.random_state),
    )
    profiles = cluster_feature_profiles(shap_df, assignments)
    summary = cluster_summary(assignments, profiles, top_n=int(args.top_n_features))

    top_features = (
        profiles.groupby("feature")["mean_abs_shap"]
        .mean()
        .sort_values(ascending=False)
        .head(int(args.top_n_features))
        .index
        .astype(str)
        .tolist()
    )
    pca_df = plot_cluster_pca(
        assignments=assignments,
        shap_matrix=shap_matrix,
        output_path=figures_dir / f"{args.output_prefix}_pca.png",
    )
    plot_cluster_heatmap(
        profiles=profiles,
        top_features=top_features,
        output_path=figures_dir / f"{args.output_prefix}_heatmap.png",
    )
    plot_city_composition(
        assignments=assignments,
        output_path=figures_dir / f"{args.output_prefix}_city_composition.png",
    )

    assignments_path = tables_dir / f"{args.output_prefix}_assignments.csv"
    summary_path = tables_dir / f"{args.output_prefix}_summary.csv"
    profiles_path = tables_dir / f"{args.output_prefix}_feature_profiles.csv"
    pca_path = tables_dir / f"{args.output_prefix}_pca_projection.csv"
    assignments.to_csv(assignments_path, index=False)
    summary.to_csv(summary_path, index=False)
    profiles.to_csv(profiles_path, index=False)
    pca_df.to_csv(pca_path, index=False)

    metadata: dict[str, Any] = {
        "shap_values": str(args.shap_values),
        "predictions": str(args.predictions),
        "n_rows": int(len(assignments)),
        "n_features": int(len([col for col in shap_df.columns if col.startswith("shap__")])),
        "n_clusters": int(args.n_clusters),
        "silhouette_score": float(silhouette),
        "top_features": top_features,
        "artifacts": {
            "assignments": str(assignments_path),
            "summary": str(summary_path),
            "feature_profiles": str(profiles_path),
            "pca_projection": str(pca_path),
            "pca_figure": str(figures_dir / f"{args.output_prefix}_pca.png"),
            "heatmap": str(figures_dir / f"{args.output_prefix}_heatmap.png"),
            "city_composition": str(figures_dir / f"{args.output_prefix}_city_composition.png"),
            "report": str(reports_dir / f"{args.output_prefix}_report.md"),
        },
        "clusters": string_key_records(summary),
    }
    metadata_path = metrics_dir / f"{args.output_prefix}_summary.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(
        report_path=reports_dir / f"{args.output_prefix}_report.md",
        summary=summary,
        profiles=profiles,
        metadata=metadata,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

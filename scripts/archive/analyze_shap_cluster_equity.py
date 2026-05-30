"""Attach ACS context to SHAP clusters for post-hoc equity interpretation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.cities import DEFAULT_CITIES, validate_cities  # noqa: E402

ACS_CONTEXT_COLUMNS = [
    "acs_population_density_per_km2",
    "acs_median_household_income",
    "acs_poverty_share",
    "acs_unemployment_share",
    "acs_zero_vehicle_household_share",
    "acs_disability_share",
    "acs_age_under_18_share",
    "acs_age_65_plus_share",
    "acs_bachelor_or_higher_share",
    "acs_public_transit_commute_share",
    "acs_car_commute_share",
    "acs_walk_commute_share",
    "acs_commute_60_plus_min_share",
    "acs_black_share",
    "acs_hispanic_share",
    "acs_asian_share",
    "acs_white_non_hispanic_share",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize ACS variables by XGBoost SHAP cluster. This is post-hoc: "
            "ACS variables are not model inputs for the main PT+BE comparison."
        )
    )
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    parser.add_argument("--interim-root", default="data/interim")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument(
        "--cluster-assignments",
        default=(
            "outputs/tables/"
            "multicity_xgboost_shap_expanded_block_only_pt_be_clusters_assignments.csv"
        ),
    )
    parser.add_argument("--target-column", default="Y_global_log_minmax")
    parser.add_argument(
        "--acs-columns",
        nargs="+",
        default=ACS_CONTEXT_COLUMNS,
        help="ACS columns to summarize by cluster.",
    )
    parser.add_argument(
        "--output-prefix",
        default="multicity_xgboost_shap_expanded_block_only_pt_be_equity",
    )
    return parser.parse_args()


def _short_label(feature: str) -> str:
    return feature.removeprefix("acs_").replace("_", " ")


def load_acs_context(
    cities: list[str],
    interim_root: str | Path,
    target_column: str,
    acs_columns: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    needed = ["block_id", target_column, *acs_columns]
    for city in cities:
        path = Path(interim_root) / city / "modeling" / "model_dataset.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Model dataset not found: {path}")
        columns = pd.read_parquet(path).columns.tolist()
        available = [col for col in needed if col in columns]
        missing = sorted(set(["block_id", target_column]) - set(available))
        if missing:
            raise KeyError(f"{path} is missing required columns: {missing}")
        city_frame = pd.read_parquet(path, columns=available)
        city_frame.insert(0, "city", city)
        frames.append(city_frame)
    context = pd.concat(frames, ignore_index=True)
    context["block_id"] = context["block_id"].astype(str)
    return context


def summarize_by_cluster(
    merged: pd.DataFrame,
    acs_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available_acs = [col for col in acs_columns if col in merged.columns]
    if not available_acs:
        raise ValueError("No requested ACS columns are available in the merged cluster table.")

    global_mean = merged[available_acs].astype(float).mean()
    global_std = merged[available_acs].astype(float).std(ddof=0).replace(0.0, np.nan)

    rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    for cluster_label, group in merged.groupby("cluster_label", sort=True):
        cluster_id = int(group["cluster_id"].iloc[0])
        row: dict[str, Any] = {
            "cluster_id": cluster_id,
            "cluster_label": str(cluster_label),
            "n_rows": int(len(group)),
            "row_share": float(len(group) / len(merged)),
            "mean_y_true": float(group["y_true"].mean()),
            "mean_y_pred": float(group["y_pred"].mean()),
            f"mean_{target_column}": float(group[target_column].mean()),
        }
        for col in available_acs:
            values = group[col].astype(float)
            mean_value = float(values.mean())
            row[f"mean_{col}"] = mean_value
            row[f"median_{col}"] = float(values.median())
            std = float(global_std[col]) if pd.notna(global_std[col]) else 0.0
            standardized = (
                (mean_value - float(global_mean[col])) / std
                if std > 0
                else 0.0
            )
            profile_rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_label": str(cluster_label),
                    "acs_variable": col,
                    "acs_label": _short_label(col),
                    "cluster_mean": mean_value,
                    "overall_mean": float(global_mean[col]),
                    "standardized_difference": float(standardized),
                }
            )
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("cluster_id")
    profiles = pd.DataFrame(profile_rows).sort_values(
        ["cluster_id", "standardized_difference"],
        ascending=[True, False],
    )
    return summary, profiles


def plot_equity_heatmap(profiles: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    order = (
        profiles.groupby("acs_variable")["standardized_difference"]
        .apply(lambda values: float(np.nanmax(np.abs(values.to_numpy(dtype=float)))))
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    labels = (
        profiles.drop_duplicates("acs_variable")
        .set_index("acs_variable")
        .loc[order, "acs_label"]
        .tolist()
    )
    heatmap = (
        profiles.pivot_table(
            index="acs_variable",
            columns="cluster_label",
            values="standardized_difference",
            aggfunc="mean",
            fill_value=0.0,
        )
        .reindex(order)
    )
    vmax = float(np.nanmax(np.abs(heatmap.to_numpy(dtype=float))))
    vmax = max(vmax, 0.1)
    fig, ax = plt.subplots(figsize=(9.5, max(6.0, 0.38 * len(order))))
    image = ax.imshow(heatmap.to_numpy(dtype=float), cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns.tolist())
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Post-hoc ACS context by SHAP explanation cluster")
    ax.set_xlabel("SHAP cluster ordered by predicted accessibility")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Standardized difference from explained-sample mean")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
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


def write_report(
    report_path: Path,
    summary: pd.DataFrame,
    profiles: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    low_cluster = summary.sort_values("mean_y_pred").iloc[0]
    low_profiles = profiles[
        profiles["cluster_label"].eq(str(low_cluster["cluster_label"]))
    ].copy()
    low_profiles["abs_standardized_difference"] = low_profiles[
        "standardized_difference"
    ].abs()
    low_top = low_profiles.sort_values(
        "abs_standardized_difference",
        ascending=False,
    ).head(10)
    lines = [
        "# Post-Hoc Equity Context for SHAP Clusters",
        "",
        "ACS variables were not used as predictors in the main PT+BE model.",
        "They are attached after clustering to interpret whether explanation-based access patterns overlap with sociodemographic vulnerability.",
        "",
        "## Run Metadata",
        "",
        f"- Cluster assignments: `{metadata['cluster_assignments']}`",
        f"- Explained rows with ACS context: `{metadata['n_rows']}`",
        f"- ACS variables: `{metadata['n_acs_variables']}`",
        "",
        "## Cluster Accessibility",
        "",
        dataframe_to_markdown(
            summary[
                [
                    "cluster_label",
                    "n_rows",
                    "row_share",
                    "mean_y_true",
                    "mean_y_pred",
                ]
            ]
        ),
        "",
        f"## Lowest Predicted Accessibility Cluster: {low_cluster['cluster_label']}",
        "",
        dataframe_to_markdown(
            low_top[
                [
                    "acs_label",
                    "cluster_mean",
                    "overall_mean",
                    "standardized_difference",
                ]
            ]
        ),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cities = validate_cities([str(city) for city in args.cities])
    assignments_path = Path(args.cluster_assignments)
    if not assignments_path.exists():
        raise FileNotFoundError(f"Cluster assignments not found: {assignments_path}")
    assignments = pd.read_csv(assignments_path, dtype={"block_id": str})
    required = {
        "city",
        "block_id",
        "cluster_id",
        "cluster_label",
        "y_true",
        "y_pred",
    }
    missing = required - set(assignments.columns)
    if missing:
        raise KeyError(f"Cluster assignment table is missing columns: {sorted(missing)}")

    acs_context = load_acs_context(
        cities=cities,
        interim_root=args.interim_root,
        target_column=str(args.target_column),
        acs_columns=[str(col) for col in args.acs_columns],
    )
    merged = assignments.merge(
        acs_context,
        on=["city", "block_id"],
        how="left",
        validate="many_to_one",
    )
    if merged[str(args.target_column)].isna().any():
        missing_rows = int(merged[str(args.target_column)].isna().sum())
        raise ValueError(f"Failed to attach ACS/model context for {missing_rows} rows.")

    summary, profiles = summarize_by_cluster(
        merged=merged,
        acs_columns=[str(col) for col in args.acs_columns],
        target_column=str(args.target_column),
    )

    output_root = Path(args.outputs_root)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    metrics_dir = output_root / "metrics"
    reports_dir = output_root / "reports"
    for path in [tables_dir, figures_dir, metrics_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    merged_path = tables_dir / f"{args.output_prefix}_cluster_acs_rows.csv"
    summary_path = tables_dir / f"{args.output_prefix}_cluster_summary.csv"
    profiles_path = tables_dir / f"{args.output_prefix}_acs_profiles.csv"
    heatmap_path = figures_dir / f"{args.output_prefix}_acs_heatmap.png"
    report_path = reports_dir / f"{args.output_prefix}_report.md"
    metadata_path = metrics_dir / f"{args.output_prefix}_summary.json"

    merged.to_csv(merged_path, index=False)
    summary.to_csv(summary_path, index=False)
    profiles.to_csv(profiles_path, index=False)
    plot_equity_heatmap(profiles, heatmap_path)

    metadata = {
        "cluster_assignments": str(assignments_path),
        "cities": cities,
        "target_column": str(args.target_column),
        "n_rows": int(len(merged)),
        "n_clusters": int(summary["cluster_id"].nunique()),
        "n_acs_variables": int(len([col for col in args.acs_columns if col in merged.columns])),
        "acs_columns": [str(col) for col in args.acs_columns if col in merged.columns],
        "artifacts": {
            "cluster_acs_rows": str(merged_path),
            "cluster_summary": str(summary_path),
            "acs_profiles": str(profiles_path),
            "acs_heatmap": str(heatmap_path),
            "report": str(report_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(report_path, summary, profiles, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

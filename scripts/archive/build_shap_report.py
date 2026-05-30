"""Create report-ready tables and figures from multi-city SHAP outputs."""

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build thesis-ready SHAP tables and figures."
    )
    parser.add_argument(
        "--summary-json",
        default="outputs/metrics/multicity_xgboost_shap_summary.json",
        help="SHAP summary JSON from explain_multicity_xgboost_shap.py",
    )
    parser.add_argument(
        "--importance-csv",
        default="outputs/tables/multicity_xgboost_shap_feature_importance.csv",
        help="Global SHAP feature importance CSV",
    )
    parser.add_argument(
        "--city-importance-csv",
        default="outputs/tables/multicity_xgboost_shap_city_feature_importance.csv",
        help="City-level SHAP feature importance CSV",
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument(
        "--prefix",
        default="multicity_xgboost_shap",
        help="Artifact filename prefix",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--heatmap-top-n", type=int, default=12)
    return parser.parse_args()


def _import_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for report figures. Install requirements first, "
            "for example: .venv/bin/pip install -r requirements.txt"
        ) from exc
    return plt


def feature_parts(feature: str) -> dict[str, str | int]:
    if feature.startswith("pt_"):
        family = "public transport"
        clean = feature.removeprefix("pt_")
    elif feature.startswith("be_"):
        family = "built environment"
        clean = feature.removeprefix("be_")
    elif feature.startswith("acs_"):
        family = "sociodemographic"
        clean = feature.removeprefix("acs_")
    else:
        family = "other"
        clean = feature

    label = clean.replace("_", " ")

    return {
        "base_feature": feature,
        "feature_label": label,
        "feature_family": family,
    }


def add_feature_metadata(df: pd.DataFrame) -> pd.DataFrame:
    parts = pd.DataFrame([feature_parts(str(feature)) for feature in df["feature"]])
    return pd.concat([df.reset_index(drop=True), parts], axis=1)


def direction_label(corr: float) -> str:
    if corr >= 0.2:
        return "higher feature values generally raise predictions"
    if corr <= -0.2:
        return "higher feature values generally lower predictions"
    return "mixed or weak directional pattern"


def build_report_tables(
    importance: pd.DataFrame,
    city_importance: pd.DataFrame,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    report = add_feature_metadata(importance).copy()
    total_importance = float(report["mean_abs_shap"].sum())
    report["relative_importance_pct"] = (
        100 * report["mean_abs_shap"] / total_importance
        if total_importance > 0
        else 0.0
    )
    report["direction"] = report["feature_shap_corr"].astype(float).map(direction_label)

    top_features = report.head(top_n).copy()

    family_context = (
        report.groupby(["feature_family"], as_index=False)
        .agg(
            total_mean_abs_shap=("mean_abs_shap", "sum"),
            n_features=("feature", "size"),
        )
        .sort_values("total_mean_abs_shap", ascending=False)
    )
    total_grouped = float(family_context["total_mean_abs_shap"].sum())
    family_context["relative_importance_pct"] = (
        100 * family_context["total_mean_abs_shap"] / total_grouped
        if total_grouped > 0
        else 0.0
    )

    city_report = add_feature_metadata(city_importance).copy()
    city_report["city_rank"] = (
        city_report.groupby("city")["mean_abs_shap"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    city_top = city_report[city_report["city_rank"] <= min(10, top_n)].sort_values(
        ["city", "city_rank"]
    )
    return top_features, family_context, city_top


def save_top_feature_plot(
    top_features: pd.DataFrame,
    path: Path,
    plt: Any,
) -> None:
    plot_df = top_features.sort_values("mean_abs_shap", ascending=True).copy()
    colors = plot_df["feature_family"].map(
        {
            "public transport": "#277da1",
            "built environment": "#f3722c",
            "sociodemographic": "#43aa8b",
            "other": "#6c757d",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["feature_label"], plot_df["mean_abs_shap"], color=colors)
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Top SHAP Drivers of Predicted Accessibility")
    ax.grid(axis="x", alpha=0.25)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_family_context_plot(
    family_context: pd.DataFrame,
    path: Path,
    plt: Any,
) -> None:
    plot_df = family_context.sort_values("relative_importance_pct", ascending=True).copy()
    colors = plot_df["feature_family"].map(
        {
            "public transport": "#277da1",
            "built environment": "#f3722c",
            "sociodemographic": "#43aa8b",
            "other": "#6c757d",
        }
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(plot_df["feature_family"], plot_df["relative_importance_pct"], color=colors)
    ax.set_xlabel("Share of total mean absolute SHAP value (%)")
    ax.set_title("SHAP Importance by Feature Family")
    ax.grid(axis="x", alpha=0.25)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_city_heatmap(
    city_importance: pd.DataFrame,
    global_top_features: pd.DataFrame,
    path: Path,
    plt: Any,
    heatmap_top_n: int,
) -> pd.DataFrame:
    top = global_top_features.head(heatmap_top_n)[["feature", "feature_label"]].copy()
    work = city_importance[city_importance["feature"].isin(set(top["feature"]))].copy()
    pivot = work.pivot_table(
        index="feature",
        columns="city",
        values="mean_abs_shap",
        aggfunc="mean",
        fill_value=0.0,
    )
    pivot = pivot.reindex(top["feature"])
    labels = top.set_index("feature").loc[pivot.index, "feature_label"].tolist()

    fig_width = max(9, 0.75 * len(pivot.columns))
    fig_height = max(5.5, 0.42 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(city).replace("_", " ").title() for city in pivot.columns], rotation=45)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Top SHAP Drivers by City")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean absolute SHAP value")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)

    out = pivot.reset_index()
    out.insert(1, "feature_label", labels)
    return out


def markdown_table(df: pd.DataFrame, columns: list[str], n: int) -> str:
    rows = df[columns].head(n).copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in rows.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def write_markdown_summary(
    summary: dict[str, Any],
    top_features: pd.DataFrame,
    family_context: pd.DataFrame,
    artifacts: dict[str, str],
    path: Path,
) -> None:
    metrics = summary["explained_sample_metrics"]["test_explained_sample"]
    lines = [
        f"# Multi-City {summary['model'].replace('_', ' ').title()} SHAP Summary",
        "",
        "## Model",
        "",
        f"- Model: `{summary['model']}`",
        f"- Cities: `{len(summary['cities'])}`",
        f"- Rows available: `{summary['n_rows']}`",
        f"- Explained held-out rows: `{summary['n_explained_rows']}`",
        f"- Features: `{summary['n_features']}`",
        f"- Spatial splits: `{summary.get('n_spatial_splits', 1)}`",
        "",
        "## Explained-Sample Performance",
        "",
        f"- Mean R2: `{metrics['r2']['mean']:.3f}`",
        f"- Mean RMSE: `{metrics['rmse']['mean']:.3f}`",
        f"- Mean MAE: `{metrics['mae']['mean']:.3f}`",
        "",
        "## Top SHAP Features",
        "",
        markdown_table(
            top_features,
            [
                "feature_label",
                "feature_family",
                "mean_abs_shap",
                "feature_shap_corr",
                "direction",
            ],
            12,
        ),
        "",
        "## Feature Family Summary",
        "",
        markdown_table(
            family_context,
            [
                "feature_family",
                "relative_importance_pct",
                "n_features",
            ],
            12,
        ),
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(f"- `{name}`: `{artifact}`" for name, artifact in artifacts.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    plt = _import_matplotlib()

    summary_path = Path(args.summary_json)
    importance_path = Path(args.importance_csv)
    city_importance_path = Path(args.city_importance_csv)
    for path in [summary_path, importance_path, city_importance_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required SHAP input not found: {path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    importance = pd.read_csv(importance_path)
    city_importance = pd.read_csv(city_importance_path)

    top_features, family_context, city_top = build_report_tables(
        importance=importance,
        city_importance=city_importance,
        top_n=args.top_n,
    )

    output_root = Path(args.outputs_root)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    reports_dir = output_root / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    top_features_path = tables_dir / f"{args.prefix}_top_features_report.csv"
    family_context_path = tables_dir / f"{args.prefix}_family_context_report.csv"
    city_top_path = tables_dir / f"{args.prefix}_city_top_features_report.csv"
    heatmap_table_path = tables_dir / f"{args.prefix}_city_heatmap_report.csv"
    top_plot_path = figures_dir / f"{args.prefix}_top_features.png"
    family_plot_path = figures_dir / f"{args.prefix}_family_context.png"
    heatmap_path = figures_dir / f"{args.prefix}_city_heatmap.png"
    markdown_path = reports_dir / f"{args.prefix}_report.md"

    top_features.to_csv(top_features_path, index=False)
    family_context.to_csv(family_context_path, index=False)
    city_top.to_csv(city_top_path, index=False)
    save_top_feature_plot(top_features, top_plot_path, plt)
    save_family_context_plot(family_context, family_plot_path, plt)
    heatmap_table = save_city_heatmap(
        city_importance=city_importance,
        global_top_features=top_features,
        path=heatmap_path,
        plt=plt,
        heatmap_top_n=args.heatmap_top_n,
    )
    heatmap_table.to_csv(heatmap_table_path, index=False)

    artifacts = {
        "top_features_table": str(top_features_path),
        "family_context_table": str(family_context_path),
        "city_top_features_table": str(city_top_path),
        "city_heatmap_table": str(heatmap_table_path),
        "top_features_figure": str(top_plot_path),
        "family_context_figure": str(family_plot_path),
        "city_heatmap_figure": str(heatmap_path),
    }
    write_markdown_summary(
        summary=summary,
        top_features=top_features,
        family_context=family_context,
        artifacts=artifacts,
        path=markdown_path,
    )
    artifacts["markdown_report"] = str(markdown_path)

    print(json.dumps({"artifacts": artifacts}, indent=2))


if __name__ == "__main__":
    main()

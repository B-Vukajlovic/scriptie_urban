"""Plot block-only PT+BE sensitivity figures for thesis reporting."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIGS = [
    {
        "label": "53 original",
        "xgb_metrics": "multicity_xgboost_block_only_pt_be_metrics.csv",
        "gnn_metrics": "multicity_gnn_block_only_pt_be_spatial_cv_metrics.csv",
    },
    {
        "label": "56 expanded",
        "xgb_metrics": "multicity_xgboost_expanded_block_only_pt_be_metrics.csv",
        "gnn_metrics": "multicity_gnn_expanded_block_only_pt_be_spatial_cv_metrics.csv",
    },
    {
        "label": "51 no area/footprint",
        "xgb_metrics": "multicity_xgboost_block_only_pt_be_no_area_footprint_metrics.csv",
        "gnn_metrics": "multicity_gnn_block_only_pt_be_no_area_footprint_spatial_cv_metrics.csv",
    },
]

MODEL_COLORS = {
    "XGBoost": "#c75d2c",
    "GCN": "#2a7f62",
    "GraphSAGE": "#255f99",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create figures for the block-only PT+BE feature sensitivity."
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument(
        "--prefix",
        default="block_only_pt_be_no_area_footprint",
        help="Filename prefix for generated figures.",
    )
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args()


def _short_feature(feature: str) -> str:
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


def _city_label(city: str) -> str:
    return city.replace("_", " ").title()


def load_pooled_metrics(tables_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, str | float]] = []
    for config in CONFIGS:
        xgb = pd.read_csv(tables_dir / str(config["xgb_metrics"]))
        xgb_row = xgb[
            (xgb["experiment"] == "pooled_spatial_cv")
            & (xgb["split"] == "test")
            & (xgb["city"] == "__all__")
        ].iloc[0]
        rows.append(
            {
                "feature_set": str(config["label"]),
                "model": "XGBoost",
                "r2": float(xgb_row["r2"]),
                "rmse": float(xgb_row["rmse"]),
                "mae": float(xgb_row["mae"]),
            }
        )

        gnn = pd.read_csv(tables_dir / str(config["gnn_metrics"]))
        for model_key, model_label in [("gcn", "GCN"), ("graphsage", "GraphSAGE")]:
            gnn_row = gnn[
                (gnn["model"] == model_key)
                & (gnn["split"] == "test")
                & (gnn["city"] == "__all__")
            ].iloc[0]
            rows.append(
                {
                    "feature_set": str(config["label"]),
                    "model": model_label,
                    "r2": float(gnn_row["r2"]),
                    "rmse": float(gnn_row["rmse"]),
                    "mae": float(gnn_row["mae"]),
                }
            )
    return pd.DataFrame(rows)


def plot_pooled_metric_bars(metrics: pd.DataFrame, figures_dir: Path, prefix: str) -> None:
    for metric, ylabel in [("r2", "Pooled spatial CV R2"), ("rmse", "Pooled spatial CV RMSE")]:
        fig, ax = plt.subplots(figsize=(10.5, 6))
        feature_sets = metrics["feature_set"].drop_duplicates().tolist()
        models = ["XGBoost", "GCN", "GraphSAGE"]
        x = np.arange(len(feature_sets), dtype=float)
        width = 0.23
        for i, model in enumerate(models):
            values = (
                metrics[metrics["model"] == model]
                .set_index("feature_set")
                .loc[feature_sets, metric]
                .to_numpy(dtype=float)
            )
            offset = (i - 1) * width
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                color=MODEL_COLORS[model],
                label=model,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(feature_sets)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"Model performance under block-only PT+BE feature sensitivity ({metric.upper()})",
            pad=16,
        )
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncols=3, loc="upper right")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{prefix}_{metric}_comparison.png", dpi=220)
        plt.close(fig)


def load_no_area_city_metrics(tables_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    xgb = pd.read_csv(tables_dir / "multicity_xgboost_block_only_pt_be_no_area_footprint_metrics.csv")
    xgb_city = xgb[
        (xgb["experiment"] == "pooled_spatial_cv")
        & (xgb["split"] == "test")
        & (xgb["city"] != "__all__")
    ][["city", "r2", "rmse", "mae"]].copy()
    xgb_city.insert(0, "model", "XGBoost")
    rows.append(xgb_city)

    pyg = pd.read_csv(tables_dir / "multicity_pyg_gnn_block_only_pt_be_no_area_footprint_metrics.csv")
    pyg_city = pyg[(pyg["split"] == "test") & (pyg["city"] != "__all__")].copy()
    pyg_city["model"] = pyg_city["model"].map({"gcn": "GCN", "graphsage": "GraphSAGE"})
    rows.append(pyg_city[["model", "city", "r2", "rmse", "mae"]])
    return pd.concat(rows, ignore_index=True)


def plot_city_r2(city_metrics: pd.DataFrame, figures_dir: Path, prefix: str) -> None:
    order = (
        city_metrics[city_metrics["model"] == "GraphSAGE"]
        .sort_values("r2", ascending=True)["city"]
        .tolist()
    )
    models = ["XGBoost", "GCN", "GraphSAGE"]
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(order), dtype=float)
    height = 0.24
    for i, model in enumerate(models):
        values = (
            city_metrics[city_metrics["model"] == model]
            .set_index("city")
            .loc[order, "r2"]
            .to_numpy(dtype=float)
        )
        ax.barh(
            y + (i - 1) * height,
            values,
            height=height,
            color=MODEL_COLORS[model],
            label=model,
        )
    ax.axvline(0.0, color="#202020", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([_city_label(city) for city in order])
    ax.set_xlabel("City-level test R2")
    ax.set_title("No-area/no-footprint sensitivity: city-level spatial CV performance")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, ncols=3, loc="lower right")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}_city_r2.png", dpi=220)
    plt.close(fig)


def plot_xgb_importance(tables_dir: Path, figures_dir: Path, prefix: str, top_n: int) -> None:
    importance = pd.read_csv(
        tables_dir / "multicity_xgboost_block_only_pt_be_no_area_footprint_feature_importance.csv"
    )
    top = (
        importance[importance["experiment"] == "pooled_spatial_cv"]
        .sort_values("importance_gain", ascending=False)
        .head(top_n)
        .iloc[::-1]
        .copy()
    )
    colors = top["feature"].map(lambda feature: "#277da1" if _family(str(feature)) == "PT" else "#f3722c")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"].map(lambda value: _short_feature(str(value))), top["importance_gain"], color=colors)
    ax.set_xlabel("XGBoost gain importance")
    ax.set_title("XGBoost drivers after removing area and footprint")
    ax.grid(axis="x", alpha=0.25)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}_xgboost_top_features.png", dpi=220)
    plt.close(fig)

    family = (
        importance[importance["experiment"] == "pooled_spatial_cv"]
        .assign(family=lambda df: df["feature"].map(lambda value: _family(str(value))))
        .groupby("family", as_index=False)["importance_gain"]
        .sum()
    )
    total = float(family["importance_gain"].sum())
    family["share"] = np.where(total > 0, family["importance_gain"] / total, 0.0)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    family = family.sort_values("share")
    colors = family["family"].map({"PT": "#277da1", "BE": "#f3722c", "ACS": "#43aa8b", "other": "#6c757d"})
    ax.barh(family["family"], family["share"] * 100, color=colors)
    ax.set_xlabel("Share of total XGBoost gain (%)")
    ax.set_title("XGBoost importance balance after sensitivity removal")
    ax.grid(axis="x", alpha=0.25)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}_xgboost_family_balance.png", dpi=220)
    plt.close(fig)


def load_prediction_panels(tables_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    xgb = pd.read_csv(
        tables_dir / "multicity_xgboost_block_only_pt_be_no_area_footprint_predictions.csv",
        usecols=["experiment", "city", "split", "y_true", "y_pred"],
    )
    xgb = xgb[(xgb["experiment"] == "pooled_spatial_cv") & (xgb["split"] == "test")].copy()
    xgb["model"] = "XGBoost"
    frames.append(xgb[["model", "city", "y_true", "y_pred"]])

    pyg = pd.read_csv(
        tables_dir / "multicity_pyg_gnn_block_only_pt_be_no_area_footprint_predictions.csv",
        usecols=["model", "city", "split", "y_true", "y_pred"],
    )
    pyg = pyg[pyg["split"] == "test"].copy()
    pyg["model"] = pyg["model"].map({"gcn": "GCN", "graphsage": "GraphSAGE"})
    frames.append(pyg[["model", "city", "y_true", "y_pred"]])
    return pd.concat(frames, ignore_index=True)


def plot_prediction_scatter(predictions: pd.DataFrame, figures_dir: Path, prefix: str) -> None:
    rng = np.random.default_rng(42)
    models = ["XGBoost", "GCN", "GraphSAGE"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True, sharey=True)
    lim_min = float(min(predictions["y_true"].min(), predictions["y_pred"].min()))
    lim_max = float(max(predictions["y_true"].max(), predictions["y_pred"].max()))
    for ax, model in zip(axes, models):
        group = predictions[predictions["model"] == model]
        if len(group) > 12000:
            group = group.iloc[rng.choice(len(group), size=12000, replace=False)]
        ax.scatter(group["y_true"], group["y_pred"], s=4, alpha=0.13, color=MODEL_COLORS[model], linewidths=0)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#1f1f1f", linewidth=1)
        ax.set_title(model)
        ax.grid(alpha=0.2)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Predicted accessibility")
    for ax in axes:
        ax.set_xlabel("Observed accessibility")
    fig.suptitle("No-area/no-footprint sensitivity: held-out predictions", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{prefix}_predicted_vs_observed.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_table(metrics: pd.DataFrame, city_metrics: pd.DataFrame, tables_dir: Path, prefix: str) -> None:
    metrics.to_csv(tables_dir / f"{prefix}_figure_summary_metrics.csv", index=False)
    city_metrics.to_csv(tables_dir / f"{prefix}_figure_city_metrics.csv", index=False)


def update_readme(figures_dir: Path, prefix: str) -> None:
    readme = figures_dir / "README.md"
    lines = [
        "# Figure Index",
        "",
        "The top-level folder currently contains the no-area/no-footprint block-only",
        "PT+BE sensitivity figures. Previous thesis figures and older experiments are",
        "preserved under `archive/`.",
        "",
        "## Current Figures",
        "",
        f"- `{prefix}_r2_comparison.png`: pooled spatial-CV R2 across the 53-, 56-, and 51-feature setups.",
        f"- `{prefix}_rmse_comparison.png`: pooled spatial-CV RMSE across the same setups.",
        f"- `{prefix}_city_r2.png`: city-level R2 for XGBoost, GCN, and GraphSAGE under the 51-feature sensitivity.",
        f"- `{prefix}_xgboost_top_features.png`: top XGBoost gain features after removing block area and footprint share.",
        f"- `{prefix}_xgboost_family_balance.png`: PT-vs-BE XGBoost gain balance under the sensitivity.",
        f"- `{prefix}_predicted_vs_observed.png`: held-out predicted-vs-observed scatter for XGBoost, GCN, and GraphSAGE.",
        "",
        "## Archive",
        "",
        "- `archive/final_expanded_block_only_pt_be_56/`: complete expanded block-only PT+BE thesis figure set.",
        "- `archive/legacy_reduced_xgboost/`: older reduced XGBoost/SHAP figures.",
        "- `archive/legacy_tuned_xgboost/`: older tuned reduced XGBoost/SHAP figures.",
        "- `archive/legacy_full_xgboost/`: older full-feature XGBoost/SHAP figures.",
        "- `archive/legacy_gnn/`: older custom/PyG GNN explanation figures.",
        "- `archive/smoke/`: smoke-test figures.",
        "",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.outputs_root)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    pooled_metrics = load_pooled_metrics(tables_dir)
    city_metrics = load_no_area_city_metrics(tables_dir)
    plot_pooled_metric_bars(pooled_metrics, figures_dir, str(args.prefix))
    plot_city_r2(city_metrics, figures_dir, str(args.prefix))
    plot_xgb_importance(tables_dir, figures_dir, str(args.prefix), int(args.top_n))
    plot_prediction_scatter(load_prediction_panels(tables_dir), figures_dir, str(args.prefix))
    write_summary_table(pooled_metrics, city_metrics, tables_dir, str(args.prefix))
    update_readme(figures_dir, str(args.prefix))

    print(
        {
            "figures_dir": str(figures_dir),
            "n_figures": 6,
            "prefix": str(args.prefix),
        }
    )


if __name__ == "__main__":
    main()

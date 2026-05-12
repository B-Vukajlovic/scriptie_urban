"""Build partial-dependence plots for the reduced multi-city XGBoost model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_multicity_xgboost import (  # noqa: E402
    FEATURE_VIEW_CHOICES,
    TARGET_VIEW_CHOICES,
    apply_target_view,
    build_feature_view,
    build_single_city_splits,
    load_multicity_inputs,
    model_params,
    select_feature_set,
    string_key_records,
)
from src.utils.cities import DEFAULT_CITIES, validate_cities  # noqa: E402


DEFAULT_PDP_FEATURES = [
    "pt_metro_peak_departures_within_2000m",
    "acs_population_density_per_km2",
    "pt_routes_within_3000m",
    "acs_car_commute_share",
    "acs_bachelor_or_higher_share",
    "be_building_footprint_share",
    "be_intersection_density_per_km2",
    "acs_avg_commute_time_min",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the reduced pooled-spatial XGBoost baseline and create one-way "
            "partial-dependence plots for report-ready interpretation."
        )
    )
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    parser.add_argument("--interim-root", default="data/interim")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--target-column", default="Y_global_log_minmax")
    parser.add_argument("--target-view", choices=TARGET_VIEW_CHOICES, default="stored")
    parser.add_argument(
        "--feature-view",
        choices=[choice for choice in FEATURE_VIEW_CHOICES if choice != "all"],
        default="log1p",
    )
    parser.add_argument(
        "--reduced-feature-set",
        default="data/interim/modeling/reduced_feature_set.json",
    )
    parser.add_argument("--split-seed", type=int, default=1000)
    parser.add_argument("--split-val-frac", type=float, default=0.15)
    parser.add_argument("--split-test-frac", type=float, default=0.15)
    parser.add_argument("--split-grid-bins-x", type=int, default=8)
    parser.add_argument("--split-grid-bins-y", type=int, default=8)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Features to plot. Defaults to selected high-importance reduced features.",
    )
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--sample-rows", type=int, default=20000)
    parser.add_argument("--random-sample-state", type=int, default=42)
    parser.add_argument(
        "--output-prefix",
        default="multicity_xgboost_reduced_pdp",
    )
    return parser.parse_args()


def _short_feature_label(feature: str) -> str:
    return (
        feature.removeprefix("pt_")
        .removeprefix("be_")
        .removeprefix("acs_")
        .replace("_", " ")
    )


def _safe_filename(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_")


def _inverse_feature_value(feature: str, value: float, log1p_columns: set[str]) -> float:
    if feature in log1p_columns:
        return float(np.expm1(value))
    return float(value)


def _axis_label(feature: str, log1p_columns: set[str]) -> str:
    label = _short_feature_label(feature)
    if feature in log1p_columns:
        return f"{label} (raw scale; model uses log1p)"
    return label


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


def resolve_pdp_features(
    requested: list[str] | None,
    feature_columns: list[str],
    shap_importance_path: Path,
) -> list[str]:
    if requested:
        features = requested
    elif shap_importance_path.exists():
        shap_importance = pd.read_csv(shap_importance_path)
        features = shap_importance["feature"].astype(str).head(8).tolist()
    else:
        features = DEFAULT_PDP_FEATURES

    selected: list[str] = []
    for feature in features:
        if feature not in feature_columns:
            continue
        if feature not in selected:
            selected.append(feature)
    if not selected:
        raise ValueError("No requested PDP features are available in the reduced feature set.")
    return selected


def sample_reference_rows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    sample_rows: int,
    random_state: int,
) -> pd.DataFrame:
    reference = frame[feature_columns].copy()
    if sample_rows > 0 and len(reference) > sample_rows:
        reference = reference.sample(n=sample_rows, random_state=random_state)
    return reference.reset_index(drop=True)


def partial_dependence_for_feature(
    model: XGBRegressor,
    reference: pd.DataFrame,
    feature: str,
    grid_size: int,
    log1p_columns: set[str],
) -> pd.DataFrame:
    values = reference[feature].astype(float)
    lower, upper = np.quantile(values, [0.05, 0.95])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        lower, upper = float(values.min()), float(values.max())
    if lower == upper:
        grid = np.array([lower], dtype=float)
    else:
        grid = np.linspace(lower, upper, grid_size)

    rows: list[dict[str, float | str]] = []
    for transformed_value in grid:
        modified = reference.copy()
        modified[feature] = float(transformed_value)
        pred = model.predict(modified).astype(float)
        rows.append(
            {
                "feature": feature,
                "feature_value_model_scale": float(transformed_value),
                "feature_value_raw_scale": _inverse_feature_value(
                    feature,
                    float(transformed_value),
                    log1p_columns,
                ),
                "mean_prediction": float(np.mean(pred)),
                "prediction_p10": float(np.quantile(pred, 0.10)),
                "prediction_p90": float(np.quantile(pred, 0.90)),
            }
        )
    return pd.DataFrame(rows)


def plot_single_pdp(pdp: pd.DataFrame, feature: str, log1p_columns: set[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        pdp["feature_value_raw_scale"],
        pdp["mean_prediction"],
        color="#1f5a7a",
        linewidth=2.4,
    )
    ax.fill_between(
        pdp["feature_value_raw_scale"].to_numpy(dtype=float),
        pdp["prediction_p10"].to_numpy(dtype=float),
        pdp["prediction_p90"].to_numpy(dtype=float),
        color="#6baed6",
        alpha=0.22,
        linewidth=0,
    )
    ax.set_xlabel(_axis_label(feature, log1p_columns))
    ax.set_ylabel("Predicted accessibility")
    ax.set_title(f"Partial dependence: {_short_feature_label(feature)}")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pdp_grid(pdp_values: pd.DataFrame, features: list[str], log1p_columns: set[str], output_path: Path) -> None:
    n_cols = 2
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.1 * n_rows))
    flat_axes = np.asarray(axes).reshape(-1)
    for ax, feature in zip(flat_axes, features):
        pdp = pdp_values[pdp_values["feature"] == feature]
        ax.plot(
            pdp["feature_value_raw_scale"],
            pdp["mean_prediction"],
            color="#1f5a7a",
            linewidth=2.2,
        )
        ax.fill_between(
            pdp["feature_value_raw_scale"].to_numpy(dtype=float),
            pdp["prediction_p10"].to_numpy(dtype=float),
            pdp["prediction_p90"].to_numpy(dtype=float),
            color="#6baed6",
            alpha=0.20,
            linewidth=0,
        )
        ax.set_title(_short_feature_label(feature))
        ax.set_xlabel(_axis_label(feature, log1p_columns))
        ax.set_ylabel("Predicted accessibility")
        ax.grid(True, alpha=0.22)
    for ax in flat_axes[len(features):]:
        ax.axis("off")
    fig.suptitle("Reduced XGBoost partial-dependence plots", y=1.0, fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(report_path: Path, pdp_values: pd.DataFrame, metadata: dict[str, Any]) -> None:
    summary_rows: list[dict[str, float | str]] = []
    for feature, group in pdp_values.groupby("feature", sort=False):
        first = float(group["mean_prediction"].iloc[0])
        last = float(group["mean_prediction"].iloc[-1])
        summary_rows.append(
            {
                "feature": str(feature),
                "min_raw_value": float(group["feature_value_raw_scale"].min()),
                "max_raw_value": float(group["feature_value_raw_scale"].max()),
                "prediction_change": last - first,
                "min_mean_prediction": float(group["mean_prediction"].min()),
                "max_mean_prediction": float(group["mean_prediction"].max()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    lines = [
        "# Reduced XGBoost Partial-Dependence Plots",
        "",
        "These plots show the model-average association between selected predictors and the predicted global log-minmax accessibility score.",
        "For log-transformed predictors, the x-axis is reported back on the raw feature scale while the model internally uses log1p values.",
        "",
        "## Run Metadata",
        "",
        f"- Target: {metadata['target_column']}",
        f"- Target view: {metadata['target_view']}",
        f"- Feature view: {metadata['feature_view']}",
        f"- Reduced features in model: {metadata['n_features']}",
        f"- Reference rows: {metadata['n_reference_rows']}",
        "",
        "## PDP Summary",
        "",
        dataframe_to_markdown(summary),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cities = validate_cities([str(city) for city in args.cities])
    output_root = Path(args.outputs_root)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    reports_dir = output_root / "reports"
    metrics_dir = output_root / "metrics"
    for path in [tables_dir, figures_dir, reports_dir, metrics_dir]:
        path.mkdir(parents=True, exist_ok=True)

    dataset, base_feature_columns, coords, _metadata_by_city = load_multicity_inputs(
        cities=cities,
        interim_root=args.interim_root,
        target_column=args.target_column,
    )
    dataset, target_view_metadata = apply_target_view(
        dataset=dataset,
        cities=cities,
        interim_root=args.interim_root,
        target_column=args.target_column,
        target_view=str(args.target_view),
    )
    selected_features, feature_set_metadata = select_feature_set(
        base_feature_columns,
        "reduced",
        args.reduced_feature_set,
    )
    dataset, feature_columns, feature_view_metadata = build_feature_view(
        dataset,
        selected_features,
        str(args.feature_view),
    )
    splits = build_single_city_splits(coords, args)
    split_df = dataset.merge(
        splits[["node_id", "split", "spatial_cell"]],
        on="node_id",
        how="inner",
    )
    if len(split_df) != len(dataset):
        raise ValueError("Spatial split does not cover all rows.")

    train = split_df[split_df["split"] == "train"].copy()
    test = split_df[split_df["split"] == "test"].copy()
    model = XGBRegressor(**model_params(args, 0))
    model.fit(train[feature_columns], train[args.target_column], verbose=False)

    shap_importance_path = tables_dir / "multicity_xgboost_shap_reduced_feature_importance.csv"
    pdp_features = resolve_pdp_features(args.features, feature_columns, shap_importance_path)
    reference = sample_reference_rows(
        test,
        feature_columns,
        sample_rows=int(args.sample_rows),
        random_state=int(args.random_sample_state),
    )
    log1p_columns = set(str(col) for col in feature_view_metadata.get("log1p_columns", []))

    pdp_frames: list[pd.DataFrame] = []
    figure_paths: dict[str, str] = {}
    for feature in pdp_features:
        pdp = partial_dependence_for_feature(
            model=model,
            reference=reference,
            feature=feature,
            grid_size=int(args.grid_size),
            log1p_columns=log1p_columns,
        )
        pdp_frames.append(pdp)
        figure_path = figures_dir / f"{args.output_prefix}_{_safe_filename(feature)}.png"
        plot_single_pdp(pdp, feature, log1p_columns, figure_path)
        figure_paths[feature] = str(figure_path)

    pdp_values = pd.concat(pdp_frames, ignore_index=True)
    pdp_values_path = tables_dir / f"{args.output_prefix}_values.csv"
    pdp_values.to_csv(pdp_values_path, index=False)
    grid_path = figures_dir / f"{args.output_prefix}_grid.png"
    plot_pdp_grid(pdp_values, pdp_features, log1p_columns, grid_path)

    metadata: dict[str, Any] = {
        "cities": cities,
        "target_column": args.target_column,
        "target_view": str(args.target_view),
        "target_view_metadata": target_view_metadata,
        "feature_view": str(args.feature_view),
        "feature_view_metadata": feature_view_metadata,
        "feature_set": "reduced",
        "feature_set_metadata": feature_set_metadata,
        "n_features": int(len(feature_columns)),
        "pdp_features": pdp_features,
        "n_reference_rows": int(len(reference)),
        "reference_split": "test",
        "grid_size": int(args.grid_size),
        "hyperparameters": {
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "max_depth": int(args.max_depth),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "min_child_weight": float(args.min_child_weight),
            "reg_alpha": float(args.reg_alpha),
            "reg_lambda": float(args.reg_lambda),
            "random_state": int(args.random_state),
            "n_jobs": int(args.n_jobs),
        },
        "artifacts": {
            "pdp_values": str(pdp_values_path),
            "grid_figure": str(grid_path),
            "feature_figures": figure_paths,
            "report": str(reports_dir / f"{args.output_prefix}_report.md"),
        },
        "pdp_summary": string_key_records(
            pdp_values.groupby("feature", sort=False)
            .agg(
                min_raw_value=("feature_value_raw_scale", "min"),
                max_raw_value=("feature_value_raw_scale", "max"),
                min_mean_prediction=("mean_prediction", "min"),
                max_mean_prediction=("mean_prediction", "max"),
            )
            .reset_index()
        ),
    }
    metadata_path = metrics_dir / f"{args.output_prefix}_summary.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(
        report_path=reports_dir / f"{args.output_prefix}_report.md",
        pdp_values=pdp_values,
        metadata=metadata,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

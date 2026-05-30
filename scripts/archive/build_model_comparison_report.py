"""Build a compact comparison report for XGBoost, GCN, and GraphSAGE outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create thesis-facing model comparison tables from saved outputs."
    )
    parser.add_argument(
        "--xgboost-summary",
        default="outputs/metrics/multicity_xgboost_summary.json",
        help="Summary JSON from evaluate_multicity_xgboost.py.",
    )
    parser.add_argument(
        "--gnn-summary",
        default="outputs/metrics/multicity_gnn_summary.json",
        help="Summary JSON from evaluate_multicity_gnn.py.",
    )
    parser.add_argument(
        "--shap-summary",
        default="outputs/metrics/multicity_xgboost_shap_summary.json",
        help="Optional SHAP summary JSON for the plain XGBoost model.",
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-prefix", default="model_comparison")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Required summary JSON not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_load_json(path: str | Path) -> dict[str, Any] | None:
    json_path = Path(path)
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(summary: dict[str, Any], split: str, metric: str, stat: str) -> float:
    return float(summary[split][metric][stat])


def add_xgboost_rows(rows: list[dict[str, Any]], xgb: dict[str, Any]) -> None:
    pooled = xgb["pooled_spatial_cv_summary_by_split"]
    loco = xgb["leave_one_city_out_summary"]
    rows.append(
        {
            "model": "XGBoost",
            "experiment": "pooled single spatial CV",
            "evaluation_unit": "one spatial test fold",
            "n_rows": int(xgb["n_rows"]),
            "n_features": int(xgb["n_features"]),
            "rmse": metric_value(pooled, "test", "rmse", "mean"),
            "mae": metric_value(pooled, "test", "mae", "mean"),
            "r2": metric_value(pooled, "test", "r2", "mean"),
            "rmse_std": metric_value(pooled, "test", "rmse", "std"),
            "r2_std": metric_value(pooled, "test", "r2", "std"),
            "protocol_note": (
                "Fast thesis baseline: one spatial train/val/test split within every city."
            ),
        }
    )
    rows.append(
        {
            "model": "XGBoost",
            "experiment": "leave-one-city-out",
            "evaluation_unit": "mean over held-out cities",
            "n_rows": int(xgb["n_rows"]),
            "n_features": int(xgb["n_features"]),
            "rmse": metric_value(loco, "test", "rmse", "mean"),
            "mae": metric_value(loco, "test", "mae", "mean"),
            "r2": metric_value(loco, "test", "r2", "mean"),
            "rmse_std": metric_value(loco, "test", "rmse", "std"),
            "r2_std": metric_value(loco, "test", "r2", "std"),
            "protocol_note": (
                "Hard generalization test: train on nine cities and test on the "
                "unseen city."
            ),
        }
    )


def add_gnn_rows(rows: list[dict[str, Any]], gnn: dict[str, Any]) -> None:
    if "summary_by_model" in gnn:
        mode = str(gnn.get("evaluation_mode", "gnn_evaluation"))
        experiment_label = mode.replace("_", " ")
        if mode == "single_spatial_cv":
            unit = "one spatial test fold"
        elif mode == "leave_one_city_out":
            unit = "mean over held-out cities"
        else:
            unit = "one saved train/val/test split"
        for model_name, split_summary in gnn["summary_by_model"].items():
            test = split_summary["test"]
            rows.append(
                {
                    "model": "GCN" if model_name == "gcn" else "GraphSAGE",
                    "experiment": experiment_label,
                    "evaluation_unit": unit,
                    "n_rows": int(gnn["n_rows"]),
                    "n_features": int(gnn["n_features"]),
                    "rmse": float(test["rmse"]["mean"]),
                    "mae": float(test["mae"]["mean"]),
                    "r2": float(test["r2"]["mean"]),
                    "rmse_std": float(test["rmse"]["std"]),
                    "r2_std": float(test["r2"]["std"]),
                "protocol_note": (
                    "True message-passing GNN over block adjacency evaluated "
                    f"with {gnn.get('evaluation_mode', 'the saved split')}."
                    ),
                }
            )
        return

    for model_name, result in gnn["results"].items():
        test = result["metrics_by_split"]["test"]
        rows.append(
            {
                "model": "GCN" if model_name == "gcn" else "GraphSAGE",
                "experiment": "single existing spatial split",
                "evaluation_unit": "one saved train/val/test split",
                "n_rows": int(gnn["n_rows"]),
                "n_features": int(gnn["n_features"]),
                "rmse": float(test["rmse"]),
                "mae": float(test["mae"]),
                "r2": float(test["r2"]),
                "rmse_std": None,
                "r2_std": None,
                "protocol_note": (
                    "True message-passing GNN over block adjacency using one saved split."
                ),
            }
        )


def feature_family(feature: str) -> str:
    if feature.startswith("pt_"):
        return "public transport"
    if feature.startswith("be_"):
        return "built environment"
    if feature.startswith("acs_"):
        return "sociodemographic"
    return "other"


def format_float(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def escape_markdown_cell(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    return text.replace("|", "\\|")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(escape_markdown_cell(col) for col in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(escape_markdown_cell(row[col]) for col in df.columns)
            + " |"
        )
    return "\n".join(lines)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df[columns].copy()
    for col in ["rmse", "mae", "r2", "rmse_std", "r2_std"]:
        if col in view.columns:
            view[col] = view[col].map(format_float)
    return dataframe_to_markdown(view)


def build_report(
    comparison: pd.DataFrame,
    xgb: dict[str, Any],
    gnn: dict[str, Any],
    shap: dict[str, Any] | None,
) -> str:
    lines: list[str] = []
    lines.append("# Model Comparison Status")
    lines.append("")
    lines.append(
        "This report compares the current leakage-safe models for predicting the "
        "network-distance accessibility target `Y`. The target is a proxy index "
        "built from reachable LEHD jobs and OSM amenities, while model inputs are "
        "restricted to public transport, built environment, and ACS features."
    )
    lines.append("")
    lines.append("## Headline Metrics")
    lines.append("")
    lines.append(
        markdown_table(
            comparison,
            [
                "model",
                "experiment",
                "evaluation_unit",
                "rmse",
                "mae",
                "r2",
                "rmse_std",
                "r2_std",
            ],
        )
    )
    lines.append("")
    lines.append("## Current Interpretation")
    lines.append("")
    lines.append(
        "- XGBoost is still the cleanest tree baseline because it has the same "
        "single spatial CV setup and leave-one-city-out evaluation."
    )
    lines.append(
        "- Single-spatial-CV GNN rows are directly comparable to pooled XGBoost "
        "single spatial CV; leave-one-city-out GNN rows are directly comparable "
        "to XGBoost LOCO."
    )
    lines.append(
        "- The low leave-one-city-out XGBoost score means the model learns within-city "
        "spatial structure much better than it transfers to a completely unseen "
        "city. That is useful evidence, not a pipeline failure."
    )
    lines.append("")
    lines.append("## Data And Graph Scope")
    lines.append("")
    lines.append(
        f"- Cities: {', '.join(str(city) for city in xgb['cities'])}."
    )
    lines.append(
        f"- Rows/features: {int(xgb['n_rows']):,} block rows and "
        f"{int(xgb['n_features'])} leakage-safe node features."
    )
    lines.append(
        f"- GNN graph: {int(gnn['n_edges']):,} undirected block-adjacency edges "
        "across disconnected city graphs."
    )
    lines.append("")
    if shap is not None and shap.get("top_features"):
        top = pd.DataFrame(shap["top_features"]).head(10).copy()
        top["feature_family"] = top["feature"].map(feature_family)
        lines.append("## Current XGBoost SHAP Drivers")
        lines.append("")
        lines.append(
            dataframe_to_markdown(
                top[
                    [
                        "feature",
                        "feature_family",
                        "mean_abs_shap",
                        "feature_shap_corr",
                    ]
                ]
                .rename(
                    columns={
                        "mean_abs_shap": "mean_abs_shap",
                        "feature_shap_corr": "direction_corr",
                    }
                )
                .round(
                    {
                        "mean_abs_shap": 4,
                        "direction_corr": 4,
                    }
                )
            )
        )
        lines.append("")
    lines.append("## Next Validation Step")
    lines.append("")
    lines.append(
        "For final thesis numbers, use the single-spatial-CV rows as the main "
        "within-city generalization comparison, and keep leave-one-city-out as the "
        "separate cross-city transfer stress test."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    xgb = load_json(args.xgboost_summary)
    gnn = load_json(args.gnn_summary)
    shap = maybe_load_json(args.shap_summary)

    rows: list[dict[str, Any]] = []
    add_xgboost_rows(rows, xgb)
    add_gnn_rows(rows, gnn)
    comparison = pd.DataFrame(rows)

    outputs_root = Path(args.outputs_root)
    tables_dir = outputs_root / "tables"
    reports_dir = outputs_root / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    table_path = tables_dir / f"{args.output_prefix}_metrics.csv"
    report_path = reports_dir / f"{args.output_prefix}_report.md"
    comparison.to_csv(table_path, index=False)
    report_path.write_text(
        build_report(comparison, xgb, gnn, shap),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "comparison_table": str(table_path),
                "report": str(report_path),
                "n_rows": int(len(comparison)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

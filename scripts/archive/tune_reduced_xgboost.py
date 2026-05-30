"""Small hyperparameter search for the reduced multi-city XGBoost baseline.

The search is intentionally modest: one pooled spatial split, reduced features,
and a fixed candidate grid. Candidates are ranked on the spatial validation
region, then the best candidate is reported on the untouched spatial test region.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

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
    select_feature_set,
    string_key_records,
)
from src.models.metrics import regression_metrics  # noqa: E402
from src.utils.cities import DEFAULT_CITIES, validate_cities  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small spatial-validation hyperparameter search for reduced XGBoost."
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
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--output-prefix",
        default="multicity_xgboost_reduced_tuning",
    )
    return parser.parse_args()


def candidate_grid() -> list[dict[str, float | int | str]]:
    """A compact, interpretable tuning grid around the current baseline."""
    return [
        {
            "candidate": "current_default",
            "n_estimators": 600,
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "candidate": "shallower_regularized",
            "n_estimators": 500,
            "learning_rate": 0.04,
            "max_depth": 3,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "min_child_weight": 5,
            "reg_alpha": 0.0,
            "reg_lambda": 3.0,
        },
        {
            "candidate": "fast_shallow",
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 3,
            "reg_alpha": 0.0,
            "reg_lambda": 2.0,
        },
        {
            "candidate": "medium_faster",
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 3,
            "reg_alpha": 0.0,
            "reg_lambda": 2.0,
        },
        {
            "candidate": "deeper_regularized",
            "n_estimators": 600,
            "learning_rate": 0.03,
            "max_depth": 5,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "min_child_weight": 5,
            "reg_alpha": 0.05,
            "reg_lambda": 5.0,
        },
        {
            "candidate": "slow_medium",
            "n_estimators": 800,
            "learning_rate": 0.02,
            "max_depth": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 2,
            "reg_alpha": 0.0,
            "reg_lambda": 2.0,
        },
        {
            "candidate": "slow_shallow",
            "n_estimators": 800,
            "learning_rate": 0.03,
            "max_depth": 3,
            "subsample": 0.90,
            "colsample_bytree": 0.85,
            "min_child_weight": 3,
            "reg_alpha": 0.0,
            "reg_lambda": 2.0,
        },
        {
            "candidate": "regularized_default_depth",
            "n_estimators": 600,
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 5,
            "reg_alpha": 0.05,
            "reg_lambda": 3.0,
        },
    ]


def xgb_params(candidate: dict[str, float | int | str], random_state: int, n_jobs: int) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "n_estimators": int(candidate["n_estimators"]),
        "learning_rate": float(candidate["learning_rate"]),
        "max_depth": int(candidate["max_depth"]),
        "subsample": float(candidate["subsample"]),
        "colsample_bytree": float(candidate["colsample_bytree"]),
        "min_child_weight": float(candidate["min_child_weight"]),
        "reg_alpha": float(candidate["reg_alpha"]),
        "reg_lambda": float(candidate["reg_lambda"]),
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
        "tree_method": "hist",
        "eval_metric": "rmse",
    }


def score_split(
    model: XGBRegressor,
    split_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    split_name: str,
    train_mean: float,
) -> dict[str, float | int | str]:
    y_true = split_df[target_column].to_numpy(dtype=float)
    y_pred = model.predict(split_df[feature_columns]).astype(float)
    metrics = regression_metrics(y_true, y_pred)
    baseline = regression_metrics(y_true, np.full(len(y_true), train_mean, dtype=float))
    return {
        "split": split_name,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "baseline_rmse": baseline["rmse"],
        "baseline_r2": baseline["r2"],
        "n_rows": int(len(split_df)),
        "target_mean": float(np.mean(y_true)),
        "prediction_mean": float(np.mean(y_pred)),
        "prediction_bias": float(np.mean(y_pred) - np.mean(y_true)),
    }


def main() -> None:
    args = parse_args()
    cities = validate_cities([str(city) for city in args.cities])
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
    val = split_df[split_df["split"] == "val"].copy()
    test = split_df[split_df["split"] == "test"].copy()
    train_mean = float(train[args.target_column].mean())

    rows: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    for i, candidate in enumerate(candidate_grid()):
        params = xgb_params(
            candidate,
            random_state=int(args.random_state),
            n_jobs=int(args.n_jobs),
        )
        model = XGBRegressor(**params)
        model.fit(train[feature_columns], train[args.target_column], verbose=False)
        for split_name, frame in [("train", train), ("val", val), ("test", test)]:
            score = score_split(
                model=model,
                split_df=frame,
                feature_columns=feature_columns,
                target_column=args.target_column,
                split_name=split_name,
                train_mean=train_mean,
            )
            rows.append(
                {
                    "candidate_rank_input": i,
                    **candidate,
                    **score,
                }
            )
        candidate_summaries.append(
            {
                "candidate_rank_input": i,
                **candidate,
                "params": params,
            }
        )

    results = pd.DataFrame(rows)
    val_results = results[results["split"] == "val"].copy()
    val_results = val_results.sort_values(
        ["r2", "rmse"],
        ascending=[False, True],
    ).reset_index(drop=True)
    val_results["validation_rank"] = np.arange(1, len(val_results) + 1)
    rank_map = dict(
        zip(
            val_results["candidate"].astype(str),
            val_results["validation_rank"].astype(int),
        )
    )
    results["validation_rank"] = results["candidate"].astype(str).map(rank_map).astype(int)
    results = results.sort_values(["validation_rank", "split"]).reset_index(drop=True)
    best_name = str(val_results.iloc[0]["candidate"])
    best_rows = results[results["candidate"] == best_name].copy()
    best_test = best_rows[best_rows["split"] == "test"].iloc[0].to_dict()

    output_root = Path(args.outputs_root)
    tables_dir = output_root / "tables"
    metrics_dir = output_root / "metrics"
    reports_dir = output_root / "reports"
    for path in [tables_dir, metrics_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    results_path = tables_dir / f"{args.output_prefix}_results.csv"
    best_path = tables_dir / f"{args.output_prefix}_best_by_split.csv"
    results.to_csv(results_path, index=False)
    best_rows.to_csv(best_path, index=False)

    metadata: dict[str, Any] = {
        "model": "xgboost",
        "experiment": "small_reduced_hyperparameter_search",
        "selection_rule": "highest validation R2, then lowest validation RMSE",
        "cities": cities,
        "target_column": args.target_column,
        "target_view": str(args.target_view),
        "target_view_metadata": target_view_metadata,
        "feature_set": "reduced",
        "feature_set_metadata": feature_set_metadata,
        "feature_view": str(args.feature_view),
        "feature_view_metadata": feature_view_metadata,
        "n_features": int(len(feature_columns)),
        "n_candidates": int(len(candidate_grid())),
        "split_counts": split_df["split"].value_counts().to_dict(),
        "best_candidate": best_name,
        "best_validation": val_results.iloc[0].to_dict(),
        "best_test": best_test,
        "candidate_ranking": string_key_records(val_results),
        "artifacts": {
            "results": str(results_path),
            "best_by_split": str(best_path),
            "summary": str(metrics_dir / f"{args.output_prefix}_summary.json"),
            "report": str(reports_dir / f"{args.output_prefix}_report.md"),
        },
    }
    summary_path = metrics_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report_lines = [
        "# Reduced XGBoost Hyperparameter Search",
        "",
        "Candidates were selected on the pooled spatial validation split.",
        "The spatial test split was kept untouched until final reporting.",
        "",
        f"- Best candidate: `{best_name}`",
        f"- Validation R2: {float(val_results.iloc[0]['r2']):.4f}",
        f"- Test R2: {float(best_test['r2']):.4f}",
        f"- Test RMSE: {float(best_test['rmse']):.4f}",
        "",
        "## Validation Ranking",
        "",
        "| rank | candidate | val_r2 | val_rmse | test_r2 | test_rmse |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in val_results.itertuples(index=False):
        candidate = str(row.candidate)
        test_row = results[(results["candidate"] == candidate) & (results["split"] == "test")].iloc[0]
        report_lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row.validation_rank)),
                    candidate,
                    f"{float(row.r2):.4f}",
                    f"{float(row.rmse):.4f}",
                    f"{float(test_row['r2']):.4f}",
                    f"{float(test_row['rmse']):.4f}",
                ]
            )
            + " |"
        )
    report_path = reports_dir / f"{args.output_prefix}_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

"""Write diagnostics for a city's leakage-safe model dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.split_diagnostics import (  # noqa: E402
    build_feature_shift_table,
    build_split_diagnostics,
)
from src.modeling.dataset import validate_model_feature_columns  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose target and feature balance across model splits."
    )
    parser.add_argument("--city", required=True, help="City slug, for example: denver")
    parser.add_argument(
        "--interim-root",
        default="data/interim",
        help="Root containing city modeling artifacts",
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Root for diagnostic outputs",
    )
    parser.add_argument(
        "--target-column",
        default="Y_global_log_minmax",
        help="Target column to inspect",
    )
    return parser.parse_args()


def load_dataset_and_features(
    city: str,
    interim_root: str | Path,
) -> tuple[pd.DataFrame, list[str]]:
    modeling_dir = Path(interim_root) / city / "modeling"
    dataset_path = modeling_dir / "model_dataset.parquet"
    metadata_path = modeling_dir / "metadata.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Model dataset not found: {dataset_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

    dataset = pd.read_parquet(dataset_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not all(
        isinstance(col, str) for col in feature_columns
    ):
        raise ValueError("Model metadata must contain a string list named 'feature_columns'.")
    validate_model_feature_columns(feature_columns)
    return dataset, feature_columns


def main() -> None:
    args = parse_args()
    city = args.city.strip().lower()
    dataset, feature_columns = load_dataset_and_features(city, args.interim_root)

    diagnostics = build_split_diagnostics(
        dataset=dataset,
        feature_columns=feature_columns,
        target_column=args.target_column,
    )
    feature_shift = build_feature_shift_table(
        dataset=dataset,
        feature_columns=feature_columns,
    ).sort_values("abs_standardized_mean_difference", ascending=False)

    diagnostics_dir = Path(args.outputs_root) / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = diagnostics_dir / f"{city}_split_diagnostics.json"
    shift_path = diagnostics_dir / f"{city}_feature_shift.csv"

    with diagnostics_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    feature_shift.to_csv(shift_path, index=False)

    print(
        json.dumps(
            {"diagnostics": str(diagnostics_path), "feature_shift": str(shift_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Build a leakage-safe supervised modeling table for one city."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.dataset import assemble_model_dataset, load_feature_columns  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a supervised model dataset for one city."
    )
    parser.add_argument("--city", required=True, help="City slug, for example: denver")
    parser.add_argument(
        "--interim-root",
        default="data/interim",
        help="Root containing city interim artifacts",
    )
    parser.add_argument(
        "--output-root",
        default="data/interim",
        help="Output root for model dataset artifacts",
    )
    parser.add_argument(
        "--target-column",
        default="Y_global_log_minmax",
        help="Target column to predict; default is the cross-city comparable global-log target.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    city = args.city.strip().lower()

    city_dir = Path(args.interim_root) / city
    feature_path = city_dir / "features" / "feature_table.parquet"
    feature_metadata_path = city_dir / "features" / "metadata.json"
    target_path = city_dir / "target" / "target_table.parquet"
    splits_path = city_dir / "target" / "spatial_splits.csv"

    for path in [feature_path, feature_metadata_path, target_path, splits_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required modeling input not found: {path}")

    feature_df = pd.read_parquet(feature_path)
    feature_columns = load_feature_columns(feature_metadata_path)
    target_df = pd.read_parquet(target_path)
    splits_df = pd.read_csv(splits_path, dtype={"block_id": "string"})

    dataset = assemble_model_dataset(
        feature_df=feature_df,
        target_df=target_df,
        splits_df=splits_df,
        feature_columns=feature_columns,
        target_column=args.target_column,
    )

    out_dir = Path(args.output_root) / city / "modeling"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_dir / "model_dataset.parquet", index=False)
    dataset.to_csv(out_dir / "model_dataset.csv", index=False)

    metadata = {
        "city": city,
        "target_column": args.target_column,
        "n_rows": int(len(dataset)),
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "split_counts": dataset["split"].value_counts().to_dict(),
        "inputs": {
            "features": str(feature_path),
            "feature_metadata": str(feature_metadata_path),
            "target": str(target_path),
            "splits": str(splits_path),
        },
        "outputs": {
            "parquet": str(out_dir / "model_dataset.parquet"),
            "csv": str(out_dir / "model_dataset.csv"),
        },
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

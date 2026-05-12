"""Persist cross-city comparable target columns into city target tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.target.global_target import (  # noqa: E402
    GLOBAL_TARGET_VIEWS,
    build_global_target_columns,
    load_multicity_target_components,
)
from src.utils.cities import DEFAULT_CITIES, validate_cities  # noqa: E402


LEGACY_ALIASES = {
    "E": "E_city_relative",
    "A": "A_city_relative",
    "Y": "Y_city_relative",
    "Y_60_40": "Y_60_40_city_relative",
    "Y_40_60": "Y_40_60_city_relative",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add global cross-city target columns to city target tables."
    )
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    parser.add_argument("--interim-root", default="data/interim")
    parser.add_argument(
        "--target-views",
        nargs="+",
        choices=GLOBAL_TARGET_VIEWS,
        default=["global_log_minmax", "global_minmax"],
        help="Global target columns to persist.",
    )
    parser.add_argument(
        "--set-main-y",
        action="store_true",
        help=(
            "Replace E/A/Y with E/A/Y_global_log_minmax while preserving city-relative "
            "aliases. Off by default to avoid surprising downstream changes."
        ),
    )
    return parser.parse_args()


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the last occurrence of duplicate columns after repeated augmentation."""
    return df.loc[:, ~df.columns.duplicated(keep="last")].copy()


def add_legacy_aliases(target: pd.DataFrame) -> pd.DataFrame:
    out = target.copy()
    for source, alias in LEGACY_ALIASES.items():
        if source in out.columns and alias not in out.columns:
            out[alias] = out[source]
    return out


def write_city_target(
    city: str,
    target: pd.DataFrame,
    global_targets: pd.DataFrame,
    interim_root: str | Path,
    set_main_y: bool,
) -> dict[str, Any]:
    target_dir = Path(interim_root) / city / "target"
    target_path = target_dir / "target_table.parquet"
    csv_path = target_dir / "target_table.csv"
    metadata_path = target_dir / "metadata.json"

    city_globals = global_targets[global_targets["city"] == city].copy()
    city_globals = city_globals.drop(columns=["city"])
    target = add_legacy_aliases(target)
    target["block_id"] = target["block_id"].astype(str)
    city_globals["block_id"] = city_globals["block_id"].astype(str)
    merged = target.merge(
        city_globals.drop(columns=["node_id"]),
        on="block_id",
        how="left",
        validate="one_to_one",
    )
    merged = _dedupe_columns(merged)

    if merged.filter(regex=r"^Y_global_").isna().any().any():
        raise ValueError(f"Missing global target values after merging {city}.")

    if set_main_y:
        for base in ["E", "A", "Y"]:
            source = f"{base}_global_log_minmax"
            if source not in merged.columns:
                raise KeyError(f"Cannot set main {base}; missing {source}.")
            merged[base] = merged[source]

    merged.to_parquet(target_path, index=False)
    merged.to_csv(csv_path, index=False)

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["target_columns"] = {
        "main": "Y_global_log_minmax" if not set_main_y else "Y",
        "main_definition": "global_log_minmax",
        "legacy_city_relative": "Y_city_relative",
        "global_log_minmax": "Y_global_log_minmax",
        "global_minmax": "Y_global_minmax",
    }
    metadata["target_table_augmented_with_global_targets"] = True
    metadata["set_main_y_to_global_log_minmax"] = bool(set_main_y)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "city": city,
        "rows": int(len(merged)),
        "target_table": str(target_path),
        "main_target_column": metadata["target_columns"]["main"],
        "has_legacy_city_relative_y": "Y_city_relative" in merged.columns,
        "has_global_log_target": "Y_global_log_minmax" in merged.columns,
    }


def main() -> None:
    args = parse_args()
    cities = validate_cities([str(city) for city in args.cities])
    target_components = load_multicity_target_components(cities, args.interim_root)

    global_frames: list[pd.DataFrame] = []
    metadata_by_view: dict[str, Any] = {}
    for view in [str(view) for view in args.target_views]:
        frame, metadata = build_global_target_columns(
            target_components=target_components,
            cities=cities,
            interim_root=args.interim_root,
            target_view=view,
        )
        keep_columns = [
            "node_id",
            "city",
            "block_id",
            f"E_{view}",
            f"A_{view}",
            f"Y_{view}",
        ]
        norm_columns = [
            col
            for col in frame.columns
            if col.startswith(("jobs_norm_", "amenity_norm_mean_"))
        ]
        global_frames.append(frame[keep_columns + norm_columns])
        metadata_by_view[view] = metadata

    global_targets = pd.concat(global_frames, axis=1)
    global_targets = _dedupe_columns(global_targets)

    city_results = []
    for city in cities:
        target_path = Path(args.interim_root) / city / "target" / "target_table.parquet"
        target = pd.read_parquet(target_path)
        city_results.append(
            write_city_target(
                city=city,
                target=target,
                global_targets=global_targets,
                interim_root=args.interim_root,
                set_main_y=bool(args.set_main_y),
            )
        )

    summary = {
        "cities": cities,
        "target_views": [str(view) for view in args.target_views],
        "main_target_column": "Y_global_log_minmax" if not args.set_main_y else "Y",
        "metadata_by_view": metadata_by_view,
        "city_results": city_results,
    }
    summary_path = Path(args.interim_root) / "global_targets_metadata.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["metadata_path"] = str(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

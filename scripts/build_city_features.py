"""Build leakage-safe PT, BE, and ACS predictors for one city.

The output intentionally excludes target ingredients, coordinates, jobs, and
amenity counts. It is meant to be joined to the target table by ``block_id``
during model training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_osm import find_state_osm_pbf  # noqa: E402
from src.data.load_acs import build_block_acs_features  # noqa: E402
from src.features.gtfs_features import build_gtfs_features  # noqa: E402
from src.features.network_features import build_network_features  # noqa: E402
from src.utils.cities import resolve_state  # noqa: E402


FORBIDDEN_FEATURE_PREFIXES = (
    "jobs_",
    "amen_",
    "amenity_",
    "E_",
    "A_",
    "Y_",
)
FORBIDDEN_FEATURE_COLUMNS = {
    "E",
    "A",
    "Y",
    "Y_60_40",
    "Y_40_60",
    "x_m",
    "y_m",
    "lon",
    "lat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe PT/BE feature table for one city."
    )
    parser.add_argument("--city", required=True, help="City slug, for example: denver")
    parser.add_argument("--state", default=None, help="State code override, for example: co")
    parser.add_argument(
        "--backbone-root",
        default="data/interim",
        help="Root containing {city}/backbone outputs",
    )
    parser.add_argument(
        "--gtfs-root",
        default="data/raw/gtfs",
        help="Root containing city GTFS folders",
    )
    parser.add_argument(
        "--osm-root",
        default="data/raw/osm",
        help="Root containing state OSM PBF folders",
    )
    parser.add_argument(
        "--acs-root",
        default="data/raw/acs",
        help="Root containing curated ACS tract CSV/ZIP files",
    )
    parser.add_argument(
        "--skip-acs",
        action="store_true",
        help="Build only PT and BE features; use for ablations or missing ACS data",
    )
    parser.add_argument(
        "--output-root",
        default="data/interim",
        help="Output root for city feature artifacts",
    )
    return parser.parse_args()


def validate_feature_columns(feature_columns: list[str]) -> None:
    forbidden = [
        col
        for col in feature_columns
        if col in FORBIDDEN_FEATURE_COLUMNS or col.startswith(FORBIDDEN_FEATURE_PREFIXES)
    ]
    if forbidden:
        raise ValueError(f"Feature table contains leakage-prone columns: {forbidden}")


def main() -> None:
    args = parse_args()
    city = args.city.strip().lower()
    state = resolve_state(city, args.state)

    backbone_dir = Path(args.backbone_root) / city / "backbone"
    blocks_path = backbone_dir / "blocks.gpkg"
    centroids_path = backbone_dir / "centroids.gpkg"
    adjacency_path = backbone_dir / "adjacency.csv"
    if not blocks_path.exists():
        raise FileNotFoundError(f"Backbone blocks not found: {blocks_path}")
    if not centroids_path.exists():
        raise FileNotFoundError(f"Backbone centroids not found: {centroids_path}")
    if not adjacency_path.exists():
        raise FileNotFoundError(f"Backbone adjacency not found: {adjacency_path}")

    blocks = gpd.read_file(blocks_path)
    centroids = gpd.read_file(centroids_path)
    if blocks.crs is None or centroids.crs is None:
        raise ValueError("Backbone blocks and centroids must both have CRS values.")
    if centroids.crs != blocks.crs:
        centroids = centroids.to_crs(blocks.crs)

    blocks["block_id"] = blocks["block_id"].astype(str)
    centroids["block_id"] = centroids["block_id"].astype(str)
    adjacency = pd.read_csv(
        adjacency_path,
        dtype={"src_block_id": "string", "dst_block_id": "string"},
    )

    gtfs_features, gtfs_metadata = build_gtfs_features(
        blocks=blocks,
        centroids=centroids,
        gtfs_city_dir=Path(args.gtfs_root) / city,
    )
    osm_pbf = find_state_osm_pbf(state, args.osm_root)
    network_features, network_metadata = build_network_features(
        blocks=blocks,
        adjacency_edges=adjacency,
        osm_pbf_path=osm_pbf,
    )

    feature_df = gtfs_features.merge(network_features, on="block_id", how="outer")
    acs_metadata: dict[str, object] | None = None
    if not args.skip_acs:
        acs_features, acs_metadata = build_block_acs_features(
            blocks["block_id"],
            acs_root=args.acs_root,
        )
        feature_df = feature_df.merge(acs_features, on="block_id", how="outer")

    feature_df = feature_df.copy().set_index("block_id").reindex(blocks["block_id"]).reset_index()
    feature_columns = [col for col in feature_df.columns if col != "block_id"]
    validate_feature_columns(feature_columns)

    out_dir = Path(args.output_root) / city / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(out_dir / "feature_table.parquet", index=False)
    feature_df.to_csv(out_dir / "feature_table.csv", index=False)

    metadata = {
        "city": city,
        "state": state,
        "n_blocks": int(len(feature_df)),
        "feature_columns": feature_columns,
        "feature_policy": {
            "included": "Public transport supply, built-environment structure, and ACS tract context.",
            "excluded": "Target ingredients, jobs, amenities, and raw coordinates.",
        },
        "gtfs": gtfs_metadata,
        "network": network_metadata,
        "acs": acs_metadata,
        "osm_pbf": str(osm_pbf),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

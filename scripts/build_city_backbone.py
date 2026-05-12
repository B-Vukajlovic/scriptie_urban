"""Build a single-city spatial backbone from raw TIGER data.

Outputs per city:
- blocks.gpkg (projected block polygons)
- centroids.gpkg (representative points with projected and lon/lat coords)
- adjacency.csv (undirected block adjacency edge list)
- qc_summary.json (basic integrity checks)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.adjacency import build_adjacency_edges
from src.preprocessing.blocks import (
    load_blocks_for_state,
    load_place_boundary,
    select_blocks_for_place,
    standardize_block_columns,
)
from src.preprocessing.centroids import build_block_centroids
from src.preprocessing.projections import estimate_utm_epsg
from src.utils.cities import resolve_city_inputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one-city spatial backbone from raw TIGER files."
    )
    parser.add_argument("--city", required=True, help="City slug, for example: new_york")
    parser.add_argument(
        "--state",
        default=None,
        help="Two-letter lowercase state code, for example: ny",
    )
    parser.add_argument(
        "--place-geoid",
        default=None,
        help="TIGER place GEOID, for example: 3651000",
    )
    parser.add_argument(
        "--tiger-root",
        default="data/raw/tiger",
        help="Root folder containing tiger/{blocks,place,...}",
    )
    parser.add_argument(
        "--output-root",
        default="data/interim",
        help="Root output folder",
    )
    parser.add_argument(
        "--place-selection",
        choices=["intersection", "representative_point", "intersects"],
        default="representative_point",
        help="How to select blocks for the city place boundary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    city = args.city.strip().lower()
    state_code, place_geoid = resolve_city_inputs(city, args.state, args.place_geoid)

    tiger_root = Path(args.tiger_root)
    output_dir = Path(args.output_root) / city / "backbone"
    output_dir.mkdir(parents=True, exist_ok=True)

    place = load_place_boundary(
        tiger_root=tiger_root,
        state_abbrev=state_code,
        place_geoid=place_geoid,
    )
    minx, miny, maxx, maxy = map(float, place.total_bounds.tolist())
    bbox = (minx, miny, maxx, maxy)
    blocks_raw = load_blocks_for_state(
        tiger_root=tiger_root,
        state_abbrev=state_code,
        bbox=bbox,
    )

    blocks = select_blocks_for_place(
        blocks_raw,
        place,
        mode=args.place_selection,
    )
    blocks = standardize_block_columns(blocks)

    epsg = estimate_utm_epsg(blocks)
    blocks_proj = blocks.to_crs(epsg)

    centroids = build_block_centroids(blocks_proj)
    edges = build_adjacency_edges(blocks_proj, id_col="block_id")

    node_ids = set(blocks_proj["block_id"])
    connected_ids = set(edges["src_block_id"]).union(set(edges["dst_block_id"]))
    isolated_count = len(node_ids - connected_ids)

    qc_summary = {
        "city": city,
        "state": state_code,
        "place_geoid": place_geoid,
        "place_selection": args.place_selection,
        "crs_epsg": epsg,
        "n_blocks": int(len(blocks_proj)),
        "n_centroids": int(len(centroids)),
        "n_edges": int(len(edges)),
        "n_duplicate_block_ids": int(blocks_proj["block_id"].duplicated().sum()),
        "n_invalid_geometries": int((~blocks_proj.geometry.is_valid).sum()),
        "n_isolated_blocks": int(isolated_count),
    }

    blocks_proj.to_file(output_dir / "blocks.gpkg", layer="blocks", driver="GPKG")
    centroids.to_file(output_dir / "centroids.gpkg", layer="centroids", driver="GPKG")
    edges.to_csv(output_dir / "adjacency.csv", index=False)
    with (output_dir / "qc_summary.json").open("w", encoding="utf-8") as f:
        json.dump(qc_summary, f, indent=2)

    print(json.dumps(qc_summary, indent=2))


if __name__ == "__main__":
    main()

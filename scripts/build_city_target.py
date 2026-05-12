"""Build a single-city accessibility target table from backbone, LEHD, and OSM.

This stage computes:
- reachable jobs by 1-5 km
- reachable amenities by category and radius
- normalized E (employment) and A (amenities) components
- final Y target variants
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_lehd import find_wac_file, load_jobs_by_block  # noqa: E402
from src.data.load_osm import find_state_osm_pbf, load_osm_amenities_in_bbox  # noqa: E402
from src.evaluation.spatial_splits import build_spatial_train_val_test_splits  # noqa: E402
from src.preprocessing.adjacency import build_adjacency_edges  # noqa: E402
from src.preprocessing.blocks import load_blocks_for_state, standardize_block_columns  # noqa: E402
from src.preprocessing.centroids import build_block_centroids  # noqa: E402
from src.target.build_target import build_target_table  # noqa: E402
from src.target.graph_reachability import (  # noqa: E402
    batched_reachability_sums,
    build_block_graph_csr,
)
from src.target.osm_network_graph import (  # noqa: E402
    attach_destination_nodes,
    build_road_graph,
    load_osm_road_lines_in_bbox,
    snap_points_to_nodes,
)
from src.target.reachable_amenities import (  # noqa: E402
    amenity_counts_by_radius,
    map_amenity_category,
)
from src.target.reachable_jobs import jobs_reachability_by_radius  # noqa: E402
from src.utils.cities import resolve_state  # noqa: E402


Bounds = tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one-city target table from opportunities.")
    parser.add_argument("--city", required=True, help="City slug, for example: denver")
    parser.add_argument("--state", default=None, help="State code override, for example: co")
    parser.add_argument(
        "--backbone-root",
        default="data/interim",
        help="Root containing {city}/backbone outputs",
    )
    parser.add_argument(
        "--lehd-root",
        default="data/raw/lehd",
        help="Root containing LEHD state/ and city/ folders",
    )
    parser.add_argument(
        "--prefer-city-lehd",
        action="store_true",
        help="Prefer city LEHD extracts over statewide LEHD when both exist.",
    )
    parser.add_argument(
        "--osm-root",
        default="data/raw/osm",
        help="Root containing state OSM PBF folders",
    )
    parser.add_argument(
        "--tiger-root",
        default="data/raw/tiger",
        help="Root containing TIGER block files",
    )
    parser.add_argument(
        "--output-root",
        default="data/interim",
        help="Output root for city target artifacts",
    )
    parser.add_argument(
        "--distance-engine",
        choices=["euclidean", "block_graph", "osm_network"],
        default="osm_network",
        help="Distance engine for reachability.",
    )
    parser.add_argument(
        "--graph-batch-size",
        type=int,
        default=128,
        help="Batch size for graph shortest-path batches",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for spatial split artifact",
    )
    parser.add_argument(
        "--split-val-frac",
        type=float,
        default=0.15,
        help="Validation fraction for split artifact",
    )
    parser.add_argument(
        "--split-test-frac",
        type=float,
        default=0.15,
        help="Test fraction for split artifact",
    )
    parser.add_argument(
        "--split-grid-bins-x",
        type=int,
        default=8,
        help="Spatial grid quantile bins on x",
    )
    parser.add_argument(
        "--split-grid-bins-y",
        type=int,
        default=8,
        help="Spatial grid quantile bins on y",
    )
    return parser.parse_args()


def bounds_tuple(bounds: np.ndarray) -> Bounds:
    """Convert GeoPandas bounds arrays into a statically typed bbox tuple."""
    minx, miny, maxx, maxy = map(float, bounds.tolist())
    return (minx, miny, maxx, maxy)


def main() -> None:
    args = parse_args()
    city = args.city.strip().lower()
    state = resolve_state(city, args.state)

    radii_km = [1, 2, 3, 4, 5]
    radii_m = [float(r * 1000) for r in radii_km]
    distance_weights = {1: 0.30, 2: 0.25, 3: 0.20, 4: 0.15, 5: 0.10}
    categories = ["food_retail", "healthcare", "education", "recreation", "public_services"]

    backbone_dir = Path(args.backbone_root) / city / "backbone"
    centroids_path = backbone_dir / "centroids.gpkg"
    blocks_path = backbone_dir / "blocks.gpkg"
    if not centroids_path.exists():
        raise FileNotFoundError(f"Backbone centroids not found: {centroids_path}")

    centroids = gpd.read_file(centroids_path)
    if centroids.crs is None:
        raise ValueError("Backbone centroids has no CRS.")
    base_crs = centroids.crs
    city_blocks = gpd.read_file(blocks_path)[["block_id", "geometry"]].copy()
    if city_blocks.crs is None:
        raise ValueError("Backbone blocks has no CRS.")
    if city_blocks.crs != base_crs:
        city_blocks = city_blocks.to_crs(base_crs)

    centroids["block_id"] = centroids["block_id"].astype(str)
    base = centroids[["block_id", "x_m", "y_m", "lon", "lat", "geometry"]].copy()
    origins_xy = np.column_stack([base["x_m"].to_numpy(), base["y_m"].to_numpy()])

    study_area_geom: BaseGeometry = city_blocks.geometry.union_all()
    study_area = gpd.GeoSeries([study_area_geom], crs=base_crs)
    max_radius_m = float(max(radii_m))
    study_area_buffer = study_area.buffer(max_radius_m)
    study_area_buffer_wgs84 = study_area_buffer.to_crs(4326)
    study_area_buffer_geom = study_area_geom.buffer(max_radius_m)
    bbox_wgs84 = bounds_tuple(study_area_buffer_wgs84.total_bounds)

    destination_blocks_raw = load_blocks_for_state(
        tiger_root=Path(args.tiger_root),
        state_abbrev=state,
        bbox=bbox_wgs84,
    )
    destination_blocks = standardize_block_columns(destination_blocks_raw)
    destination_blocks = destination_blocks.to_crs(base_crs)
    destination_blocks = destination_blocks[
        destination_blocks.geometry.intersects(study_area_buffer_geom)
    ].copy()
    destination_blocks = destination_blocks.reset_index(drop=True)
    destination_centroids = build_block_centroids(destination_blocks)
    destination_centroids["block_id"] = destination_centroids["block_id"].astype(str)

    wac_path, lehd_scope = find_wac_file(
        city=city,
        state=state,
        lehd_root=args.lehd_root,
        prefer_state=not args.prefer_city_lehd,
    )
    jobs_df = load_jobs_by_block(wac_path)
    city_block_ids = set(base["block_id"].astype(str))
    source_job_block_ids = set(jobs_df["block_id"].astype(str))
    jobs_dest = destination_centroids[["block_id", "x_m", "y_m"]].merge(
        jobs_df,
        on="block_id",
        how="inner",
    )
    jobs_dest["jobs"] = jobs_dest["jobs"].fillna(0.0)
    jobs_dest = jobs_dest[jobs_dest["jobs"] > 0].copy()

    pbf_path = find_state_osm_pbf(state_code=state, osm_root=args.osm_root)
    osm_amenities = load_osm_amenities_in_bbox(pbf_path, bbox_wgs84)
    osm_amenities["category"] = map_amenity_category(
        amenity=osm_amenities["amenity"],
        shop=osm_amenities["shop"],
        leisure=osm_amenities["leisure"],
    )
    amenities = osm_amenities[osm_amenities["category"].notna()].copy()
    amenities = amenities.to_crs(base_crs)
    qa_metrics: dict[str, float | int | None] = {}

    if args.distance_engine == "euclidean":
        jobs_by_radius = jobs_reachability_by_radius(
            origins_xy=origins_xy,
            destinations_xy=jobs_dest[["x_m", "y_m"]].to_numpy(),
            destination_jobs=jobs_dest["jobs"].to_numpy(),
            radii_m=radii_m,
        )

        amenity_counts = amenity_counts_by_radius(
            origins_xy=origins_xy,
            amenities_xy=np.column_stack(
                [amenities.geometry.x.to_numpy(), amenities.geometry.y.to_numpy()]
            ),
            amenity_categories=amenities["category"],
            radii_m=radii_m,
            categories=categories,
        )

        qa_metrics["network_n_nodes"] = None
        qa_metrics["network_n_components"] = None
        qa_metrics["share_origins_in_largest_component"] = None
        qa_metrics["origin_snap_dist_mean_m"] = 0.0
        qa_metrics["origin_snap_dist_median_m"] = 0.0
        qa_metrics["origin_snap_dist_p95_m"] = 0.0
        qa_metrics["origin_snap_dist_max_m"] = 0.0
    elif args.distance_engine == "block_graph":
        destination_blocks["block_id"] = destination_blocks["block_id"].astype(str)
        destination_centroids = destination_centroids[["block_id", "x_m", "y_m"]].copy()
        adjacency = build_adjacency_edges(destination_blocks, id_col="block_id")
        graph, _ = build_block_graph_csr(
            block_ids=destination_centroids["block_id"],
            x_m=destination_centroids["x_m"].to_numpy(),
            y_m=destination_centroids["y_m"].to_numpy(),
            adjacency_edges=adjacency,
        )
        graph_shape = graph.shape
        if graph_shape is None:
            raise ValueError("block graph has no shape")
        graph_n_nodes = int(graph_shape[0])

        destination_block_index = pd.Index(destination_centroids["block_id"])
        source_block_index = pd.Index(base["block_id"])
        source_indices = destination_block_index.get_indexer(source_block_index)
        if np.any(source_indices < 0):
            raise ValueError(
                "Some city blocks were not found in the buffered destination block graph."
            )

        jobs_per_block = (
            jobs_dest.set_index("block_id")["jobs"]
            .reindex(destination_block_index)
            .fillna(0.0)
            .to_numpy()
        )

        amenity_join = gpd.sjoin(
            amenities[["category", "geometry"]],
            destination_blocks[["block_id", "geometry"]],
            how="left",
            predicate="within",
        )
        amenity_join = amenity_join.dropna(subset=["block_id"]).copy()
        amenity_block_counts = (
            amenity_join.groupby(["block_id", "category"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )

        destination_matrix = np.zeros(
            (len(destination_centroids), 1 + len(categories)),
            dtype=float,
        )
        destination_matrix[:, 0] = jobs_per_block
        for j, category in enumerate(categories, start=1):
            cat_counts = (
                amenity_block_counts[amenity_block_counts["category"] == category]
                .set_index("block_id")["count"]
                .reindex(destination_block_index)
                .fillna(0.0)
                .to_numpy()
            )
            destination_matrix[:, j] = cat_counts

        sums = batched_reachability_sums(
            graph=graph,
            destination_matrix=destination_matrix,
            radii_m=radii_m,
            batch_size=args.graph_batch_size,
            source_indices=source_indices,
        )

        jobs_by_radius = pd.DataFrame(
            {f"jobs_{int(r/1000)}km": sums[float(r)][:, 0] for r in radii_m}
        )

        amenity_cols: dict[str, np.ndarray] = {}
        for j, category in enumerate(categories, start=1):
            for r in radii_m:
                amenity_cols[f"amen_{category}_{int(r/1000)}km"] = sums[float(r)][:, j]
        amenity_counts = pd.DataFrame(amenity_cols)

        n_components, labels = connected_components(graph, directed=False, return_labels=True)
        label_counts = np.bincount(labels)
        largest_label = int(label_counts.argmax())
        qa_metrics["network_n_nodes"] = graph_n_nodes
        qa_metrics["network_n_components"] = int(n_components)
        qa_metrics["share_origins_in_largest_component"] = float(
            (labels[source_indices] == largest_label).mean()
        )
        qa_metrics["origin_snap_dist_mean_m"] = 0.0
        qa_metrics["origin_snap_dist_median_m"] = 0.0
        qa_metrics["origin_snap_dist_p95_m"] = 0.0
        qa_metrics["origin_snap_dist_max_m"] = 0.0
        qa_metrics["share_unreachable_any_5km"] = float(
            (sums[max_radius_m].sum(axis=1) <= 0).mean()
        )
    else:
        roads = load_osm_road_lines_in_bbox(str(pbf_path), bbox_wgs84)
        roads = roads.to_crs(base_crs)
        road_graph, road_nodes_xy = build_road_graph(roads)
        road_graph_shape = road_graph.shape
        if road_graph_shape is None:
            raise ValueError("road graph has no shape")
        road_graph_n_nodes = int(road_graph_shape[0])

        origin_node_idx, origin_snap_dist = snap_points_to_nodes(origins_xy, road_nodes_xy)
        job_xy = jobs_dest[["x_m", "y_m"]].to_numpy()
        jobs_node_idx, jobs_snap_dist = snap_points_to_nodes(job_xy, road_nodes_xy)

        amenity_xy = np.column_stack(
            [amenities.geometry.x.to_numpy(), amenities.geometry.y.to_numpy()]
        )
        amenity_node_idx, amenity_snap_dist = snap_points_to_nodes(amenity_xy, road_nodes_xy)

        job_values = np.zeros((len(jobs_dest), 1 + len(categories)), dtype=float)
        job_values[:, 0] = jobs_dest["jobs"].to_numpy(dtype=float)

        amenity_values = np.zeros((len(amenities), 1 + len(categories)), dtype=float)
        for j, category in enumerate(categories, start=1):
            amenity_values[:, j] = (amenities["category"].to_numpy() == category).astype(float)

        destination_node_idx = np.concatenate([jobs_node_idx, amenity_node_idx])
        destination_offsets = np.concatenate([jobs_snap_dist, amenity_snap_dist])
        destination_values = np.vstack([job_values, amenity_values])
        augmented_graph, destination_matrix = attach_destination_nodes(
            graph=road_graph,
            destination_node_idx=destination_node_idx,
            destination_offsets_m=destination_offsets,
            destination_values=destination_values,
        )

        sums = batched_reachability_sums(
            graph=augmented_graph,
            destination_matrix=destination_matrix,
            radii_m=radii_m,
            batch_size=args.graph_batch_size,
            source_indices=origin_node_idx,
            source_offsets_m=origin_snap_dist,
        )

        jobs_by_radius = pd.DataFrame(
            {f"jobs_{int(r/1000)}km": sums[float(r)][:, 0] for r in radii_m}
        )

        amenity_cols: dict[str, np.ndarray] = {}
        for j, category in enumerate(categories, start=1):
            for r in radii_m:
                amenity_cols[f"amen_{category}_{int(r/1000)}km"] = sums[float(r)][:, j]
        amenity_counts = pd.DataFrame(amenity_cols)

        n_components, labels = connected_components(road_graph, directed=False, return_labels=True)
        label_counts = np.bincount(labels)
        largest_label = int(label_counts.argmax())
        origin_labels = labels[origin_node_idx]
        qa_metrics["network_n_nodes"] = road_graph_n_nodes
        qa_metrics["network_n_components"] = int(n_components)
        qa_metrics["share_origins_in_largest_component"] = float(
            (origin_labels == largest_label).mean()
        )
        qa_metrics["origin_snap_dist_mean_m"] = float(np.mean(origin_snap_dist))
        qa_metrics["origin_snap_dist_median_m"] = float(np.median(origin_snap_dist))
        qa_metrics["origin_snap_dist_p95_m"] = float(np.percentile(origin_snap_dist, 95))
        qa_metrics["origin_snap_dist_max_m"] = float(np.max(origin_snap_dist))
        qa_metrics["destination_snap_dist_mean_m"] = (
            float(np.mean(destination_offsets)) if len(destination_offsets) else 0.0
        )
        qa_metrics["destination_snap_dist_median_m"] = (
            float(np.median(destination_offsets)) if len(destination_offsets) else 0.0
        )
        qa_metrics["destination_snap_dist_p95_m"] = (
            float(np.percentile(destination_offsets, 95))
            if len(destination_offsets)
            else 0.0
        )
        qa_metrics["destination_snap_dist_max_m"] = (
            float(np.max(destination_offsets)) if len(destination_offsets) else 0.0
        )
        qa_metrics["share_unreachable_any_5km"] = float(
            (sums[max_radius_m].sum(axis=1) <= 0).mean()
        )

    target_df = build_target_table(
        base_df=base.drop(columns=["geometry"]),
        jobs_by_radius=jobs_by_radius,
        amenity_counts_by_cat_radius=amenity_counts,
        radii_km=radii_km,
        categories=categories,
        distance_weights=distance_weights,
    )

    out_dir = Path(args.output_root) / city / "target"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_df.to_parquet(out_dir / "target_table.parquet", index=False)
    target_df.to_csv(out_dir / "target_table.csv", index=False)

    splits_df = build_spatial_train_val_test_splits(
        target_df[["block_id", "x_m", "y_m"]],
        seed=args.split_seed,
        val_frac=args.split_val_frac,
        test_frac=args.split_test_frac,
        grid_bins_x=args.split_grid_bins_x,
        grid_bins_y=args.split_grid_bins_y,
    )
    splits_df.to_csv(out_dir / "spatial_splits.csv", index=False)

    jobs_last_col = f"jobs_{radii_km[-1]}km"
    amen_last_cols = [f"amen_{cat}_{radii_km[-1]}km" for cat in categories]
    qa_metrics["share_unreachable_jobs_5km"] = float((jobs_by_radius[jobs_last_col] <= 0).mean())
    qa_metrics["share_unreachable_amenities_5km"] = float(
        (amenity_counts[amen_last_cols].sum(axis=1) <= 0).mean()
    )

    metadata = {
        "city": city,
        "state": state,
        "distance_engine": args.distance_engine,
        "radii_km": radii_km,
        "distance_weights": distance_weights,
        "n_blocks": int(len(target_df)),
        "n_destination_blocks_buffered": int(len(destination_blocks)),
        "n_source_job_blocks": int(len(source_job_block_ids)),
        "n_source_job_blocks_outside_city": int(len(source_job_block_ids - city_block_ids)),
        "n_destination_job_blocks": int(len(jobs_dest)),
        "n_amenities_used": int(len(amenities)),
        "amenities_by_source_layer": amenities["source_layer"].value_counts().to_dict(),
        "amenities_by_category": amenities["category"].value_counts().to_dict(),
        "study_area_buffer_km": max(radii_km),
        "lehd_scope": lehd_scope,
        "wac_file": str(wac_path),
        "osm_pbf": str(pbf_path),
        "spatial_split": {
            "seed": int(args.split_seed),
            "val_frac": float(args.split_val_frac),
            "test_frac": float(args.split_test_frac),
            "grid_bins_x": int(args.split_grid_bins_x),
            "grid_bins_y": int(args.split_grid_bins_y),
            "counts": splits_df["split"].value_counts().to_dict(),
            "file": str(out_dir / "spatial_splits.csv"),
        },
        "qa": qa_metrics,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

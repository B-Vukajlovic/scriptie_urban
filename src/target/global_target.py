"""Build cross-city comparable accessibility targets from raw reachability columns."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


GLOBAL_TARGET_VIEWS = ("global_minmax", "global_log_minmax")


def minmax(values: pd.Series) -> pd.Series:
    """Min-max scale a numeric series, returning zeros for constant values."""
    values = values.astype(float)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - lo) / (hi - lo)


def infer_target_columns(columns: list[str]) -> tuple[list[int], list[str]]:
    """Infer available reachability radii and amenity categories from target columns."""
    radii: set[int] = set()
    categories: set[str] = set()
    for column in columns:
        job_match = re.match(r"^jobs_(\d+)km$", column)
        if job_match:
            radii.add(int(job_match.group(1)))
            continue

        amenity_match = re.match(r"^amen_(.+)_(\d+)km$", column)
        if amenity_match:
            categories.add(amenity_match.group(1))
            radii.add(int(amenity_match.group(2)))

    if not radii:
        raise ValueError("No jobs_{radius}km columns found for global target construction.")
    if not categories:
        raise ValueError("No amenity count columns found for global target construction.")
    return sorted(radii), sorted(categories)


def load_distance_weights(
    cities: list[str],
    interim_root: str | Path,
    radii_km: list[int],
) -> dict[int, float]:
    """Load and validate target distance weights from city target metadata."""
    weights_by_city: list[dict[int, float]] = []
    for city in cities:
        metadata_path = Path(interim_root) / city / "target" / "metadata.json"
        if not metadata_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        raw_weights = metadata.get("distance_weights")
        if isinstance(raw_weights, dict):
            weights_by_city.append(
                {int(radius): float(weight) for radius, weight in raw_weights.items()}
            )

    if weights_by_city:
        first = weights_by_city[0]
        for weights in weights_by_city[1:]:
            if weights != first:
                raise ValueError("Target distance weights differ across cities.")
        return {radius: first[radius] for radius in radii_km}

    equal_weight = 1.0 / len(radii_km)
    return {radius: equal_weight for radius in radii_km}


def load_multicity_target_components(
    cities: list[str],
    interim_root: str | Path,
) -> pd.DataFrame:
    """Load target tables and add city/node identifiers for global target construction."""
    frames: list[pd.DataFrame] = []
    for city in cities:
        target_path = Path(interim_root) / city / "target" / "target_table.parquet"
        if not target_path.exists():
            raise FileNotFoundError(f"Target table not found: {target_path}")
        target = pd.read_parquet(target_path)
        target["block_id"] = target["block_id"].astype(str)
        target.insert(0, "city", city)
        target.insert(0, "node_id", city + ":" + target["block_id"].astype(str))
        frames.append(target)
    return pd.concat(frames, ignore_index=True)


def build_global_target_columns(
    target_components: pd.DataFrame,
    cities: list[str],
    interim_root: str | Path,
    target_view: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build global E/A/Y columns from raw job and amenity reachability values."""
    columns = [str(col) for col in target_components.columns]
    radii_km, categories = infer_target_columns(columns)
    distance_weights = load_distance_weights(cities, interim_root, radii_km)

    out = target_components[["node_id", "city", "block_id"]].copy()
    work = target_components.copy()
    transform = "identity"
    if target_view == "global_log_minmax":
        transform = "log1p"
    elif target_view != "global_minmax":
        raise ValueError(f"Unsupported global target view: {target_view}")

    e = np.zeros(len(out), dtype=float)
    a = np.zeros(len(out), dtype=float)
    suffix = target_view
    for radius in radii_km:
        job_col = f"jobs_{radius}km"
        if job_col not in work.columns:
            raise KeyError(f"Missing required target component: {job_col}")
        job_values = work[job_col].astype(float)
        if transform == "log1p":
            job_values = np.log1p(job_values.clip(lower=0.0))
        job_norm = minmax(job_values)
        out[f"jobs_norm_{suffix}_{radius}km"] = job_norm

        category_norms: list[pd.Series] = []
        for category in categories:
            amenity_col = f"amen_{category}_{radius}km"
            if amenity_col not in work.columns:
                raise KeyError(f"Missing required target component: {amenity_col}")
            amenity_values = work[amenity_col].astype(float)
            if transform == "log1p":
                amenity_values = np.log1p(amenity_values.clip(lower=0.0))
            norm = minmax(amenity_values)
            out[f"amen_{category}_norm_{suffix}_{radius}km"] = norm
            category_norms.append(norm)

        amenity_mean_col = f"amenity_norm_mean_{suffix}_{radius}km"
        out[amenity_mean_col] = pd.concat(category_norms, axis=1).mean(axis=1)
        weight = float(distance_weights[radius])
        e += job_norm.to_numpy(dtype=float) * weight
        a += out[amenity_mean_col].to_numpy(dtype=float) * weight

    out[f"E_{suffix}"] = e
    out[f"A_{suffix}"] = a
    out[f"Y_{suffix}"] = 0.5 * e + 0.5 * a
    metadata: dict[str, Any] = {
        "target_view": target_view,
        "target_column": f"Y_{suffix}",
        "employment_column": f"E_{suffix}",
        "amenity_column": f"A_{suffix}",
        "target_normalization_scope": "selected_cities_global",
        "target_component_transform": transform,
        "radii_km": radii_km,
        "categories": categories,
        "distance_weights": distance_weights,
    }
    return out, metadata


def target_value_frame(global_targets: pd.DataFrame, target_view: str) -> pd.DataFrame:
    """Return node_id plus the Y column for one global target view."""
    column = f"Y_{target_view}"
    if column not in global_targets.columns:
        raise KeyError(f"Global target frame is missing {column}.")
    return global_targets[["node_id", column]].rename(columns={column: "target_value"})

"""Compute jobs reachability by radius for each origin block."""

from __future__ import annotations

from scipy.spatial import KDTree
import numpy as np
import pandas as pd


def jobs_reachability_by_radius(
	origins_xy: np.ndarray,
	destinations_xy: np.ndarray,
	destination_jobs: np.ndarray,
	radii_m: list[float],
) -> pd.DataFrame:
	"""Return reachable jobs for each origin and radius.

	Parameters
	----------
	origins_xy:
		Array of shape (n_origins, 2) in projected meters.
	destinations_xy:
		Array of shape (n_destinations, 2) in projected meters.
	destination_jobs:
		Array of jobs counts aligned with destinations.
	radii_m:
		Search radii in meters.
	"""
	if len(destinations_xy) == 0:
		return pd.DataFrame(
			{
				f"jobs_{int(r/1000)}km": np.zeros(len(origins_xy), dtype=float)
				for r in radii_m
			}
		)

	tree = KDTree(destinations_xy)
	destination_jobs = destination_jobs.astype(float)
	data: dict[str, np.ndarray] = {}

	for r in radii_m:
		neighbors = tree.query_ball_point(origins_xy, r=float(r))
		sums = np.array(
			[destination_jobs[idx].sum() if len(idx) else 0.0 for idx in neighbors],
			dtype=float,
		)
		data[f"jobs_{int(r/1000)}km"] = sums

	return pd.DataFrame(data)

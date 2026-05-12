"""Assemble accessibility target components from jobs and amenity reachability."""

from __future__ import annotations

import pandas as pd

from src.target.normalization import minmax_series, weighted_sum


def build_target_table(
	base_df: pd.DataFrame,
	jobs_by_radius: pd.DataFrame,
	amenity_counts_by_cat_radius: pd.DataFrame,
	radii_km: list[int],
	categories: list[str],
	distance_weights: dict[int, float],
) -> pd.DataFrame:
	"""Build final target components E, A, and Y from raw reachability inputs."""
	df = pd.concat([base_df.reset_index(drop=True), jobs_by_radius.reset_index(drop=True)], axis=1)
	df = pd.concat([df, amenity_counts_by_cat_radius.reset_index(drop=True)], axis=1)

	job_norm_cols: list[str] = []
	amen_radius_cols: list[str] = []

	for r in radii_km:
		j_col = f"jobs_{r}km"
		j_norm_col = f"jobs_norm_{r}km"
		df[j_norm_col] = minmax_series(df[j_col])
		job_norm_cols.append(j_norm_col)

		per_category_norm_cols: list[str] = []
		for cat in categories:
			c_col = f"amen_{cat}_{r}km"
			c_norm_col = f"amen_{cat}_norm_{r}km"
			df[c_norm_col] = minmax_series(df[c_col])
			per_category_norm_cols.append(c_norm_col)

		m_col = f"amenity_norm_mean_{r}km"
		df[m_col] = df[per_category_norm_cols].mean(axis=1)
		amen_radius_cols.append(m_col)

	e_weights = {f"jobs_norm_{r}km": distance_weights[r] for r in radii_km}
	a_weights = {f"amenity_norm_mean_{r}km": distance_weights[r] for r in radii_km}
	df["E"] = weighted_sum(df, e_weights)
	df["A"] = weighted_sum(df, a_weights)

	df["Y"] = 0.5 * df["E"] + 0.5 * df["A"]
	df["Y_60_40"] = 0.6 * df["E"] + 0.4 * df["A"]
	df["Y_40_60"] = 0.4 * df["E"] + 0.6 * df["A"]

	selected_cols = ["block_id", "x_m", "y_m", "lon", "lat", "E", "A", "Y", "Y_60_40", "Y_40_60"]
	selected_cols += [f"jobs_{r}km" for r in radii_km]
	selected_cols += [f"jobs_norm_{r}km" for r in radii_km]
	selected_cols += [f"amenity_norm_mean_{r}km" for r in radii_km]
	for cat in categories:
		selected_cols += [f"amen_{cat}_{r}km" for r in radii_km]

	return df[selected_cols].copy()

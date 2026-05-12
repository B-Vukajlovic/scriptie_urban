"""Spatially aware split utilities."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


def _safe_qcut(series: pd.Series, n_bins: int) -> pd.Series:
	"""Quantile-bin a numeric series with duplicate-safe fallback."""
	if n_bins <= 1:
		return pd.Series(np.zeros(len(series), dtype=int), index=series.index)

	# duplicates='drop' avoids failures when values have low variance.
	bins = pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
	if bins.isna().all():
		return pd.Series(np.zeros(len(series), dtype=int), index=series.index)
	return bins.fillna(0).astype(int)


def _pick_seed_cell(cells: np.ndarray, rng: np.random.Generator) -> int:
	"""Pick a reproducible start cell from the remaining occupied cells."""
	return int(cells[rng.integers(0, len(cells))])


def _grow_contiguous_region(
	available_cells: set[int],
	size_map: dict[int, int],
	x_lookup: dict[int, int],
	y_lookup: dict[int, int],
	target_size: int,
	rng: np.random.Generator,
) -> set[int]:
	"""Grow a contiguous set of occupied grid cells until the target size is reached."""
	if not available_cells or target_size <= 0:
		return set()

	seed = _pick_seed_cell(np.array(sorted(available_cells), dtype=int), rng)
	selected: set[int] = set()
	queue: deque[int] = deque([seed])
	running = 0

	while queue and running < target_size:
		cell = queue.popleft()
		if cell not in available_cells or cell in selected:
			continue
		selected.add(cell)
		running += int(size_map[cell])

		cx = x_lookup[cell]
		cy = y_lookup[cell]
		neighbors = [
			other
			for other in available_cells
			if other not in selected and abs(x_lookup[other] - cx) + abs(y_lookup[other] - cy) == 1
		]
		rng.shuffle(neighbors)
		queue.extend(neighbors)

	if running < target_size:
		remaining = [cell for cell in sorted(available_cells) if cell not in selected]
		for cell in remaining:
			selected.add(cell)
			running += int(size_map[cell])
			if running >= target_size:
				break

	return selected


def build_spatial_train_val_test_splits(
	df: pd.DataFrame,
	seed: int = 42,
	val_frac: float = 0.15,
	test_frac: float = 0.15,
	grid_bins_x: int = 8,
	grid_bins_y: int = 8,
) -> pd.DataFrame:
	"""Create reproducible contiguous spatial splits from block coordinates.

	The procedure builds coarse spatial cells from x/y quantile bins, then grows
	contiguous holdout regions cell-by-cell so validation and test sets are made
	up of coherent spatial subareas rather than scattered individual cells.
	"""
	required = {"block_id", "x_m", "y_m"}
	missing = required - set(df.columns)
	if missing:
		raise KeyError(f"Missing required columns for spatial splits: {sorted(missing)}")

	if not (0.0 <= val_frac < 1.0 and 0.0 <= test_frac < 1.0 and (val_frac + test_frac) < 1.0):
		raise ValueError("val_frac and test_frac must be in [0,1) and sum to < 1")

	work = df[["block_id", "x_m", "y_m"]].copy().reset_index(drop=True)
	x_bin = _safe_qcut(work["x_m"], grid_bins_x)
	y_bin = _safe_qcut(work["y_m"], grid_bins_y)

	max_y_bin = int(y_bin.max()) + 1
	work["spatial_cell"] = x_bin * max_y_bin + y_bin

	cell_sizes = work.groupby("spatial_cell", as_index=False).size().rename(columns={"size": "n"})
	cell_coords = work[["spatial_cell"]].copy()
	cell_coords["x_bin"] = x_bin
	cell_coords["y_bin"] = y_bin
	cell_coords = cell_coords.drop_duplicates("spatial_cell").reset_index(drop=True)

	rng = np.random.default_rng(seed)

	n_total = len(work)
	n_target_test = int(round(test_frac * n_total))
	n_target_val = int(round(val_frac * n_total))

	size_map = dict(zip(cell_sizes["spatial_cell"], cell_sizes["n"]))
	x_lookup = dict(zip(cell_coords["spatial_cell"], cell_coords["x_bin"]))
	y_lookup = dict(zip(cell_coords["spatial_cell"], cell_coords["y_bin"]))
	available_cells = set(int(cell) for cell in cell_sizes["spatial_cell"])

	test_cells = _grow_contiguous_region(
		available_cells=available_cells,
		size_map=size_map,
		x_lookup=x_lookup,
		y_lookup=y_lookup,
		target_size=n_target_test,
		rng=rng,
	)
	available_cells -= test_cells

	val_cells = _grow_contiguous_region(
		available_cells=available_cells,
		size_map=size_map,
		x_lookup=x_lookup,
		y_lookup=y_lookup,
		target_size=n_target_val,
		rng=rng,
	)

	test_set = set(test_cells)
	val_set = set(val_cells)

	split = np.where(
		work["spatial_cell"].isin(test_set),
		"test",
		np.where(work["spatial_cell"].isin(val_set), "val", "train"),
	)

	out = work[["block_id", "spatial_cell"]].copy()
	out["split"] = split
	return out

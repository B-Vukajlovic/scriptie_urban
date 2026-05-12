"""Normalization and weighted-combination utilities for target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def minmax_series(values: pd.Series) -> pd.Series:
	"""Min-max scale a numeric series; returns zeros when constant."""
	v = values.astype(float)
	lo = float(v.min())
	hi = float(v.max())
	if hi <= lo:
		return pd.Series(np.zeros(len(v), dtype=float), index=v.index)
	return (v - lo) / (hi - lo)


def weighted_sum(df: pd.DataFrame, column_weights: dict[str, float]) -> pd.Series:
	"""Compute weighted sum over selected columns."""
	missing = [c for c in column_weights if c not in df.columns]
	if missing:
		raise KeyError(f"Missing columns for weighted sum: {missing}")

	result = np.zeros(len(df), dtype=float)
	for col, w in column_weights.items():
		result += df[col].astype(float).to_numpy() * float(w)
	return pd.Series(result, index=df.index)

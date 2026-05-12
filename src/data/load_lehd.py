"""Load LEHD/LODES workplace area characteristics for jobs accessibility."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


STATE_FIPS = {
	"ar": "05",
	"ca": "06",
	"co": "08",
	"dc": "11",
	"ga": "13",
	"ma": "25",
	"ny": "36",
	"ok": "40",
	"tx": "48",
}


def _first_wac_file(folder: Path) -> Path | None:
	matches = sorted(folder.glob("wac_JT00_*.csv"))
	return matches[0] if matches else None


def _wac_matches_state(path: Path, state: str, min_share: float = 0.95) -> bool:
	"""Check that sampled WAC workplace GEOIDs match the expected state prefix."""
	expected = STATE_FIPS.get(state.strip().lower())
	if expected is None:
		return True

	sample = pd.read_csv(path, usecols=["w_geocode"], dtype={"w_geocode": "string"}, nrows=1000)
	if sample.empty:
		return False
	block_ids = sample["w_geocode"].astype(str)
	return bool(block_ids.str.startswith(expected).mean() >= min_share)


def find_wac_file(
	city: str,
	state: str,
	lehd_root: str | Path,
	prefer_state: bool = True,
) -> tuple[Path, str]:
	"""Return a WAC CSV path and source scope.

	Preferred layout:
	- statewide: ``{lehd_root}/state/{state}/wac_JT00_*.csv``
	- city extracts: ``{lehd_root}/city/{city}/wac_JT00_*.csv``

	The legacy ``{lehd_root}/{city}`` layout is still supported as a fallback.
	"""
	root = Path(lehd_root)
	city = city.strip().lower()
	state = state.strip().lower()

	candidates: list[tuple[str, Path]] = []
	if prefer_state:
		candidates.append(("state", root / "state" / state))
	candidates.append(("city", root / "city" / city))
	candidates.append(("legacy_city", root / city))
	if not prefer_state:
		candidates.append(("state", root / "state" / state))

	checked: list[str] = []
	rejected: list[str] = []
	for scope, folder in candidates:
		checked.append(str(folder))
		if not folder.exists():
			continue
		wac_file = _first_wac_file(folder)
		if wac_file is not None:
			if not _wac_matches_state(wac_file, state):
				rejected.append(f"{wac_file} has non-{state} block GEOIDs")
				continue
			return wac_file, scope

	raise FileNotFoundError(
		"No valid LEHD WAC file found. Checked: "
		+ ", ".join(checked)
		+ ("; rejected: " + "; ".join(rejected) if rejected else "")
	)


def load_jobs_by_block(wac_csv_path: str | Path) -> pd.DataFrame:
	"""Load job counts by census block from WAC data.

	Returns a DataFrame with columns:
	- block_id
	- jobs
	"""
	path = Path(wac_csv_path)
	if not path.exists():
		raise FileNotFoundError(f"LEHD WAC file not found: {path}")

	df = pd.read_csv(path, usecols=["w_geocode", "C000"], dtype={"w_geocode": "string"})
	raw = df.rename(columns={"w_geocode": "block_id", "C000": "jobs"}).copy()
	raw["block_id"] = raw["block_id"].astype(str)
	raw["jobs"] = raw["jobs"].fillna(0).astype(float)

	out = raw.groupby("block_id", as_index=False).agg(jobs=("jobs", "sum"))
	return out.reset_index(drop=True)

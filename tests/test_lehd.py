from pathlib import Path

import pandas as pd
import pytest

from src.data.load_lehd import find_wac_file, load_jobs_by_block


def _write_wac(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "w_geocode": ["080310001001000", "080310001001000", "080310001001001"],
            "C000": [2, 3, 5],
        }
    ).to_csv(path, index=False)


def test_find_wac_file_prefers_state_layout(tmp_path: Path) -> None:
    state_file = tmp_path / "state" / "co" / "wac_JT00_state.csv"
    city_file = tmp_path / "city" / "denver" / "wac_JT00_city.csv"
    _write_wac(state_file)
    _write_wac(city_file)

    path, scope = find_wac_file("denver", "co", tmp_path)

    assert path == state_file
    assert scope == "state"


def test_find_wac_file_can_prefer_city_layout(tmp_path: Path) -> None:
    state_file = tmp_path / "state" / "co" / "wac_JT00_state.csv"
    city_file = tmp_path / "city" / "denver" / "wac_JT00_city.csv"
    _write_wac(state_file)
    _write_wac(city_file)

    path, scope = find_wac_file("denver", "co", tmp_path, prefer_state=False)

    assert path == city_file
    assert scope == "city"


def test_find_wac_file_keeps_legacy_city_fallback(tmp_path: Path) -> None:
    legacy_file = tmp_path / "denver" / "wac_JT00_legacy.csv"
    _write_wac(legacy_file)

    path, scope = find_wac_file("denver", "co", tmp_path)

    assert path == legacy_file
    assert scope == "legacy_city"


def test_find_wac_file_rejects_wrong_state_prefix_and_falls_back(
    tmp_path: Path,
) -> None:
    wrong_state_file = tmp_path / "state" / "ar" / "wac_JT00_wrong.csv"
    city_file = tmp_path / "city" / "little_rock" / "wac_JT00_city.csv"
    _write_wac(wrong_state_file)
    city_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"w_geocode": ["051190001001000"], "C000": [10]}).to_csv(
        city_file,
        index=False,
    )

    path, scope = find_wac_file("little_rock", "ar", tmp_path)

    assert path == city_file
    assert scope == "city"


def test_find_wac_file_reports_checked_locations(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="state/co"):
        find_wac_file("denver", "co", tmp_path)


def test_find_wac_file_reports_wrong_state_prefix_rejections(tmp_path: Path) -> None:
    wrong_state_file = tmp_path / "state" / "ar" / "wac_JT00_wrong.csv"
    _write_wac(wrong_state_file)

    with pytest.raises(FileNotFoundError, match="non-ar block GEOIDs"):
        find_wac_file("little_rock", "ar", tmp_path)


def test_load_jobs_by_block_aggregates_duplicate_work_blocks(tmp_path: Path) -> None:
    wac_file = tmp_path / "wac_JT00_test.csv"
    _write_wac(wac_file)

    jobs = load_jobs_by_block(wac_file)

    assert jobs.to_dict(orient="records") == [
        {"block_id": "080310001001000", "jobs": 5.0},
        {"block_id": "080310001001001", "jobs": 5.0},
    ]

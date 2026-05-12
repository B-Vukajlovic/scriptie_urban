from __future__ import annotations

import zipfile

import pandas as pd

from src.data.load_acs import ACS_FEATURE_COLUMNS, build_block_acs_features


def _write_csv(path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_table_zip(root, table: str, rows: list[dict[str, object]], labels: dict[str, str]) -> None:
    path = root / f"ACSDT5Y2020.{table}_test.zip"
    data_name = f"ACSDT5Y2020.{table}-Data.csv"
    meta_name = f"ACSDT5Y2020.{table}-Column-Metadata.csv"
    data = pd.DataFrame(rows)
    label_row = {col: labels.get(col, col) for col in data.columns}
    data_with_labels = pd.concat([pd.DataFrame([label_row]), data], ignore_index=True)
    metadata = pd.DataFrame(
        [{"Column Name": col, "Label": labels.get(col, col)} for col in data.columns]
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(data_name, data_with_labels.to_csv(index=False))
        zf.writestr(meta_name, metadata.to_csv(index=False))
        zf.writestr(f"ACSDT5Y2020.{table}-Table-Notes.txt", "")


def test_build_block_acs_features_derives_expected_shares(tmp_path) -> None:
    tract = "08031000100"
    _write_csv(
        tmp_path / "ACS_5YR_ESTIMATES_DEMOGRAPHIC_TRACT_test.csv",
        [
            {
                "GEOID": tract,
                "B03002EST1": 1000,
                "Shape__Area": 2_000_000,
                "B03002EST3_PCT": 50,
                "B03002EST4_PCT": 10,
                "B03002EST6_PCT": 5,
                "B03002EST12_PCT": 25,
                "B01001_UNDER17_PCT": 20,
                "B01001_ABOVE64_PCT": 15,
                "AGE_25TO34": 200,
                "AGE_35TO44": 200,
                "AGE_45TO64": 300,
                "AGE_65UP": 100,
                "BD_25TO34": 50,
                "GD_25TO34": 10,
                "BD_35TO44": 60,
                "GD_35TO44": 20,
                "BD_45TO64": 90,
                "GD_45TO64": 30,
                "BD_65UP": 20,
                "GD_65UP": 10,
            }
        ],
    )
    _write_csv(
        tmp_path / "ACS_5YR_ESTIMATES_SOCIOECONOMIC_TRACT_test.csv",
        [
            {
                "GEOID": tract,
                "B19013EST1": 75_000,
                "B17021EST2_PCT": 12.5,
                "B23001_UE_PCT": 4,
                "B08013_AVG_TTW": 28,
                "B08303_60PLUS_TTW_PCT": 8,
            }
        ],
    )
    _write_csv(
        tmp_path / "ACS_5YR_ESTIMATES_HOUSING_TRACT_test.csv",
        [{"GEOID": tract, "B25002EST3_PCT": 6, "B25058EST1": 1200}],
    )
    _write_table_zip(
        tmp_path,
        "B01002",
        [{"GEO_ID": f"1400000US{tract}", "NAME": "tract", "B01002_001E": 37}],
        {"GEO_ID": "Geography", "NAME": "Geographic Area Name"},
    )
    _write_table_zip(
        tmp_path,
        "B08201",
        [
            {
                "GEO_ID": f"1400000US{tract}",
                "NAME": "tract",
                "B08201_001E": 400,
                "B08201_002E": 80,
            }
        ],
        {"GEO_ID": "Geography", "NAME": "Geographic Area Name"},
    )
    _write_table_zip(
        tmp_path,
        "B08301",
        [
            {
                "GEO_ID": f"1400000US{tract}",
                "NAME": "tract",
                "B08301_001E": 500,
                "B08301_002E": 300,
                "B08301_010E": 100,
                "B08301_018E": 25,
                "B08301_019E": 50,
                "B08301_021E": 20,
            }
        ],
        {"GEO_ID": "Geography", "NAME": "Geographic Area Name"},
    )
    _write_table_zip(
        tmp_path,
        "B18101",
        [
            {
                "GEO_ID": f"1400000US{tract}",
                "NAME": "tract",
                "B18101_001E": 1000,
                "B18101_004E": 30,
                "B18101_007E": 20,
            }
        ],
        {
            "GEO_ID": "Geography",
            "NAME": "Geographic Area Name",
            "B18101_004E": "Estimate!!Total:!!Male:!!Under 5 years:!!With a disability",
            "B18101_007E": "Estimate!!Total:!!Male:!!5 to 17 years:!!With a disability",
        },
    )

    features, metadata = build_block_acs_features(pd.Series([f"{tract}1000"]), acs_root=tmp_path)

    assert features.columns.tolist() == ["block_id", *ACS_FEATURE_COLUMNS]
    row = features.iloc[0]
    assert row["acs_population_density_per_km2"] == 500
    assert row["acs_white_non_hispanic_share"] == 0.5
    assert row["acs_bachelor_or_higher_share"] == 290 / 800
    assert row["acs_zero_vehicle_household_share"] == 0.2
    assert row["acs_public_transit_commute_share"] == 0.2
    assert row["acs_car_commute_share"] == 0.6
    assert row["acs_disability_share"] == 0.05
    assert metadata["n_blocks_attached"] == 1
    assert metadata["n_blocks_missing_acs_imputed"] == 0


def test_build_block_acs_features_imputes_missing_tracts(tmp_path) -> None:
    test_build_block_acs_features_derives_expected_shares(tmp_path)

    features, metadata = build_block_acs_features(
        pd.Series(["080310001001000", "060879901001000"]),
        acs_root=tmp_path,
    )

    assert len(features) == 2
    assert features[ACS_FEATURE_COLUMNS].isna().sum().sum() == 0
    assert metadata["n_blocks_missing_acs_imputed"] == 1
    assert metadata["missing_acs_tracts_imputed"] == ["06087990100"]

"""Load leakage-safe ACS tract features for block-level modeling."""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


ACS_FEATURE_COLUMNS = [
    "acs_population",
    "acs_population_density_per_km2",
    "acs_median_household_income",
    "acs_poverty_share",
    "acs_unemployment_share",
    "acs_white_non_hispanic_share",
    "acs_black_share",
    "acs_hispanic_share",
    "acs_asian_share",
    "acs_age_under_18_share",
    "acs_age_65_plus_share",
    "acs_bachelor_or_higher_share",
    "acs_avg_commute_time_min",
    "acs_commute_60_plus_min_share",
    "acs_vacant_housing_share",
    "acs_median_contract_rent",
    "acs_median_age",
    "acs_zero_vehicle_household_share",
    "acs_public_transit_commute_share",
    "acs_car_commute_share",
    "acs_walk_commute_share",
    "acs_bike_commute_share",
    "acs_work_from_home_share",
    "acs_disability_share",
]

ACS_MISSING_SENTINELS = {
    -999_999_999,
    -888_888_888,
    -666_666_666,
    -555_555_555,
    -333_333_333,
    -222_222_222,
}


def _single_file(root: Path, pattern: str) -> Path:
    matches = sorted(path for path in root.rglob(pattern) if "__pycache__" not in path.parts)
    if not matches:
        raise FileNotFoundError(f"ACS file not found for pattern: {root / pattern}")
    if len(matches) > 1:
        raise ValueError(f"Expected one ACS file for pattern {pattern}, found {len(matches)}.")
    return matches[0]


def _read_esri_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"GEOID": "string"}, encoding="utf-8-sig")
    if "GEOID" not in df.columns:
        raise KeyError(f"ACS Esri CSV is missing GEOID: {path}")
    tract_geoid = df["GEOID"].astype("string").str.zfill(11).rename("tract_geoid")
    return pd.concat([df, tract_geoid], axis=1).copy()


def _read_census_zip(root: Path, table: str) -> pd.DataFrame:
    path = _single_file(root, f"ACSDT5Y2020.{table}_*.zip")
    with zipfile.ZipFile(path) as zf:
        data_members = [name for name in zf.namelist() if name.endswith("-Data.csv")]
        if len(data_members) != 1:
            raise ValueError(f"Expected one Data CSV in {path}, found {len(data_members)}.")
        with zf.open(data_members[0]) as f:
            df = pd.read_csv(f, dtype=str, skiprows=[1])
    if "GEO_ID" not in df.columns:
        raise KeyError(f"ACS Census ZIP is missing GEO_ID: {path}")
    df["tract_geoid"] = df["GEO_ID"].astype(str).str.extract(r"US(\d{11})", expand=False)
    df = df[df["tract_geoid"].notna()].copy()
    return df


def _read_census_metadata(root: Path, table: str) -> pd.DataFrame:
    path = _single_file(root, f"ACSDT5Y2020.{table}_*.zip")
    with zipfile.ZipFile(path) as zf:
        metadata_members = [name for name in zf.namelist() if "Column-Metadata" in name]
        if len(metadata_members) != 1:
            raise ValueError(
                f"Expected one Column-Metadata CSV in {path}, found {len(metadata_members)}."
            )
        with zf.open(metadata_members[0]) as f:
            return pd.read_csv(f, dtype=str)


def _num(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out.mask(out.isin(ACS_MISSING_SENTINELS))


def _pct(series: pd.Series) -> pd.Series:
    return _num(series) / 100.0


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return (numerator / denom).replace([np.inf, -np.inf], np.nan)


def _sum_existing(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    present = [col for col in columns if col in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.concat([_num(df[col]) for col in present], axis=1).sum(axis=1, min_count=1)


def _with_disability_columns(root: Path) -> list[str]:
    metadata = _read_census_metadata(root, "B18101")
    if not {"Column Name", "Label"}.issubset(metadata.columns):
        raise KeyError("B18101 metadata must contain 'Column Name' and 'Label'.")
    mask = (
        metadata["Column Name"].astype(str).str.endswith("E")
        & metadata["Label"].astype(str).str.contains("With a disability", case=False, na=False)
    )
    return metadata.loc[mask, "Column Name"].astype(str).tolist()


def _finalize(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    out = features[["tract_geoid", *ACS_FEATURE_COLUMNS]].copy()
    imputed_counts: dict[str, int] = {}
    for column in ACS_FEATURE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        imputed_counts[column] = int(out[column].isna().sum())
        fill_value = out[column].median(skipna=True)
        if pd.isna(fill_value):
            fill_value = 0.0
        out[column] = out[column].fillna(float(fill_value)).astype(float)

    return out, imputed_counts


def load_acs_tract_features(acs_root: str | Path = "data/raw/acs") -> tuple[pd.DataFrame, dict[str, object]]:
    """Load ACS 2020 tract predictors from the curated raw ACS folder."""
    root = Path(acs_root)
    demographic = _read_esri_csv(_single_file(root, "ACS_5YR_ESTIMATES_DEMOGRAPHIC_TRACT_*.csv"))
    housing = _read_esri_csv(_single_file(root, "ACS_5YR_ESTIMATES_HOUSING_TRACT_*.csv"))
    socioeconomic = _read_esri_csv(
        _single_file(root, "ACS_5YR_ESTIMATES_SOCIOECONOMIC_TRACT_*.csv")
    )
    median_age = _read_census_zip(root, "B01002")
    vehicles = _read_census_zip(root, "B08201")
    commute_mode = _read_census_zip(root, "B08301")
    disability = _read_census_zip(root, "B18101")

    features = demographic[["tract_geoid"]].copy()
    features["acs_population"] = _num(demographic["B03002EST1"])
    tract_area_km2 = _num(demographic["Shape__Area"]) / 1_000_000.0
    features["acs_population_density_per_km2"] = _safe_divide(
        features["acs_population"],
        tract_area_km2,
    )
    features["acs_white_non_hispanic_share"] = _pct(demographic["B03002EST3_PCT"])
    features["acs_black_share"] = _pct(demographic["B03002EST4_PCT"])
    features["acs_asian_share"] = _pct(demographic["B03002EST6_PCT"])
    features["acs_hispanic_share"] = _pct(demographic["B03002EST12_PCT"])
    features["acs_age_under_18_share"] = _pct(demographic["B01001_UNDER17_PCT"])
    features["acs_age_65_plus_share"] = _pct(demographic["B01001_ABOVE64_PCT"])

    bachelor_cols = [
        "BD_25TO34",
        "GD_25TO34",
        "BD_35TO44",
        "GD_35TO44",
        "BD_45TO64",
        "GD_45TO64",
        "BD_65UP",
        "GD_65UP",
    ]
    age_25_plus_cols = ["AGE_25TO34", "AGE_35TO44", "AGE_45TO64", "AGE_65UP"]
    features["acs_bachelor_or_higher_share"] = _safe_divide(
        _sum_existing(demographic, bachelor_cols),
        _sum_existing(demographic, age_25_plus_cols),
    )

    features = features.merge(
        socioeconomic[
            [
                "tract_geoid",
                "B19013EST1",
                "B17021EST2_PCT",
                "B23001_UE_PCT",
                "B08013_AVG_TTW",
                "B08303_60PLUS_TTW_PCT",
            ]
        ],
        on="tract_geoid",
        how="left",
    )
    features["acs_median_household_income"] = _num(features.pop("B19013EST1"))
    features["acs_poverty_share"] = _pct(features.pop("B17021EST2_PCT"))
    features["acs_unemployment_share"] = _pct(features.pop("B23001_UE_PCT"))
    features["acs_avg_commute_time_min"] = _num(features.pop("B08013_AVG_TTW"))
    features["acs_commute_60_plus_min_share"] = _pct(features.pop("B08303_60PLUS_TTW_PCT"))

    features = features.merge(
        housing[["tract_geoid", "B25002EST3_PCT", "B25058EST1"]],
        on="tract_geoid",
        how="left",
    )
    features["acs_vacant_housing_share"] = _pct(features.pop("B25002EST3_PCT"))
    features["acs_median_contract_rent"] = _num(features.pop("B25058EST1"))

    features = features.merge(
        median_age[["tract_geoid", "B01002_001E"]],
        on="tract_geoid",
        how="left",
    )
    features["acs_median_age"] = _num(features.pop("B01002_001E"))

    features = features.merge(
        vehicles[["tract_geoid", "B08201_001E", "B08201_002E"]],
        on="tract_geoid",
        how="left",
    )
    features["acs_zero_vehicle_household_share"] = _safe_divide(
        _num(features.pop("B08201_002E")),
        _num(features.pop("B08201_001E")),
    )

    features = features.merge(
        commute_mode[
            [
                "tract_geoid",
                "B08301_001E",
                "B08301_002E",
                "B08301_010E",
                "B08301_018E",
                "B08301_019E",
                "B08301_021E",
            ]
        ],
        on="tract_geoid",
        how="left",
    )
    commute_total = _num(features.pop("B08301_001E"))
    features["acs_car_commute_share"] = _safe_divide(_num(features.pop("B08301_002E")), commute_total)
    features["acs_public_transit_commute_share"] = _safe_divide(
        _num(features.pop("B08301_010E")),
        commute_total,
    )
    features["acs_bike_commute_share"] = _safe_divide(
        _num(features.pop("B08301_018E")),
        commute_total,
    )
    features["acs_walk_commute_share"] = _safe_divide(
        _num(features.pop("B08301_019E")),
        commute_total,
    )
    features["acs_work_from_home_share"] = _safe_divide(
        _num(features.pop("B08301_021E")),
        commute_total,
    )

    disability_cols = _with_disability_columns(root)
    features = features.merge(
        disability[["tract_geoid", "B18101_001E", *disability_cols]],
        on="tract_geoid",
        how="left",
    )
    features["acs_disability_share"] = _safe_divide(
        _sum_existing(features, disability_cols),
        _num(features["B18101_001E"]),
    )
    features = features.drop(columns=["B18101_001E", *disability_cols])

    features, imputed_counts = _finalize(features)
    metadata = {
        "source": "ACS 2016-2020 5-year tract data",
        "geography": "tract",
        "n_tracts": int(len(features)),
        "feature_columns": ACS_FEATURE_COLUMNS,
        "imputed_missing_values": imputed_counts,
        "files": {
            "demographic": str(_single_file(root, "ACS_5YR_ESTIMATES_DEMOGRAPHIC_TRACT_*.csv")),
            "housing": str(_single_file(root, "ACS_5YR_ESTIMATES_HOUSING_TRACT_*.csv")),
            "socioeconomic": str(
                _single_file(root, "ACS_5YR_ESTIMATES_SOCIOECONOMIC_TRACT_*.csv")
            ),
            "median_age": str(_single_file(root, "ACSDT5Y2020.B01002_*.zip")),
            "vehicles": str(_single_file(root, "ACSDT5Y2020.B08201_*.zip")),
            "commute_mode": str(_single_file(root, "ACSDT5Y2020.B08301_*.zip")),
            "disability": str(_single_file(root, "ACSDT5Y2020.B18101_*.zip")),
        },
    }
    return features, metadata


def build_block_acs_features(
    block_ids: pd.Series,
    acs_root: str | Path = "data/raw/acs",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach tract-level ACS features to block IDs by 11-digit tract GEOID."""
    acs_features, metadata = load_acs_tract_features(acs_root)
    blocks = pd.DataFrame({"block_id": block_ids.astype(str)})
    blocks["tract_geoid"] = blocks["block_id"].str[:11]
    out = blocks.merge(acs_features, on="tract_geoid", how="left")
    missing_blocks = int(out[ACS_FEATURE_COLUMNS].isna().any(axis=1).sum())
    if missing_blocks:
        missing_tracts = sorted(out.loc[out[ACS_FEATURE_COLUMNS].isna().any(axis=1), "tract_geoid"].unique())
        fill_values = acs_features[ACS_FEATURE_COLUMNS].median(numeric_only=True)
        out[ACS_FEATURE_COLUMNS] = out[ACS_FEATURE_COLUMNS].fillna(fill_values)
    else:
        missing_tracts = []
    out = out.drop(columns=["tract_geoid"])
    metadata = dict(metadata)
    metadata["n_blocks_attached"] = int(len(out))
    metadata["n_unique_block_tracts"] = int(blocks["tract_geoid"].nunique())
    metadata["n_blocks_missing_acs_imputed"] = missing_blocks
    metadata["missing_acs_tracts_imputed"] = missing_tracts
    return out, metadata

import geopandas as gpd
from shapely.geometry import Point, Polygon

from src.preprocessing.blocks import select_blocks_for_place, standardize_block_columns


def _raw_blocks() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "GEOID20": ["inside", "outside"],
            "STATEFP20": ["01", "01"],
            "COUNTYFP20": ["001", "001"],
            "TRACTCE20": ["000100", "000100"],
            "BLOCKCE20": ["1000", "1001"],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
            ],
        },
        crs="EPSG:4326",
    )


def test_select_blocks_for_place_uses_representative_point_membership():
    blocks = _raw_blocks()
    place = gpd.GeoDataFrame(
        {"geometry": [Point(0.5, 0.5).buffer(0.75)]},
        crs="EPSG:4326",
    )

    selected = select_blocks_for_place(blocks, place, mode="representative_point")

    assert selected["GEOID20"].tolist() == ["inside"]


def test_standardize_block_columns_renames_tiger_identifiers():
    standardized = standardize_block_columns(_raw_blocks())

    assert standardized.columns.tolist() == [
        "block_id",
        "state_fips",
        "county_fips",
        "tract_code",
        "block_code",
        "geometry",
    ]
    assert standardized["block_id"].tolist() == ["inside", "outside"]

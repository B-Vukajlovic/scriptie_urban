import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.target.build_target import build_target_table
from src.target.graph_reachability import batched_reachability_sums


def test_build_target_table_combines_jobs_and_amenities():
    base = pd.DataFrame(
        {
            "block_id": ["a", "b"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 1.0],
            "lon": [0.0, 0.0],
            "lat": [0.0, 0.0],
        }
    )
    jobs = pd.DataFrame({"jobs_1km": [0.0, 10.0], "jobs_2km": [0.0, 20.0]})
    amenities = pd.DataFrame(
        {
            "amen_food_1km": [0.0, 2.0],
            "amen_food_2km": [0.0, 4.0],
            "amen_health_1km": [0.0, 6.0],
            "amen_health_2km": [0.0, 8.0],
        }
    )

    target = build_target_table(
        base_df=base,
        jobs_by_radius=jobs,
        amenity_counts_by_cat_radius=amenities,
        radii_km=[1, 2],
        categories=["food", "health"],
        distance_weights={1: 0.75, 2: 0.25},
    )

    assert target["Y"].tolist() == [0.0, 1.0]
    assert target["E"].tolist() == [0.0, 1.0]
    assert target["A"].tolist() == [0.0, 1.0]


def test_batched_reachability_sums_respects_source_offsets():
    graph = csr_matrix(
        np.array(
            [
                [0.0, 100.0, 0.0],
                [100.0, 0.0, 100.0],
                [0.0, 100.0, 0.0],
            ]
        )
    )
    destination_matrix = np.array([[0.0], [0.0], [5.0]])

    sums = batched_reachability_sums(
        graph=graph,
        destination_matrix=destination_matrix,
        radii_m=[150.0, 300.0],
        source_indices=np.array([0]),
        source_offsets_m=np.array([75.0]),
    )

    assert sums[150.0][0, 0] == 0.0
    assert sums[300.0][0, 0] == 5.0

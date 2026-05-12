import pandas as pd
import pytest

from src.modeling.dataset import assemble_model_dataset, validate_model_feature_columns


def test_validate_model_feature_columns_rejects_leakage_columns():
    with pytest.raises(ValueError):
        validate_model_feature_columns(["pt_stop_count", "Y"])

    with pytest.raises(ValueError):
        validate_model_feature_columns(["be_compactness", "jobs_1km"])

    with pytest.raises(ValueError):
        validate_model_feature_columns(["pt_route_count", "lon"])


def test_assemble_model_dataset_joins_features_target_and_splits():
    features = pd.DataFrame(
        {
            "block_id": ["a", "b", "c"],
            "pt_stop_count": [1, 0, 2],
            "be_compactness": [0.5, 0.7, 0.9],
            "acs_poverty_share": [0.1, 0.2, 0.3],
        }
    )
    target = pd.DataFrame({"block_id": ["a", "b", "c"], "Y": [0.1, 0.2, 0.3]})
    splits = pd.DataFrame(
        {
            "block_id": ["a", "b", "c"],
            "split": ["train", "val", "test"],
            "spatial_cell": [1, 2, 3],
        }
    )

    dataset = assemble_model_dataset(
        feature_df=features,
        target_df=target,
        splits_df=splits,
        feature_columns=["pt_stop_count", "be_compactness", "acs_poverty_share"],
    )

    assert dataset.columns.tolist() == [
        "block_id",
        "pt_stop_count",
        "be_compactness",
        "acs_poverty_share",
        "Y",
        "split",
        "spatial_cell",
    ]
    assert dataset["split"].tolist() == ["train", "val", "test"]

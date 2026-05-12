import pandas as pd

from src.evaluation.split_diagnostics import (
    build_feature_shift_table,
    build_split_diagnostics,
)


def test_build_split_diagnostics_reports_target_and_feature_shift():
    dataset = pd.DataFrame(
        {
            "block_id": ["a", "b", "c", "d"],
            "split": ["train", "train", "val", "test"],
            "Y": [1.0, 3.0, 5.0, 7.0],
            "pt_stop_count": [0.0, 2.0, 4.0, 6.0],
            "be_compactness": [1.0, 1.0, 2.0, 3.0],
        }
    )

    diagnostics = build_split_diagnostics(
        dataset,
        feature_columns=["pt_stop_count", "be_compactness"],
    )

    assert diagnostics["n_rows"] == 4
    assert diagnostics["split_counts"] == {"train": 2, "val": 1, "test": 1}
    assert "target_summary_by_split" in diagnostics
    assert "train_mean_baseline_metrics" in diagnostics
    assert diagnostics["top_feature_shifts_vs_train"]


def test_build_feature_shift_table_skips_train_split():
    dataset = pd.DataFrame(
        {
            "split": ["train", "train", "test"],
            "Y": [1.0, 2.0, 3.0],
            "pt_stop_count": [1.0, 2.0, 5.0],
        }
    )

    table = build_feature_shift_table(dataset, feature_columns=["pt_stop_count"])

    assert table["split"].tolist() == ["test"]
    assert table["feature"].tolist() == ["pt_stop_count"]
    assert table["abs_standardized_mean_difference"].iloc[0] > 0

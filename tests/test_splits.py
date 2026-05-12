import pandas as pd

from src.evaluation.spatial_splits import build_spatial_train_val_test_splits


def test_build_spatial_train_val_test_splits_keeps_all_blocks_once():
	df = pd.DataFrame(
		{
			"block_id": [str(i) for i in range(100)],
			"x_m": [i % 10 for i in range(100)],
			"y_m": [i // 10 for i in range(100)],
		}
	)

	splits = build_spatial_train_val_test_splits(
		df,
		seed=7,
		val_frac=0.2,
		test_frac=0.2,
		grid_bins_x=5,
		grid_bins_y=5,
	)

	assert len(splits) == len(df)
	assert set(splits["block_id"]) == set(df["block_id"])
	assert set(splits["split"]) == {"train", "val", "test"}
	assert splits["block_id"].is_unique

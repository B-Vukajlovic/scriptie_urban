import json
from pathlib import Path

import pytest

from src.modeling.dataset import validate_model_feature_columns


def _feature_metadata_paths() -> list[Path]:
    return sorted(Path("data/interim").glob("*/features/metadata.json"))


def _model_metadata_paths() -> list[Path]:
    return sorted(Path("data/interim").glob("*/modeling/metadata.json"))


def test_interim_feature_metadata_contains_no_target_ingredients() -> None:
    paths = _feature_metadata_paths()
    if not paths:
        pytest.skip("No interim feature metadata found.")

    for path in paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        feature_columns = metadata["feature_columns"]
        validate_model_feature_columns(feature_columns)


def test_interim_model_metadata_contains_no_target_ingredients_in_x() -> None:
    paths = _model_metadata_paths()
    if not paths:
        pytest.skip("No interim model metadata found.")

    for path in paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        feature_columns = metadata["feature_columns"]
        validate_model_feature_columns(feature_columns)

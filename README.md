# Urban Accessibility Thesis

This repository builds and evaluates a thesis pipeline for predicting
block-level distance-based opportunity accessibility across ten US cities.

The current final scope is documented in [METHOD_PIPELINE.md](METHOD_PIPELINE.md).

## Current Thesis Scope

- Target: `Y_global_log_minmax`
- Accessibility: road-network distance-based access to LEHD jobs and OSM
  amenities
- Predictors: block-level public transport and built-environment indicators
- Graph: Census-block adjacency
- Models: XGBoost, GCN, GraphSAGE
- Evaluation: pooled spatial train/validation/test holdout
- Explainability: SHAP for XGBoost, PyG GNNExplainer for graph models

## Active Workflow

The final script workflow is documented in [scripts/README.md](scripts/README.md).

Main stages:

```text
scripts/run_city_data_pipeline.py
scripts/build_global_targets.py
scripts/evaluate_multicity_xgboost.py
scripts/evaluate_pyg_gnn.py
scripts/explain_multicity_xgboost_shap.py
scripts/cluster_xgboost_shap_signatures.py
scripts/cluster_gnn_explainer_signatures.py
scripts/plot_block_only_sensitivity.py
```

## Data Layout

Raw inputs live under `data/raw/` and are ignored by git:

```text
acs/    ACS tract data, used mainly for exploratory/post-hoc context
gtfs/   GTFS feeds by city/agency
lehd/   LEHD workplace area characteristics
osm/    OSM PBF extracts
tiger/  Census TIGER/Line block geometries
```

Intermediate city artifacts are written to `data/interim/{city}/`.

## Outputs

Use `outputs/final/` for thesis-facing artifacts. Older experiments are
preserved under `outputs/archive/`.

## Tests

Run the focused method tests with:

```bash
.venv/bin/python -m pytest \
  tests/test_features.py \
  tests/test_target.py \
  tests/test_modeling_dataset.py \
  tests/test_gnn_models.py \
  tests/test_xgboost_feature_views.py
```

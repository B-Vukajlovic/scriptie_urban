# urban-accessibility-thesis

Pipeline for building block-level accessibility targets, leakage-safe spatial
features, and thesis-facing model evaluations across the selected US cities.

## Script Workflow

The script folder is intentionally split into three groups.

### City Data Preparation

```bash
python scripts/run_city_data_pipeline.py --cities denver
```

This runs, in order:

```text
build_city_backbone.py      TIGER blocks, centroids, adjacency, QC
build_city_target.py        reachable LEHD jobs + OSM amenities target
build_city_features.py      leakage-safe GTFS, OSM built environment, ACS features
build_city_model_dataset.py      joined supervised modeling table
diagnose_city_model_dataset.py   split/feature/target diagnostics
```

Run a specific stage when needed:

```bash
python scripts/run_city_data_pipeline.py --cities denver --stages features model_dataset
```

### Model Evaluation

Use the multi-city scripts for reportable model results:

```bash
python scripts/evaluate_multicity_xgboost.py --n-jobs 4
python scripts/evaluate_multicity_gnn.py --models gcn graphsage --evaluation-mode single_spatial_cv
```

`evaluate_multicity_xgboost.py` reports pooled single-spatial-split performance
and leave-one-city-out generalization. `evaluate_multicity_gnn.py` trains true
GCN/GraphSAGE regressors on the block graph.

### Explainability And Reports

```bash
python scripts/explain_multicity_xgboost_shap.py --n-jobs 4
python scripts/build_shap_report.py
python scripts/build_model_comparison_report.py
```

These create SHAP tables/figures and compact model-comparison outputs under
`outputs/`.

## Data Layout

Raw inputs live under `data/raw/`:

```text
acs/    ACS tract socioeconomic predictors
gtfs/   transit feeds by city/agency
lehd/   state-level LEHD WAC files
osm/    state OSM PBF extracts
tiger/  Census TIGER/Line geometries
```

Intermediate city artifacts are written to `data/interim/{city}/`.

## Current Modeling Target

The target table contains a normalized accessibility score `Y` built from
reachable employment opportunities and reachable amenities within network
distance radii. Predictor tables intentionally exclude target ingredients such
as reachable job totals and reachable amenity counts to avoid leakage.

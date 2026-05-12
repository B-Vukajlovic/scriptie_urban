# Scripts

This folder contains only entry-point scripts. Shared logic should live in
`src/`.

## Data Preparation

```text
run_city_data_pipeline.py   run the city data stages in order
build_city_backbone.py      build TIGER block geometry, centroids, adjacency
build_city_target.py        build reachable jobs/amenities target
build_global_targets.py     add cross-city comparable global target columns
build_city_features.py      build leakage-safe PT, BE, and ACS predictors
build_city_model_dataset.py join features and target into model tables
diagnose_city_model_dataset.py write split and feature diagnostics
```

## Modeling

```text
evaluate_multicity_xgboost.py  pooled XGBoost plus leave-one-city-out; defaults to city-z-scored features
tune_reduced_xgboost.py        small spatial-validation hyperparameter search for reduced XGBoost
evaluate_multicity_gnn.py      true GCN/GraphSAGE graph regressors
evaluate_pyg_gnn.py            official PyTorch Geometric GCN/GraphSAGE plus GNNExplainer
```

## Explainability And Reports

```text
explain_multicity_xgboost_shap.py  SHAP values for held-out XGBoost rows using the selected feature view/set
cluster_xgboost_shap_signatures.py cluster XGBoost SHAP vectors into explanation-based access zones
plot_reduced_xgboost_pdp.py        partial-dependence plots for the reduced XGBoost baseline
explain_multicity_gnn.py           GNNExplainer-style masks for reduced GCN/GraphSAGE models
build_shap_report.py               report-ready SHAP tables and figures
build_model_comparison_report.py   compact model comparison tables/report
```

## Recommended XGBoost Runs

```bash
.venv/bin/python scripts/evaluate_multicity_xgboost.py \
  --target-column Y_global_log_minmax \
  --target-view stored \
  --feature-views log1p \
  --feature-sets full \
  --n-jobs 4

.venv/bin/python scripts/evaluate_multicity_xgboost.py \
  --target-column Y_global_log_minmax \
  --target-view stored \
  --feature-views log1p \
  --feature-sets all_ablation \
  --output-prefix multicity_xgboost_global_log_feature_ablation \
  --n-jobs 4

.venv/bin/python scripts/evaluate_multicity_xgboost.py \
  --target-column Y_city_relative \
  --target-view stored \
  --feature-views city_zscore \
  --feature-sets full \
  --output-prefix multicity_xgboost_legacy_city_relative \
  --n-jobs 4
```

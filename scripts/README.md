# Scripts

This folder contains the entry points for the final thesis workflow. Shared
implementation logic lives in `src/`.

The current thesis scope is:

- target: `Y_global_log_minmax`
- predictors: block-level public transport and built-environment features
- graph: Census-block spatial adjacency
- evaluation: pooled spatial train/validation/test holdout
- models: XGBoost, GCN, GraphSAGE
- explanation: SHAP for XGBoost and PyG GNNExplainer for GCN/GraphSAGE

Older experiments are preserved in `scripts/archive/`.

## Data Preparation

```text
run_city_data_pipeline.py   run one city's data stages in order
build_city_backbone.py      build TIGER blocks, centroids, and adjacency
build_city_target.py        build road-network reachable jobs/amenities target
build_global_targets.py     add cross-city comparable target columns
build_city_features.py      build leakage-safe PT, BE, and ACS feature tables
build_city_model_dataset.py join features, target, and split labels
diagnose_city_model_dataset.py write split and feature diagnostics
```

## Modeling

```text
evaluate_multicity_xgboost.py  tabular XGBoost baseline
evaluate_multicity_gnn.py      custom PyTorch GCN/GraphSAGE regressors
evaluate_pyg_gnn.py            official PyG GCN/GraphSAGE and GNNExplainer
```

## Explainability And Figures

```text
explain_multicity_xgboost_shap.py  XGBoost SHAP values and summary tables
cluster_xgboost_shap_signatures.py cluster SHAP vectors into explanation profiles
cluster_gnn_explainer_signatures.py cluster PyG GNNExplainer feature masks
plot_block_only_sensitivity.py      final no-area/no-footprint sensitivity figures
```

## Final Thesis Commands

Expanded block-only PT+BE XGBoost:

```bash
.venv/bin/python scripts/evaluate_multicity_xgboost.py \
  --target-column Y_global_log_minmax \
  --target-view stored \
  --feature-views log1p \
  --feature-sets reduced \
  --reduced-feature-set data/interim/modeling/expanded_block_only_pt_be_feature_set.json \
  --output-prefix multicity_xgboost_expanded_block_only_pt_be \
  --n-jobs 4
```

Expanded block-only PT+BE graph models:

```bash
.venv/bin/python scripts/evaluate_pyg_gnn.py \
  --target-column Y_global_log_minmax \
  --feature-set reduced \
  --reduced-feature-set data/interim/modeling/expanded_block_only_pt_be_feature_set.json \
  --feature-view log1p \
  --models gcn graphsage \
  --output-prefix multicity_pyg_gnn_expanded_block_only_pt_be \
  --epochs 300 \
  --patience 35 \
  --hidden-dim 96 \
  --dropout 0.15 \
  --learning-rate 0.003 \
  --weight-decay 1e-4 \
  --torch-threads 4
```

XGBoost SHAP explanations:

```bash
.venv/bin/python scripts/explain_multicity_xgboost_shap.py \
  --target-column Y_global_log_minmax \
  --feature-sets reduced \
  --reduced-feature-set data/interim/modeling/expanded_block_only_pt_be_feature_set.json \
  --feature-views log1p \
  --output-prefix multicity_xgboost_shap_expanded_block_only_pt_be \
  --n-jobs 4
```

Official PyG GNNExplainer explanations:

```bash
.venv/bin/python scripts/evaluate_pyg_gnn.py \
  --target-column Y_global_log_minmax \
  --feature-set reduced \
  --reduced-feature-set data/interim/modeling/expanded_block_only_pt_be_feature_set.json \
  --feature-view log1p \
  --models gcn graphsage \
  --output-prefix multicity_pyg_gnn_expanded_block_only_pt_be \
  --epochs 300 \
  --patience 35 \
  --hidden-dim 96 \
  --dropout 0.15 \
  --learning-rate 0.003 \
  --weight-decay 1e-4 \
  --torch-threads 4 \
  --explain \
  --max-explain-nodes 100 \
  --explain-nodes-per-city 10
```

Cluster explanation signatures:

```bash
.venv/bin/python scripts/cluster_xgboost_shap_signatures.py \
  --shap-values outputs/tables/multicity_xgboost_shap_expanded_block_only_pt_be_values.parquet \
  --predictions outputs/tables/multicity_xgboost_shap_expanded_block_only_pt_be_predictions.csv \
  --output-prefix multicity_xgboost_shap_expanded_block_only_pt_be_clusters

.venv/bin/python scripts/cluster_gnn_explainer_signatures.py \
  --feature-masks outputs/tables/multicity_pyg_gnn_expanded_block_only_pt_be_gnnexplainer_feature_masks.csv \
  --nodes outputs/tables/multicity_pyg_gnn_expanded_block_only_pt_be_gnnexplainer_nodes.csv \
  --output-prefix multicity_pyg_gnn_expanded_block_only_pt_be_gnnexplainer_clusters
```

No-area/no-footprint sensitivity figures:

```bash
.venv/bin/python scripts/plot_block_only_sensitivity.py
```

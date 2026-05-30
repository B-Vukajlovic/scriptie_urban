# Method Pipeline

This file freezes the current thesis implementation so the method chapter can be
written against one stable workflow.

## Research Scope

The thesis predicts and explains block-level distance-based opportunity
accessibility across ten US cities.

The active research questions are:

1. How accurately can XGBoost, GraphSAGE, and GCN predict block-level
   distance-based opportunity accessibility?
2. Do graph-based models improve predictive performance compared with a tabular
   XGBoost baseline?
3. Which public-transport and built-environment characteristics are most
   strongly associated with predicted accessibility patterns?
4. What recurring explanation patterns of predicted accessibility can be
   identified across cities?

Cross-city transfer and ACS/equity analysis are preserved as exploratory
material, but they are not part of the final core scope.

## Unit Of Analysis

Each observation is a Census block. Blocks are represented as:

- one row in the modelling table;
- one node in the graph;
- one prediction target value.

## Target

The main target is `Y_global_log_minmax`.

It measures distance-based cumulative opportunity accessibility:

- reachable LEHD workplace jobs within 1-5 km;
- reachable OSM amenities within 1-5 km;
- reachability computed over the OSM road network;
- employment and amenity components combined into one score;
- log-transformed and globally min-max normalized across all cities.

This is not public-transport travel-time accessibility. Public transport enters
as predictor variables, not as the routing engine for the target.

Core target code:

```text
scripts/build_city_target.py
scripts/build_global_targets.py
src/target/build_target.py
src/target/graph_reachability.py
src/target/osm_network_graph.py
src/target/reachable_jobs.py
src/target/reachable_amenities.py
src/target/global_target.py
```

## Predictors

The final model comparison uses block-level public transport and
built-environment predictors.

Public transport features are derived from GTFS and include stops, stop density,
route count, weekday departures, peak/off-peak departures, headway, service
span, modal variety, and mode-specific bus/tram/metro/train indicators.

Built-environment features are derived from OSM and block geometry and include
street length density, intersection density, bikeable share, major-road share,
low-speed street share, land-use shares, compactness, block area, and building
footprint or morphology variables.

Main feature set:

```text
data/interim/modeling/expanded_block_only_pt_be_feature_set.json
```

Sensitivity feature set:

```text
data/interim/modeling/block_only_pt_be_no_area_footprint_feature_set.json
```

The sensitivity removes:

```text
be_block_area_m2
be_building_footprint_share
```

## Graph

The graph is a Census-block adjacency graph.

- Nodes: Census blocks.
- Edges: neighbouring blocks.
- Public transport is not encoded as graph edges.
- Public transport and built environment enter as node attributes.

This design tests whether neighbouring-block context improves prediction beyond
local block attributes.

Core graph code:

```text
scripts/build_city_backbone.py
src/preprocessing/adjacency.py
src/graph/adjacency.py
```

## Models

### XGBoost

XGBoost is the tabular baseline. It receives only the selected block-level
feature columns and does not receive the graph.

```text
scripts/evaluate_multicity_xgboost.py
```

### GCN

GCN receives the same node features plus the block adjacency graph. The PyG
implementation uses two `GCNConv` layers followed by a linear regression head.

```text
src/models/pyg_gnn.py
scripts/evaluate_pyg_gnn.py
```

### GraphSAGE

GraphSAGE receives the same node features plus the block adjacency graph. The
implementation uses mean neighbourhood aggregation over adjacent blocks. It is
used in a transductive node-regression setting: all node features and graph
positions are available, but validation and test labels are withheld from the
loss.

```text
src/models/pyg_gnn.py
scripts/evaluate_pyg_gnn.py
```

The custom PyTorch models in `src/models/gnn.py` and
`scripts/evaluate_multicity_gnn.py` are retained as validation/supporting
implementations.

## Evaluation

The primary evaluation is a pooled spatial holdout.

For each city:

1. block centroids are binned into an 8 x 8 quantile grid;
2. contiguous grid regions are assigned to validation and test sets;
3. remaining blocks are used for training.

The city-level train, validation, and test subsets are then pooled across all
cities.

This is a single spatial train/validation/test holdout, not repeated k-fold
cross-validation.

Core evaluation code:

```text
src/evaluation/spatial_splits.py
src/models/metrics.py
```

## Explainability

XGBoost is explained with SHAP:

```text
scripts/explain_multicity_xgboost_shap.py
scripts/cluster_xgboost_shap_signatures.py
```

GCN and GraphSAGE are explained with official PyTorch Geometric GNNExplainer:

```text
scripts/evaluate_pyg_gnn.py --explain
scripts/cluster_gnn_explainer_signatures.py
```

Explanation signatures are clustered across blocks to identify recurring
feature-attribution patterns.

## Final Artifacts

Use `outputs/final/` for thesis writing.

Important result families:

```text
multicity_xgboost_expanded_block_only_pt_be*
multicity_gnn_expanded_block_only_pt_be_spatial_cv*
multicity_pyg_gnn_expanded_block_only_pt_be*
multicity_xgboost_shap_expanded_block_only_pt_be*
block_only_pt_be_no_area_footprint*
```

Old experiments are preserved under:

```text
scripts/archive/
outputs/archive/legacy_outputs/
outputs/figures/archive/
```

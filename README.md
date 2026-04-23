# urban-accessibility-thesis

## Overview
This repository contains a placeholder scaffold for an urban accessibility thesis pipeline. The planned workflow covers data ingestion, preprocessing, feature engineering, target construction, model training (XGBoost and graph models), explainability, and evaluation across a pilot city and all cities.

## Repository Structure
- `configs/`: Configuration files for paths, cities, targets, features, models, and data splits.
- `data/`: Raw, interim, and processed datasets organized by source and scope.
- `notebooks/`: Sequential exploratory and modeling notebooks for each pipeline stage.
- `outputs/`: Generated figures, tables, metrics, and logs.
- `models/`: Serialized model artifacts grouped by model family.
- `scripts/`: Entry-point scripts for running pipelines, training models, and map generation.
- `src/`: Core source modules for data loading, preprocessing, features, targets, graph construction, modeling, explainability, evaluation, and visualization.
- `tests/`: Placeholder tests for key pipeline components.

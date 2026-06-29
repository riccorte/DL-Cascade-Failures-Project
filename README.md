# CascadeGPS — didactical Deep Learning notebooks

This folder is a self-contained teaching version of the supervised Deep Learning part of CascadeGPS/CIRCLES. It is intentionally written with dense PyTorch tensors rather than PyTorch Geometric, so batching, masking, message passing, attention, losses, backpropagation, validation, checkpointing, and evaluation remain visible.

## Notebook order

1. `00_dataset.ipynb` generates graph-disjoint synthetic train/validation/test data.
2. `01_models.ipynb` defines and tests an MLP, a dense GNN, and a hybrid GNN–Transformer.
3. `02_training.ipynb` trains all models with a multi-task loss, early stopping, and checkpointing.
4. `03_evaluation.ipynb` selects classification thresholds on validation data and evaluates models and heuristics on test data.
5. `04_results.ipynb` consolidates tables, figures, qualitative predictions, and a short interpretation.

Run the notebooks from this folder and in this order. The supplied archive already contains a completed smoke run, but every artifact can be regenerated.

`dl_models.py` mirrors the model definitions from `01_models.ipynb` and gives the later notebooks a stable import path; the model code remains fully visible in the notebook for study.

## Prediction tasks

For a graph state at time `t`, the models predict:

- the cumulative failed state of every node at `t+1`;
- the node load at `t+1`;
- the final failed fraction after the remaining cascade.

The synthetic simulator is deliberately simple. Its role is to provide a controlled data contract for learning PyTorch, not to claim physical realism.

## Quick setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## Relation to the larger project

This notebook folder is the transparent prototype. The larger `cascadegps_project` repository is the modularized continuation: simulator, models, training, and evaluation are moved into reusable Python packages and command-line scripts.

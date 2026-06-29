# Smoke-run result summary

- Best neural node-ranking model by PR-AUC: **gnn** (0.972).
- Best heuristic by PR-AUC: **failed_neighbor** (0.826).
- Best graph-level cascade-size predictor: **gnn** (MAE 0.121).
- Best next-load predictor: **hybrid** (RMSE 0.442).

These numbers validate the software pipeline, not a final scientific claim. The dataset is small, synthetic and based on one simplified family of cascade rules.
Stronger tests should use larger datasets, multiple random seeds, held-out graph families, stronger-shock tests and a target restricted to newly failed nodes.
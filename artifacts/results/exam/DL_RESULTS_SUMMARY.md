# Final Deep Learning result summary

- Validation-selected architecture: **Hybrid GNN–Transformer**.
- Test new-failure PR-AUC: **0.959** versus prevalence **0.215**.
- Difference from the strongest heuristic: **+0.294** PR-AUC.
- Difference from the MLP: **+0.323** PR-AUC.
- Node-level precision / recall / F1: **0.873 / 0.881 / 0.877**.
- Final cascade-fraction MAE: **0.092**.
- Next-load MAE: **0.135**.

**Defensible conclusion:** the pipeline can learn and evaluate next-step cascade risk on graph-disjoint synthetic networks. The strength of the architecture claim depends on the observed MLP–graph-model gap and its stability across graph families, regimes, bootstrap intervals, and repeated seeds.

**Main limitation:** this remains a controlled synthetic experiment, not validation for a real critical infrastructure system.
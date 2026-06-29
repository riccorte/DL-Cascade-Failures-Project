# CIRCLES prototype result summary

- Validation-selected candidate: **Hybrid GNN–Transformer**.
- Selection rule: **lowest validation cost per active node**.
- Illustrative cost ratio FN:FP: **5:1**.
- Test cost per active node: **0.101**.
- Test PR-AUC: **0.971**.
- Precision / recall / alert rate: **0.802 / 0.970 / 0.315**.
- Recall and lift at 10% budget: **0.337 / 2.99×**.
- Calibrated Brier / ECE: **0.036 / 0.006**.
- Final cascade-fraction MAE: **0.072**.
- Weakest observed operating condition: **by_regime / long_range**.

**Prototype value:** produce a ranked, probability-based shortlist of assets and quantify how much future failure risk is captured under a limited intervention budget.

**Main limitation:** all evidence is synthetic and tied to the current simulator, graph families, cascade rules, and illustrative costs.
# Additive-oracle numerical-precision control

**Note date:** 2026-04-27 17:25:51

The planned control evaluated the pairwise Shapley interaction estimator on a
known additive predictor, without training a model:

```python
import numpy as np

def f_add(X):
    return np.sin(X).sum(axis=1)
```

`X` has one row per sample and one column per panel SNP, with float32 inputs.
For this additive function, each pair's second-order difference cancels in exact
arithmetic when the four coalition predictions use matched background rows.
Numerical residuals therefore diagnose finite-precision computation.

The specified settings were 30 coalition samples, 15 background rows, 15 evaluation
rows, and pair seed `42 + ridx * 1009`. `ridx` is the pair-table row index.
The test set contained 12 published reference pairs for Neomycin and 12 for
Lactate; `panel_j_local` and `panel_k_local` identify their panel-column indices.

The retained implementation is `scripts/history/yeast_bloom_estimator_floor_control.py`.
The numerical output directory is `outputs/history/yeast/yeast_bloom_kan/estimator_floor/`.
This note specifies the control; it is not an execution receipt or a fitted-model verdict.

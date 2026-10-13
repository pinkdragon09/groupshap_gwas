# Additive-oracle control execution

**Note date:** 2026-04-27 17:35:49

`scripts/history/yeast_bloom_estimator_floor_control.py` was written and run on
12 published reference pairs for each of Neomycin and Lactate. It evaluated
`f_add(X) = np.sin(X).sum(axis=1)` without model fitting.

The interaction estimator used 30 coalition samples, 15 background rows and
15 evaluation rows, with seed `42 + ridx * 1009` for each pair-table row `ridx`.
These matched the settings of the control that replaces the KAN outer mapping
with an affine layer.

The run produced `floor_results.csv` and `summary.json` under
`outputs/history/yeast/yeast_bloom_kan/estimator_floor/`.
This receipt records execution and output types; it does not assert a numerical
verdict or interpret fitted-model residuals as biological evidence.

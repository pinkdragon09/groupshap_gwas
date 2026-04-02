# Prediction-loop DataFrame overhead

**Note date:** 2026-03-31

The grouped Shapley helper `value_function_numpy` in `scripts/grouped_shap_utils.py`
constructs a pandas DataFrame for each synthetic-background prediction. Repeating
that conversion inside the coalition-sampling loop can add CPU and allocation cost.

The proposed optimization was to avoid repeated conversion when a model accepts
NumPy arrays, or to reuse a compatible wrapper when feature names are required.
Column-name-aware models must continue to receive the input format they expect.
This record describes a proposed optimization, not a completed fix or measured speedup.

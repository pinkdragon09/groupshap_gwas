"""Shared first-order ALE, validated against PyALE 1.2.0.

See docs/ale_migration.md for estimator changes and historical provenance.
"""
import numpy as np
import pandas as pd


def ale_effect(model, X, feature_col, n_bins=20, *, feature_type="auto"):
    """Return x, ale, n_obs using PyALE, without plots or confidence intervals.

    Auto retains the historical <= n_bins distinct-value threshold. Discrete
    transitions pool both adjacent states and use frequency-weighted centering.
    Continuous results use PyALE's boundary grid and trapezoidal centering;
    their boundary-count-weighted mean is not generally zero.
    Constant features return one zero; empty/nonfinite target features fail.
    """
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
        raise ValueError("X must be a DataFrame with unique columns")
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if feature_type not in ("auto", "discrete", "continuous"):
        raise ValueError("feature_type must be auto, discrete, or continuous")
    values = X[feature_col].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("ALE requires a nonempty, finite target feature")
    unique = np.unique(values)
    if len(unique) == 1:
        return pd.DataFrame({"x": unique, "ale": [0.0], "n_obs": [len(X)]})
    if feature_type == "auto":
        feature_type = "discrete" if len(unique) <= n_bins else "continuous"
    from PyALE import ale

    # PyALE uses positional prediction arrays; normalizing row labels also
    # permits callers with repeated indices without alignment surprises.
    result = ale(X=X.reset_index(drop=True), model=model, feature=[feature_col],
                 feature_type=feature_type, grid_size=n_bins,
                 include_CI=False, plot=False)
    return pd.DataFrame({"x": result.index.to_numpy(),
                         "ale": result["eff"].to_numpy(),
                         "n_obs": result["size"].to_numpy(dtype=int)})


def binary_ale(predict, x, n_features, *, low=-1, high=1):
    """Batched binary specialization, retaining per-row contrasts for fivefold.

    Equivalent to discrete PyALE where both states occur. For monomorphic
    columns, retains the historical counterfactual contrast at the absent
    state; this is not an empirically supported ALE curve.
    Predictions must already be in the desired outcome units.
    """
    x = np.asarray(x)
    if (x.ndim != 2 or not len(x) or not 0 < n_features <= x.shape[1]
            or low >= high or not np.isin(x[:, :n_features], [low, high]).all()):
        raise ValueError("Expected nonempty binary feature columns and low < high")
    batches = []
    for j in range(n_features):
        lower, upper = x.copy(), x.copy()
        lower[:, j], upper[:, j] = low, high
        batches.extend([lower, upper])
    pred = np.asarray(predict(np.concatenate(batches))).reshape(n_features, 2, len(x))
    effects = (pred[:, 1] - pred[:, 0]).T
    delta = effects.mean(axis=0)
    p = ((x[:, :n_features] - low) / (high - low)).mean(axis=0)
    return dict(binary_ale_local_difference=effects,
                binary_ale_levels=np.column_stack([-p * delta, (1 - p) * delta]),
                test_allele_frequency=p)

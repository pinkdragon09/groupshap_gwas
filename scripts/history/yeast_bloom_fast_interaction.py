"""Fast pairwise Shapley-interaction sweep for yeast Bloom experiment.

The generic ``grouped_interaction_all`` in ``grouped_shap_utils.py`` is too
slow at the scale required for top-N markers because it pays a per-call
DataFrame construction cost and four separate ``model.predict`` calls per
MC sample.

This module is the same Shapley interaction estimator (algebraically
identical), specialized to:
  - single-feature groups (one marker per group)
  - numpy-only path (RF was trained on numpy, predict on numpy)
  - per pair, process eval points in a streaming inner loop, and batch
    all 4 value-function evaluations across one (eval_pt, mc_window)
    chunk into ONE ``model.predict`` call

Memory: kept small (one (4*mc_chunk*n_bg, p) buffer per pair worker).

Determinism: per (pair, eval_point) we draw a deterministic
``np.random.default_rng(random_state + pair_offset + eval_offset)``.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def _interaction_for_pair(
    model,
    bg: np.ndarray,
    eval_arr: np.ndarray,
    j_idx: int,
    k_idx: int,
    pair_label_j: str,
    pair_label_k: str,
    candidate_indices: np.ndarray,
    n_samples: int,
    base_seed: int,
) -> dict:
    """Compute I_{jk} averaged over all eval points.

    Per eval point, builds a (4 * n_samples * n_bg, p) batch and runs ONE
    predict for that eval point. Results across eval points are accumulated
    streaming -- we never hold more than one eval-point batch in memory.
    """
    n_bg, p = bg.shape
    n_eval = eval_arr.shape[0]
    rest_mask = (candidate_indices != j_idx) & (candidate_indices != k_idx)
    rest_indices = candidate_indices[rest_mask]
    m_minus_2 = len(rest_indices)
    block = 4 * n_bg

    per_eval_I = np.empty(n_eval, dtype=np.float64)

    # One reusable buffer per pair-worker, sized for one eval point.
    X_batch = np.empty((n_samples * block, p), dtype=bg.dtype)

    for e_i in range(n_eval):
        rng = np.random.default_rng(base_seed + e_i)
        eval_x = eval_arr[e_i]
        eval_j = eval_x[j_idx]
        eval_k = eval_x[k_idx]
        for s_i in range(n_samples):
            if m_minus_2 > 0:
                perm = rng.permutation(rest_indices)
                s_size = int(rng.integers(0, m_minus_2 + 1))
                S = perm[:s_size]
            else:
                S = np.empty(0, dtype=np.int64)
            base = s_i * block
            # v(S): bg with S columns pinned to eval_x
            X_batch[base:base + n_bg] = bg
            if S.size:
                X_batch[base:base + n_bg, S] = eval_x[S]
            # v(S, j): same as above + pin j
            X_batch[base + n_bg:base + 2 * n_bg] = X_batch[base:base + n_bg]
            X_batch[base + n_bg:base + 2 * n_bg, j_idx] = eval_j
            # v(S, k): same as v(S) + pin k
            X_batch[base + 2 * n_bg:base + 3 * n_bg] = X_batch[base:base + n_bg]
            X_batch[base + 2 * n_bg:base + 3 * n_bg, k_idx] = eval_k
            # v(S, j, k): same as v(S, j) + pin k
            X_batch[base + 3 * n_bg:base + 4 * n_bg] = X_batch[base + n_bg:base + 2 * n_bg]
            X_batch[base + 3 * n_bg:base + 4 * n_bg, k_idx] = eval_k

        preds = model.predict(X_batch)
        # Reshape: (n_samples, 4, n_bg)
        preds = preds.reshape(n_samples, 4, n_bg)
        v = preds.mean(axis=2)  # (n_samples, 4)
        delta = v[:, 3] - v[:, 1] - v[:, 2] + v[:, 0]
        per_eval_I[e_i] = delta.mean()

    mean_I = float(per_eval_I.mean())
    sd_I = float(per_eval_I.std(ddof=0))
    mean_abs_I = float(np.mean(np.abs(per_eval_I)))
    return {
        "marker_j": pair_label_j,
        "marker_k": pair_label_k,
        "j_idx": int(j_idx),
        "k_idx": int(k_idx),
        "mean_interaction": mean_I,
        "mean_abs_interaction": mean_abs_I,
        "sd_interaction": sd_I,
    }


def pairwise_interaction_fast(
    model,
    X_bg: np.ndarray,
    X_eval: np.ndarray,
    candidate_indices: Sequence[int],
    candidate_labels: Sequence[str],
    n_samples: int = 30,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute pairwise Shapley interaction values across candidate markers.

    Each candidate is a single-feature "group". The estimator is the same
    unbiased MC Shapley interaction index used in
    ``grouped_shap_utils.grouped_shapley_interaction_single_mc`` but with
    batched predict for ~50x speedup.

    Parameters
    ----------
    model : fitted sklearn-compatible regressor (numpy predict path).
    X_bg, X_eval : ndarray of shape (n_bg, p) and (n_eval, p).
    candidate_indices : indices into the columns of X_bg/X_eval for the
        candidate markers (length m).
    candidate_labels : human-readable labels of the same length (e.g.,
        marker IDs).
    n_samples : MC samples per (pair, eval_point).
    random_state : seed; per (pair, eval_point) seed = random_state +
        pair_offset + eval_offset.
    """
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    bg = np.ascontiguousarray(X_bg, dtype=np.float32)
    ev = np.ascontiguousarray(X_eval, dtype=np.float32)
    m = len(candidate_indices)

    pairs = [(j, k) for j in range(m) for k in range(j + 1, m)]
    n_eval = ev.shape[0]
    if verbose:
        print(f"[fast pairwise] {m} markers -> {len(pairs)} pairs, "
              f"n_bg={bg.shape[0]}, n_eval={n_eval}, mc={n_samples}")

    def pair_seed(pi: int) -> int:
        return random_state + (pi + 1) * (n_eval + 1) * 1009

    # Force RF predict to single-thread inside each worker so the loky
    # outer parallelism (default backend) gets clean per-process scaling.
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_interaction_for_pair)(
            model, bg, ev,
            int(candidate_indices[j]), int(candidate_indices[k]),
            str(candidate_labels[j]), str(candidate_labels[k]),
            candidate_indices, n_samples, pair_seed(pi),
        )
        for pi, (j, k) in enumerate(pairs)
    )
    if verbose:
        print(f"[fast pairwise] done in {time.time() - t0:.1f}s "
              f"({(time.time() - t0) / max(len(pairs), 1) * 1000:.1f} ms/pair)")
    df = pd.DataFrame(results)
    return df.sort_values("mean_abs_interaction", ascending=False).reset_index(drop=True)

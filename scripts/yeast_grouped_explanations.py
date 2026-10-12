"""Streaming interventional GroupSHAP for frozen yeast fold models.

The game is the same empirical-background game as
``xai_gwas.attribution.shapley.value_function_numpy``: all columns in a
group are replaced together, while the other columns retain a complete
background row. Groups must form a partition. Predictors accept a NumPy
matrix and return one scalar prediction per row.

First-order values share permutation paths across groups. Every sampled
path telescopes exactly from the background prediction to the explained
prediction; this changes Monte Carlo coupling, not the target estimand.
Interaction indices use the existing project's convention without a
factor of 1/2. Their coalition size is uniform on 0,...,m-2.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
from typing import Callable, Sequence

import numpy as np

Predictor = Callable[[np.ndarray], np.ndarray]


@dataclass
class ShapleyResult:
    phi: np.ndarray
    phi_halves: np.ndarray | None
    baseline: float
    predictions: np.ndarray
    efficiency_max_error: float


@dataclass
class InteractionResult:
    values: np.ndarray
    pairs: list[tuple[int, int]]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate(background, evaluation, groups, batch_size, eval_chunk_size, seed):
    background = np.asarray(background)
    evaluation = np.asarray(evaluation)
    for name, arr in (("background", background), ("evaluation", evaluation)):
        if arr.ndim != 2 or not all(arr.shape):
            raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
        if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
            raise ValueError(f"{name} must contain real numeric values")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must contain finite values")
    if background.shape[1] != evaluation.shape[1]:
        raise ValueError("background and evaluation must have the same columns")
    dtype = np.result_type(background.dtype, evaluation.dtype, np.float32)
    background = np.asarray(background, dtype=dtype)
    evaluation = np.asarray(evaluation, dtype=dtype)
    groups = list(groups)
    if not groups:
        raise ValueError("groups must be a nonempty partition")
    owners = np.full(background.shape[1], -1, dtype=int)
    for j, group in enumerate(groups):
        columns = list(group)
        if not columns:
            raise ValueError("groups must be nonempty")
        for column in columns:
            if isinstance(column, (bool, np.bool_)) or not isinstance(column, Integral):
                raise ValueError("group columns must be integer indices")
            if not 0 <= column < len(owners):
                raise ValueError("group column is outside the feature matrix")
            if owners[column] != -1:
                raise ValueError("groups must be disjoint, with no duplicate columns")
            owners[column] = j
    if np.any(owners < 0):
        raise ValueError("groups must cover every feature column")
    batch_size = _positive_integer(batch_size, "batch_size")
    eval_chunk_size = _positive_integer(eval_chunk_size, "eval_chunk_size")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return background, evaluation, owners, len(groups), batch_size, eval_chunk_size, int(seed)


def _predict_rows(predict: Predictor, rows: np.ndarray, batch_size: int) -> np.ndarray:
    predictions = np.empty(len(rows), dtype=float)
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        values = np.asarray(predict(rows[start:stop]))
        if values.shape == (stop - start, 1):
            values = values[:, 0]
        if values.shape != (stop - start,) or not np.issubdtype(values.dtype, np.number):
            raise ValueError("predictor must return one numeric scalar per input row")
        if np.iscomplexobj(values) or not np.isfinite(values).all():
            raise ValueError("predictor returned nonfinite or complex predictions")
        predictions[start:stop] = values
    return predictions


def _coalition_values(predict, background, evaluation, owners, row_indices, masks, batch_size):
    """Materialize at most one prediction batch of background interventions."""
    values = np.empty(len(masks), dtype=float)
    n_background = len(background)
    requests_per_chunk = max(1, batch_size // n_background)
    for start in range(0, len(masks), requests_per_chunk):
        stop = min(start + requests_per_chunk, len(masks))
        included_columns = masks[start:stop, owners]
        synthetic = np.where(
            included_columns[:, None, :],
            evaluation[row_indices[start:stop], None, :],
            background[None, :, :],
        )
        predictions = _predict_rows(predict, synthetic.reshape(-1, background.shape[1]), batch_size)
        values[start:stop] = predictions.reshape(stop - start, n_background).mean(axis=1)
    return values


def grouped_shapley_batch(
    predict: Predictor,
    background: np.ndarray,
    evaluation: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    n_permutations: int = 32,
    seed: int = 42,
    batch_size: int = 8192,
    eval_chunk_size: int = 16,
    return_halves: bool = True,
) -> ShapleyResult:
    """Return local grouped values and optional independent MC half estimates.

    Row-position seeds make results independent of both chunk sizes. Using
    the same seed for equally ordered train/test explanations matches the
    Monte Carlo draws, while backgrounds must remain training-only. The
    half estimates split independent permutations; they measure Monte Carlo
    sensitivity, not sampling uncertainty or out-of-sample validity.
    """
    background, evaluation, owners, m, batch_size, eval_chunk_size, seed = _validate(
        background, evaluation, groups, batch_size, eval_chunk_size, seed
    )
    n_permutations = _positive_integer(n_permutations, "n_permutations")
    if return_halves and n_permutations % 2:
        raise ValueError("an even n_permutations is required for independent halves")
    baseline = float(_predict_rows(predict, background, batch_size).mean())
    predictions = _predict_rows(predict, evaluation, batch_size)
    phi = np.empty((len(evaluation), m), dtype=float)
    halves = np.empty((2, len(evaluation), m), dtype=float) if return_halves else None
    for start in range(0, len(evaluation), eval_chunk_size):
        stop = min(start + eval_chunk_size, len(evaluation))
        n_rows = stop - start
        permutations = np.empty((n_rows, n_permutations, m), dtype=int)
        for local_row, row in enumerate(range(start, stop)):
            rng = np.random.default_rng(np.random.SeedSequence([seed, row, 1]))
            for sample in range(n_permutations):
                permutations[local_row, sample] = rng.permutation(m)
        path = np.empty((n_rows, n_permutations, m + 1), dtype=float)
        path[:, :, 0] = baseline
        path[:, :, -1] = predictions[start:stop, None]
        if m > 1:
            # Each prefix retains all previously included groups.
            masks = np.zeros((n_rows, n_permutations, m, m), dtype=bool)
            np.put_along_axis(masks, permutations[..., None], True, axis=-1)
            masks = np.logical_or.accumulate(masks, axis=2)[:, :, :-1, :]
            row_indices = np.repeat(np.arange(start, stop), n_permutations * (m - 1))
            path[:, :, 1:-1] = _coalition_values(
                predict, background, evaluation, owners, row_indices,
                masks.reshape(-1, m), batch_size,
            ).reshape(n_rows, n_permutations, m - 1)
        contributions = np.take_along_axis(
            np.diff(path, axis=2), np.argsort(permutations, axis=2), axis=2
        )
        phi[start:stop] = contributions.mean(axis=1)
        if halves is not None:
            midpoint = n_permutations // 2
            halves[0, start:stop] = contributions[:, :midpoint].mean(axis=1)
            halves[1, start:stop] = contributions[:, midpoint:].mean(axis=1)
    error = float(np.max(np.abs(phi.sum(axis=1) - (predictions - baseline))))
    return ShapleyResult(phi, halves, baseline, predictions, error)


def grouped_interactions_batch(
    predict: Predictor,
    background: np.ndarray,
    evaluation: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    n_samples: int = 10,
    seed: int = 4042,
    batch_size: int = 8192,
    eval_chunk_size: int = 16,
) -> InteractionResult:
    """Compute every unordered local group interaction, with no /2 factor.

    For each pair/row draw subset size uniformly from 0,...,m-2, then
    choose a uniformly random subset of that size. This targets weights
    |S|! (m-|S|-2)! / (m-1)!, matching the existing estimator.
    """
    background, evaluation, owners, m, batch_size, eval_chunk_size, seed = _validate(
        background, evaluation, groups, batch_size, eval_chunk_size, seed
    )
    n_samples = _positive_integer(n_samples, "n_samples")
    pairs = list(combinations(range(m), 2))
    values = np.empty((len(evaluation), len(pairs)), dtype=float)
    for pair_index, (j, k) in enumerate(pairs):
        rest = np.array([g for g in range(m) if g not in (j, k)], dtype=int)
        for start in range(0, len(evaluation), eval_chunk_size):
            stop = min(start + eval_chunk_size, len(evaluation))
            masks = np.zeros((stop - start, n_samples, 4, m), dtype=bool)
            for local_row, row in enumerate(range(start, stop)):
                rng = np.random.default_rng(np.random.SeedSequence([seed, row, j, k, 2]))
                for sample in range(n_samples):
                    permutation = rng.permutation(rest)
                    subset_size = int(rng.integers(0, len(rest) + 1))
                    masks[local_row, sample, :, permutation[:subset_size]] = True
            masks[:, :, 1, j] = True
            masks[:, :, 2, k] = True
            masks[:, :, 3, j] = True
            masks[:, :, 3, k] = True
            row_indices = np.repeat(np.arange(start, stop), n_samples * 4)
            coalition_values = _coalition_values(
                predict, background, evaluation, owners, row_indices,
                masks.reshape(-1, m), batch_size,
            ).reshape(stop - start, n_samples, 4)
            differences = (coalition_values[:, :, 3] - coalition_values[:, :, 1]
                           - coalition_values[:, :, 2] + coalition_values[:, :, 0])
            values[start:stop, pair_index] = differences.mean(axis=1)
    return InteractionResult(values, pairs)

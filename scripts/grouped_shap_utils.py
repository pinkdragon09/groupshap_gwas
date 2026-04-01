"""
Cooperative-game grouped Shapley utilities.

These functions implement the exact grouped Shapley formalism.

Design
------
- Players  = feature GROUPS (e.g. chromosomes, LD blocks, cis windows), not
             individual SNPs.  This is the cooperative-game form, not a sum of
             per-SNP SHAP values.
- Value function  v(S; x) = E[f(x_S, X_{-S})]  (interventional): groups in
  coalition S are fixed to the evaluation point x; remaining groups AND any
  non-grouped features (covariates, PCs) are drawn from a background dataset.
- The sum of grouped Shapley values equals v(M; x) - v(∅; x), where M is the
  set of all SNP groups.  Because covariates are always marginalized, this is
  NOT identically f(x) - E[f(X)], but rather the explained variation that is
  attributable to the SNP groups alone.

Compute cost
------------
With m SNP groups, each call to grouped_shapley_all() makes
  2^m * n_background * n_eval_points
model predictions.  For m ≤ 5 (Arabidopsis chromosomes) and
n_background = 100, n_eval = 50 this is 160 000 predictions per fold —
feasible for RF (~2 min); slower for transformers.
"""

from __future__ import annotations

import math
from itertools import chain, combinations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Core coalition-form utilities
# ─────────────────────────────────────────────────────────────────────────────

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def value_function(
    model,
    X_background: pd.DataFrame,
    eval_x: pd.Series,
    coalition: frozenset,
    group_cols: list[list[str]],
) -> float:
    """
    Interventional v(S; x): fix groups in S to eval_x values; marginalise rest.
    Non-grouped columns (covariates) remain at their background values.
    """
    X_syn = X_background.copy()
    for j in coalition:
        for col in group_cols[j]:
            X_syn[col] = float(eval_x[col])
    return float(model.predict(X_syn).mean())


def all_value_functions(
    model,
    X_background: pd.DataFrame,
    eval_x: pd.Series,
    group_cols: list[list[str]],
) -> dict[frozenset, float]:
    """Pre-compute v(S; eval_x) for all 2^m group subsets."""
    m = len(group_cols)
    return {
        frozenset(S): value_function(model, X_background, eval_x, frozenset(S), group_cols)
        for S in powerset(range(m))
    }


def grouped_shapley_single(v_cache: dict[frozenset, float], k: int, m: int) -> float:
    """
    phi_k = sum_{S not containing k} w(S) [v(S u {k}) - v(S)]
    w(S)  = |S|! (m - |S| - 1)! / m!
    """
    total = 0.0
    for S in powerset(range(m)):
        if k in S:
            continue
        s = len(S)
        w = math.factorial(s) * math.factorial(m - s - 1) / math.factorial(m)
        S_with_k = frozenset(S) | frozenset([k])
        total += w * (v_cache[S_with_k] - v_cache[frozenset(S)])
    return total


def grouped_shapley_all(
    model,
    X_background: pd.DataFrame,
    eval_points: pd.DataFrame,
    group_cols: list[list[str]],
    group_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    True cooperative-game grouped Shapley for all eval points.
    Returns DataFrame of shape (n_eval, m) with one column per group.
    model.predict() must accept a pd.DataFrame and return a 1-D array.
    """
    m = len(group_cols)
    cols = group_names if group_names is not None else [f"group_{k}" for k in range(m)]
    rows = []
    for _, eval_x in eval_points.iterrows():
        v_cache = all_value_functions(model, X_background, eval_x, group_cols)
        rows.append([grouped_shapley_single(v_cache, k, m) for k in range(m)])
    return pd.DataFrame(rows, columns=cols)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for real-data scripts
# ─────────────────────────────────────────────────────────────────────────────

def build_group_cols_from_meta(
    feature_meta: pd.DataFrame,
    group_col: str = "chromosome",
) -> tuple[list[str], list[list[str]]]:
    """
    Build (group_names, group_cols) from feature_meta.
    Only SNP features (feature_type == "snp") are grouped; covariates are
    left out of the grouping and are always marginalised in the value function.

    Returns
    -------
    group_names : list[str]   sorted group labels (e.g. ['Chr1', 'Chr2', ...])
    group_cols  : list[list[str]]  group_cols[j] = SNP column names in group j
    """
    snp_meta = feature_meta[feature_meta["feature_type"] == "snp"].copy()
    group_names = sorted(snp_meta[group_col].dropna().unique())
    group_cols = [
        snp_meta.loc[snp_meta[group_col] == g, "feature"].tolist()
        for g in group_names
    ]
    # Drop empty groups
    non_empty = [(n, c) for n, c in zip(group_names, group_cols) if c]
    group_names = [p[0] for p in non_empty]
    group_cols = [p[1] for p in non_empty]
    return group_names, group_cols


def load_group_manifest(manifest_path: str) -> tuple[list[str], list[list[str]], dict[str, str]]:
    """
    Load a group manifest TSV with columns: feature, group.
    Returns (group_names, group_cols, feature_to_group).
    Used by the generic shap_gwas_template.py.
    """
    df = pd.read_csv(manifest_path, sep="\t")
    if not {"feature", "group"}.issubset(df.columns):
        raise ValueError(f"Group manifest must have 'feature' and 'group' columns; got {list(df.columns)}")
    feature_to_group = dict(zip(df["feature"], df["group"]))
    group_names = sorted(df["group"].unique())
    group_cols = [
        df.loc[df["group"] == g, "feature"].tolist()
        for g in group_names
    ]
    return group_names, group_cols, feature_to_group


def summarize_grouped_shapley(
    phi_df: pd.DataFrame,
    group_names: list[str],
) -> pd.DataFrame:
    """
    Aggregate grouped Shapley values across eval points.
    Returns one row per group with mean_phi and sd_phi.
    """
    rows = []
    for k, name in enumerate(group_names):
        col = phi_df.columns[k]
        vals = phi_df[col].values
        rows.append(
            {
                "group": name,
                "mean_phi": float(np.mean(vals)),
                "mean_abs_phi": float(np.mean(np.abs(vals))),
                "sd_abs_phi": float(np.std(np.abs(vals))),
                "n_eval_points": len(vals),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_phi", ascending=False)

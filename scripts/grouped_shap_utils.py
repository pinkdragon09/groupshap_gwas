"""
Cooperative-game grouped Shapley utilities.
Optimized for high-dimensional SNP data using NumPy.
"""

from __future__ import annotations

import math
from itertools import chain, combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def value_function_numpy(
    model,
    background_arr: np.ndarray,
    eval_x_arr: np.ndarray,
    coalition: frozenset,
    group_indices: list[list[int]],
    feature_names: list[str],
) -> float:
    """
    Interventional value function v(S; x).
    Groups in coalition S are fixed to eval_x_arr values; remaining groups
    are marginalised over background_arr rows.
    Converts to DataFrame before predict() so pipeline models and column-name-
    aware wrappers (e.g. SklearnWrapperForGroupedShap) work correctly.
    """
    assert eval_x_arr.ndim == 1, (
        f"eval_x_arr must be 1-D (a single row), got shape {eval_x_arr.shape}"
    )
    X_syn = background_arr.copy()
    for j in coalition:
        indices = group_indices[j]
        X_syn[:, indices] = eval_x_arr[indices]
    df_syn = pd.DataFrame(X_syn, columns=feature_names)
    return float(model.predict(df_syn).mean())


def grouped_shapley_single_mc(
    model,
    background_arr: np.ndarray,
    eval_x_arr: np.ndarray,
    k: int,
    m: int,
    group_indices: list[list[int]],
    feature_names: list[str],
    n_samples: int = 50,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> float:
    """
    Monte Carlo estimate of grouped Shapley φ_k.
    Samples random permutations of all m groups; S = groups appearing before k.
    This reproduces the Shapley weight |S|!(m-|S|-1)!/m! exactly in expectation.

    rng: a seeded RNG for deterministic output. If None, uses the global
    numpy RNG (legacy non-deterministic behavior).
    """
    groups = list(range(m))
    total_diff = 0.0
    permute = rng.permutation if rng is not None else np.random.permutation
    for _ in range(n_samples):
        perm = permute(groups)
        k_idx = int(np.where(perm == k)[0][0])
        S = frozenset(perm[:k_idx])
        v_with_k = value_function_numpy(model, background_arr, eval_x_arr, S | {k}, group_indices, feature_names)
        v_without_k = value_function_numpy(model, background_arr, eval_x_arr, S, group_indices, feature_names)
        total_diff += v_with_k - v_without_k
    return total_diff / n_samples


def grouped_shapley_interaction_single_mc(
    model,
    background_arr: np.ndarray,
    eval_x_arr: np.ndarray,
    j: int,
    k: int,
    m: int,
    group_indices: list[list[int]],
    feature_names: list[str],
    n_samples: int = 100,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> float:
    """
    Unbiased Monte Carlo estimate of the Shapley Interaction Index I_{jk}.

        I_{jk} = Σ_{S ⊆ M\\{j,k}} w(s,m) [v(S∪{j,k}) - v(S∪{j}) - v(S∪{k}) + v(S)]
        w(s,m) = s!(m-s-2)! / (m-1)!

    Sampling distribution (proved unbiased):
      Draw s uniformly from {0,...,m-2}  →  P(size=s) = 1/(m-1)
      Draw a random permutation of M\\{j,k} and take its first s elements as S.
      Then P(specific S of size s) = s!(m-2-s)!/((m-1)(m-2)!) = s!(m-s-2)!/(m-1)! = w(s,m).

    rng: a seeded RNG for deterministic output. If None, uses the global
    numpy RNG (legacy non-deterministic behavior).
    """
    rest = [g for g in range(m) if g not in (j, k)]
    permute = rng.permutation if rng is not None else np.random.permutation
    randint = rng.integers if isinstance(rng, np.random.Generator) else (
        rng.randint if rng is not None else np.random.randint
    )
    total_delta = 0.0
    for _ in range(n_samples):
        if rest:
            perm = permute(rest)
            s_size = randint(0, len(rest) + 1)   # uniform over {0,...,m-2}
            S = frozenset(perm[:s_size])
        else:
            S = frozenset()
        v_jk = value_function_numpy(model, background_arr, eval_x_arr, S | {j, k}, group_indices, feature_names)
        v_j  = value_function_numpy(model, background_arr, eval_x_arr, S | {j},     group_indices, feature_names)
        v_k  = value_function_numpy(model, background_arr, eval_x_arr, S | {k},     group_indices, feature_names)
        v_0  = value_function_numpy(model, background_arr, eval_x_arr, S,           group_indices, feature_names)
        total_delta += v_jk - v_j - v_k + v_0
    return total_delta / n_samples


def _explain_one_point(
    model,
    background_arr: np.ndarray,
    eval_row: np.ndarray,
    m: int,
    group_indices: list[list[int]],
    feature_names: list[str],
    mc_samples: int,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> list[float]:
    """Compute grouped Shapley values for a single eval point (parallelisable unit)."""
    return [
        grouped_shapley_single_mc(
            model, background_arr, eval_row, k, m, group_indices, feature_names, mc_samples, rng=rng
        )
        for k in range(m)
    ]


def grouped_shapley_all(
    model,
    X_background: pd.DataFrame,
    X_eval: pd.DataFrame,
    group_cols: list[list[str]],
    group_names: list[str] | None = None,
    mc_samples: int = 50,
    n_jobs: int = -1,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Compute grouped Shapley values for all eval points via Monte Carlo.
    Returns DataFrame of shape (n_eval, m_groups).

    n_jobs: number of parallel workers (passed to joblib). -1 uses all cores.
    random_state: if set, seeds a deterministic RNG per eval point
      (seed_i = random_state + i). If None, uses the global numpy RNG
      (legacy non-deterministic behavior). Determinism is only guaranteed
      with n_jobs=1; with threads, scheduling order can still affect ties.
    """
    m = len(group_cols)
    feature_names = list(X_background.columns)
    col_to_idx = {name: i for i, name in enumerate(feature_names)}
    group_indices = [[col_to_idx[c] for c in g] for g in group_cols]

    background_arr = X_background.values
    eval_arr = X_eval.values

    cols = group_names if group_names is not None else [f"group_{k}" for k in range(m)]

    if random_state is None:
        rngs = [None] * len(eval_arr)
    else:
        rngs = [np.random.default_rng(random_state + i) for i in range(len(eval_arr))]

    rows = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_explain_one_point)(
            model, background_arr, eval_arr[i], m, group_indices, feature_names, mc_samples, rng=rngs[i]
        )
        for i in range(len(eval_arr))
    )
    return pd.DataFrame(rows, columns=cols)


def grouped_interaction_all(
    model,
    X_background: pd.DataFrame,
    X_eval: pd.DataFrame,
    group_cols: list[list[str]],
    group_names: list[str],
    mc_samples: int = 50,
    n_jobs: int = -1,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Compute all pairwise Shapley Interaction Index values between groups.
    Returns DataFrame sorted by mean_abs_interaction.

    n_jobs: number of parallel workers (passed to joblib). -1 uses all cores.
    random_state: if set, seeds a deterministic RNG per (pair, eval-point).
      If None, uses the global numpy RNG (legacy non-deterministic behavior).
    """
    m = len(group_cols)
    feature_names = list(X_background.columns)
    col_to_idx = {name: i for i, name in enumerate(feature_names)}
    group_indices = [[col_to_idx[c] for c in g] for g in group_cols]

    background_arr = X_background.values
    eval_arr = X_eval.values

    def _pair_interactions(j: int, k: int) -> dict:
        if random_state is None:
            rngs = [None] * len(eval_arr)
        else:
            base = random_state + (j * m + k) * len(eval_arr)
            rngs = [np.random.default_rng(base + i) for i in range(len(eval_arr))]
        vals = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(grouped_shapley_interaction_single_mc)(
                model, background_arr, eval_arr[i], j, k, m, group_indices, feature_names, mc_samples, rng=rngs[i]
            )
            for i in range(len(eval_arr))
        )
        return {
            "group_j":            group_names[j],
            "group_k":            group_names[k],
            "mean_interaction":     float(np.mean(vals)),
            "mean_abs_interaction": float(np.mean(np.abs(vals))),
            "sd_interaction":       float(np.std(vals)),
        }

    results = [_pair_interactions(j, k) for j in range(m) for k in range(j + 1, m)]
    return pd.DataFrame(results).sort_values("mean_abs_interaction", ascending=False)


def summarize_grouped_shapley(
    phi_df: pd.DataFrame,
    group_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Column-name-driven, NaN-aware summary of grouped Shapley values.

    - Iterates by actual column name in ``phi_df`` (not by position).
    - Ignores NaNs so that cross-fold concatenated frames with fold-specific
      group columns are aggregated correctly: each group's statistics reflect
      only the rows where that group was actually evaluated.
    - The ``group_names`` argument is accepted for backwards compatibility but
      is ignored when ``phi_df`` has named columns; it is only used as a
      positional fallback when the frame's columns are unnamed/integer.

    **Caveat for dynamic groupings (LD/PLINK):** if the same group label
    refers to different SNP sets across folds (e.g. ``ld_cluster_0`` computed
    on different training subsamples), averaging the column across folds is
    *not* semantically valid. Callers should use per-fold summaries in that
    case; this helper will still return a numerical summary but the caller
    is responsible for interpreting it.
    """
    # Drop any non-numeric helper columns (e.g. 'fold')
    numeric_df = phi_df.select_dtypes(include=[np.number])
    if "fold" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["fold"])

    cols = list(numeric_df.columns)
    if group_names is not None and all(isinstance(c, (int, np.integer)) for c in cols):
        # Fallback: unnamed integer columns, use positional group_names
        rename = {c: group_names[i] for i, c in enumerate(cols) if i < len(group_names)}
        numeric_df = numeric_df.rename(columns=rename)
        cols = list(numeric_df.columns)

    rows = []
    for col in cols:
        vals = numeric_df[col].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        rows.append({
            "group":         col,
            "mean_phi":      float(np.mean(finite)),
            "mean_abs_phi":  float(np.mean(np.abs(finite))),
            "sd_abs_phi":    float(np.std(np.abs(finite))),
            "n_eval_points": int(finite.size),
        })
    return pd.DataFrame(rows).sort_values("mean_abs_phi", ascending=False)


def load_group_manifest(
    manifest_path: str,
) -> tuple[list[str], list[list[str]], pd.DataFrame]:
    """
    Load a group manifest TSV with columns 'feature' and 'group'.

    Returns
    -------
    group_names   : sorted list of unique group labels
    group_cols    : group_cols[j] = list of feature names in group j
    manifest_df   : raw manifest DataFrame (for downstream inspection)
    """
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    if "feature" not in manifest_df.columns or "group" not in manifest_df.columns:
        raise ValueError(
            f"Manifest file '{manifest_path}' must have 'feature' and 'group' columns. "
            f"Found: {list(manifest_df.columns)}"
        )
    group_names = sorted(manifest_df["group"].dropna().unique().tolist())
    group_cols = [
        manifest_df.loc[manifest_df["group"] == g, "feature"].tolist()
        for g in group_names
    ]
    pairs = [(n, c) for n, c in zip(group_names, group_cols) if c]
    if not pairs:
        return [], [], manifest_df
    group_names_out, group_cols_out = zip(*pairs)
    return list(group_names_out), list(group_cols_out), manifest_df


def build_group_cols_from_meta(
    feature_meta: pd.DataFrame,
    group_col: str = "chromosome",
) -> tuple[list[str], list[list[str]]]:
    """
    Build (group_names, group_cols) from feature_meta.
    Only SNP features (feature_type == 'snp') are grouped; covariates are
    excluded and always marginalised in the value function.

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
    pairs = [(n, c) for n, c in zip(group_names, group_cols) if c]
    if not pairs:
        return [], []
    group_names, group_cols = zip(*pairs)
    return list(group_names), list(group_cols)


# ---------------------------------------------------------------------------
# Gene-window grouping
# ---------------------------------------------------------------------------

def assign_snps_to_gene_windows(
    snp_meta: pd.DataFrame,
    gene_df: pd.DataFrame,
    window_bp: int = 50_000,
    snp_chrom_col: str = "chromosome",
    snp_pos_col: str = "position",
    snp_feature_col: str = "feature",
    gene_chrom_col: str = "chrom",
    gene_start_col: str = "start",
    gene_end_col: str = "end",
    gene_id_col: str = "gene_id",
    multi_gene: bool = False,
) -> pd.DataFrame:
    """
    Assign each SNP to the gene whose body (± window_bp) it falls within.

    Parameters
    ----------
    snp_meta        : DataFrame with columns for SNP feature name, chromosome, position.
                      Must also have a 'feature_type' column; only rows where
                      feature_type == 'snp' are processed.
    gene_df         : DataFrame with columns for chromosome, TSS/start, TES/end, gene_id.
    window_bp       : Window added symmetrically around each gene body (default 50 kb).
    snp_*_col       : Column names in snp_meta.
    gene_*_col      : Column names in gene_df.
    multi_gene      : If True, a SNP overlapping multiple gene windows is assigned to
                      all of them (one row per gene).  If False (default), each SNP is
                      assigned only to the closest gene centre within range; SNPs outside
                      all windows receive group label 'intergenic_<chrom>_<pos>'.

    Returns
    -------
    DataFrame with columns ['feature', 'group'] — the manifest expected by
    load_group_manifest() and build_group_cols_from_meta().
    """
    snp_only = snp_meta[snp_meta["feature_type"] == "snp"].copy()
    snp_only = snp_only[[snp_feature_col, snp_chrom_col, snp_pos_col]].rename(
        columns={snp_feature_col: "feature", snp_chrom_col: "chrom", snp_pos_col: "pos"}
    )
    snp_only["pos"] = snp_only["pos"].astype(int)

    gene_df = gene_df.copy()
    gene_df[gene_start_col] = gene_df[gene_start_col].astype(int)
    gene_df[gene_end_col] = gene_df[gene_end_col].astype(int)
    gene_df["window_start"] = gene_df[gene_start_col] - window_bp
    gene_df["window_end"] = gene_df[gene_end_col] + window_bp
    gene_df["gene_centre"] = (gene_df[gene_start_col] + gene_df[gene_end_col]) // 2

    records: list[dict] = []
    for chrom, snp_group in snp_only.groupby("chrom"):
        genes_on_chrom = gene_df[gene_df[gene_chrom_col] == chrom]
        for _, snp_row in snp_group.iterrows():
            pos = int(snp_row["pos"])
            overlapping = genes_on_chrom[
                (genes_on_chrom["window_start"] <= pos) &
                (genes_on_chrom["window_end"] >= pos)
            ]
            if overlapping.empty:
                group_label = f"intergenic_{chrom}_{pos}"
                records.append({"feature": snp_row["feature"], "group": group_label})
            elif multi_gene:
                for _, gr in overlapping.iterrows():
                    records.append({"feature": snp_row["feature"], "group": gr[gene_id_col]})
            else:
                # nearest gene centre
                dist = (overlapping["gene_centre"] - pos).abs()
                nearest = overlapping.loc[dist.idxmin(), gene_id_col]
                records.append({"feature": snp_row["feature"], "group": nearest})

    return pd.DataFrame(records, columns=["feature", "group"])


def build_gene_window_groups(
    snp_meta: pd.DataFrame,
    gene_df: pd.DataFrame,
    window_bp: int = 50_000,
    **kwargs,
) -> tuple[list[str], list[list[str]]]:
    """
    Convenience wrapper: returns (group_names, group_cols) using gene-window grouping.

    Parameters are forwarded to assign_snps_to_gene_windows().
    Compatible with grouped_shapley_all() and grouped_interaction_all().
    """
    manifest = assign_snps_to_gene_windows(snp_meta, gene_df, window_bp=window_bp, **kwargs)
    group_names = sorted(manifest["group"].unique().tolist())
    group_cols = [
        manifest.loc[manifest["group"] == g, "feature"].tolist()
        for g in group_names
    ]
    pairs = [(n, c) for n, c in zip(group_names, group_cols) if c]
    if not pairs:
        return [], []
    group_names_out, group_cols_out = zip(*pairs)
    return list(group_names_out), list(group_cols_out)


# ---------------------------------------------------------------------------
# Physical LD-block grouping (fixed-window proxy)
# ---------------------------------------------------------------------------

def build_ld_block_groups(
    snp_meta: pd.DataFrame,
    block_size_bp: int = 200_000,
    chrom_col: str = "chromosome",
    pos_col: str = "position",
    feature_col: str = "feature",
) -> tuple[list[str], list[list[str]]]:
    """
    Partition SNPs into fixed-size physical blocks as a proxy for LD blocks.

    Each block is labelled '<chrom>_block<block_index>' where block_index = position // block_size_bp.
    This is a practical stand-in when a precomputed LD block map (e.g. from PLINK --blocks) is not
    available; replace with load_group_manifest() once a proper block map exists.

    Parameters
    ----------
    snp_meta      : DataFrame with feature_type, chromosome, position, and feature columns.
    block_size_bp : Physical block size in base pairs (default 200 kb).
    chrom_col     : Column name for chromosome in snp_meta.
    pos_col       : Column name for position in snp_meta.
    feature_col   : Column name for SNP feature name in snp_meta.

    Returns
    -------
    (group_names, group_cols) compatible with grouped_shapley_all().
    """
    snp_only = snp_meta[snp_meta["feature_type"] == "snp"].copy()
    snp_only["_block"] = (
        snp_only[chrom_col].astype(str)
        + "_block"
        + (snp_only[pos_col].astype(int) // block_size_bp).astype(str)
    )
    group_names = sorted(snp_only["_block"].unique().tolist())
    group_cols = [
        snp_only.loc[snp_only["_block"] == g, feature_col].tolist()
        for g in group_names
    ]
    pairs = [(n, c) for n, c in zip(group_names, group_cols) if c]
    if not pairs:
        return [], []
    group_names_out, group_cols_out = zip(*pairs)
    return list(group_names_out), list(group_cols_out)


# ---------------------------------------------------------------------------
# Conditional Permutation Importance (CPI)
# ---------------------------------------------------------------------------

def conditional_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    group_cols: list[list[str]],
    group_names: list[str],
    n_repeats: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Group Permutation Importance (GPI), retaining the historical function name.

    For each group, jointly permute that group's columns across rows, without
    conditioning on the other groups (leaving their columns fixed),
    and measure the drop in R².  Repeating n_repeats times gives a stable
    mean and standard deviation.

    Parameters
    ----------
    model        : fitted sklearn-compatible regressor with predict()
    X            : feature matrix (DataFrame)
    y            : target (Series or array-like)
    group_cols   : list of lists of column names per group
    group_names  : labels matching group_cols
    n_repeats    : number of permutation repetitions per group
    random_state : RNG seed

    Returns
    -------
    DataFrame with columns: group, mean_importance, sd_importance
    sorted descending by mean_importance.
    """
    from sklearn.metrics import r2_score as _r2_score

    rng = np.random.RandomState(random_state)
    y_arr = np.asarray(y)
    baseline_r2 = _r2_score(y_arr, model.predict(X))

    records = []
    for name, cols in zip(group_names, group_cols):
        if not cols:
            continue
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            perm_idx = rng.permutation(len(X))
            X_perm[cols] = X[cols].values[perm_idx]
            perm_r2 = _r2_score(y_arr, model.predict(X_perm))
            drops.append(baseline_r2 - perm_r2)
        records.append(
            {
                "group": name,
                "mean_importance": float(np.mean(drops)),
                "sd_importance": float(np.std(drops)),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Accumulated Local Effects (ALE)
# ---------------------------------------------------------------------------

def ale_effect(
    model,
    X: pd.DataFrame,
    feature_col: str,
    n_bins: int = 20,
) -> pd.DataFrame:
    """
    Accumulated Local Effects (ALE) for a single feature.

    Standard first-order binning estimator:
      1. Partition the feature's range into n_bins quantile bins.
      2. For each bin, replace the feature values for in-bin observations
         with the bin's lower/upper edges and compute the local prediction
         difference.
      3. Accumulate those local effects and centre (subtract the weighted mean).

    Parameters
    ----------
    model       : fitted sklearn-compatible regressor with predict()
    X           : feature matrix (DataFrame)
    feature_col : name of the feature column to explain
    n_bins      : number of quantile bins

    Returns
    -------
    DataFrame with columns:
      bin_lower   – left edge of each bin
      bin_upper   – right edge of each bin
      bin_center  – midpoint (for plotting)
      ale         – centred accumulated local effect
      n_obs       – number of observations in the bin
    """
    feat = X[feature_col].values.astype(float)
    unique_vals = np.unique(feat[~np.isnan(feat)])

    # For discrete features (few unique values), use each unique value as its
    # own "bin" boundary and compute ALE at each transition.  For continuous
    # features, fall back to quantile binning.
    if len(unique_vals) <= n_bins:
        # Discrete / near-discrete: breakpoints are the unique values themselves.
        # For each adjacent pair (u_{j-1}, u_j), the local neighbourhood is
        # observations whose feature value is u_j (following Apley & Zhu 2020, §3.2).
        # Special case: single unique value → zero effect everywhere.
        if len(unique_vals) < 2:
            return pd.DataFrame(
                {"bin_lower": unique_vals, "bin_upper": unique_vals,
                 "bin_center": unique_vals, "ale": np.zeros(len(unique_vals)),
                 "n_obs": [len(feat)]}
            )
        quantiles = unique_vals
    else:
        quantiles = np.unique(np.nanpercentile(feat, np.linspace(0, 100, n_bins + 1)))

    n_intervals = len(quantiles) - 1
    local_effects = np.zeros(n_intervals)
    bin_counts = np.zeros(n_intervals, dtype=int)

    for i in range(n_intervals):
        lo, hi = quantiles[i], quantiles[i + 1]
        # For discrete features: use observations at the upper breakpoint as
        # the local neighbourhood (Apley & Zhu discrete ALE).
        # For continuous features: any observation within [lo, hi].
        if len(unique_vals) <= n_bins:
            mask = feat == hi
        elif i == 0:
            mask = (feat >= lo) & (feat <= hi)
        else:
            mask = (feat > lo) & (feat <= hi)
        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            # Fall back to using all observations when bin is empty
            mask = (feat >= lo) & (feat <= hi)
            n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            continue
        X_lo = X[mask].copy()
        X_hi = X[mask].copy()
        X_lo[feature_col] = lo
        X_hi[feature_col] = hi
        local_effects[i] = float((model.predict(X_hi) - model.predict(X_lo)).mean())
        bin_counts[i] = n_in_bin

    # ALE is defined at the K+1 quantile boundaries, with anchor ALE[0] = 0.
    # ALE[k] = ALE[k-1] + local_effect[k-1]  (accumulated from the left)
    ale_at_boundaries = np.concatenate([[0.0], np.cumsum(local_effects)])

    # Centre: c = (1/n) * Σ_i ALE(x_i) over all observations.
    # For each observation with feature value x_i, ALE(x_i) = ale_at_boundaries[k]
    # where k = the boundary index that x_i maps to.
    #
    # Continuous features: x_i in bin k → maps to upper boundary k+1.
    #   Boundary 0 (anchor) has no observations; boundary k>0 gets bin_counts[k-1].
    #
    # Discrete features: x_i == quantile[k] → maps exactly to boundary k.
    #   Boundary 0 is the first unique value; it gets its own observation count.
    if len(unique_vals) <= n_bins:
        # Discrete: count observations at each unique value
        boundary_weights = np.array(
            [(feat == q).sum() for q in quantiles], dtype=float
        )
    else:
        # Continuous: observations map to the upper boundary of their bin
        boundary_weights = np.concatenate([[0.0], bin_counts.astype(float)])

    total_obs = boundary_weights.sum()
    if total_obs > 0:
        ale_mean = float(np.dot(boundary_weights, ale_at_boundaries)) / total_obs
    else:
        ale_mean = 0.0
    ale_centred = ale_at_boundaries - ale_mean

    return pd.DataFrame(
        {
            "x": quantiles,
            "ale": ale_centred,
            "n_obs": boundary_weights.astype(int),
        }
    )


def build_cis_window_groups(
    snp_meta: pd.DataFrame,
    window_bp: int = 200_000,
    chrom_col: str = "chromosome",
    pos_col: str = "position",
    feature_col: str = "feature",
) -> tuple[list[str], list[list[str]]]:
    """
    Sliding cis-window grouping centred on evenly spaced anchors.

    Partitions the genome into non-overlapping windows of width window_bp,
    labelling each window '<chrom>_cis<anchor_kb>kb' where anchor_kb is the
    window start in kilobases.  This is coarser than gene-window grouping but
    avoids requiring a gene annotation file.

    Delegates to build_ld_block_groups() with block_size_bp=window_bp.
    """
    return build_ld_block_groups(
        snp_meta,
        block_size_bp=window_bp,
        chrom_col=chrom_col,
        pos_col=pos_col,
        feature_col=feature_col,
    )

"""SNP grouping strategies for grouped Shapley analysis.

GroupResult = (group_names: list[str], group_cols: list[list[str]])

All public functions return a GroupResult tuple compatible with
``xai_gwas.attribution.grouped_shapley_all()`` and ``grouped_interaction_all()``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Manifest-based grouping
# ---------------------------------------------------------------------------

def load_group_manifest(
    manifest_path: str,
) -> tuple[list[str], list[list[str]], pd.DataFrame]:
    """Load a group manifest TSV with columns 'feature' and 'group'.

    Returns
    -------
    group_names   : sorted list of unique group labels
    group_cols    : group_cols[j] = list of feature names in group j
    manifest_df   : raw manifest DataFrame
    """
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    if "feature" not in manifest_df.columns or "group" not in manifest_df.columns:
        raise ValueError(
            f"Manifest '{manifest_path}' must have 'feature' and 'group' columns. "
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
    """Build (group_names, group_cols) from a feature metadata DataFrame.

    Only SNP features (feature_type == 'snp') are grouped; covariates are
    excluded and always marginalised in the value function.
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
    gn, gc = zip(*pairs)
    return list(gn), list(gc)


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
    """Assign each SNP to the nearest gene body (± window_bp).

    Parameters
    ----------
    snp_meta   : DataFrame with feature_type, chromosome, position, feature columns.
    gene_df    : DataFrame with chrom, start, end, gene_id columns.
    window_bp  : Symmetric window around each gene body (default 50 kb).
    multi_gene : If True, SNPs overlapping multiple windows appear once per gene.
                 If False, each SNP is assigned to the nearest gene centre.

    Returns
    -------
    DataFrame with columns ['feature', 'group'] — compatible with load_group_manifest().
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
                records.append({"feature": snp_row["feature"],
                                "group": f"intergenic_{chrom}_{pos}"})
            elif multi_gene:
                for _, gr in overlapping.iterrows():
                    records.append({"feature": snp_row["feature"], "group": gr[gene_id_col]})
            else:
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
    """Convenience wrapper: returns (group_names, group_cols) using gene-window grouping."""
    manifest = assign_snps_to_gene_windows(snp_meta, gene_df, window_bp=window_bp, **kwargs)
    group_names = sorted(manifest["group"].unique().tolist())
    group_cols = [
        manifest.loc[manifest["group"] == g, "feature"].tolist()
        for g in group_names
    ]
    pairs = [(n, c) for n, c in zip(group_names, group_cols) if c]
    if not pairs:
        return [], []
    gn, gc = zip(*pairs)
    return list(gn), list(gc)


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
    """Partition SNPs into fixed-size physical blocks as a proxy for LD blocks.

    Each block is labelled '<chrom>_block<block_index>' where
    block_index = position // block_size_bp.

    Use ``load_group_manifest()`` when a precomputed LD block map from
    PLINK ``--blocks`` is available; this function is the fallback.
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
    gn, gc = zip(*pairs)
    return list(gn), list(gc)


def build_cis_window_groups(
    snp_meta: pd.DataFrame,
    window_bp: int = 200_000,
    chrom_col: str = "chromosome",
    pos_col: str = "position",
    feature_col: str = "feature",
) -> tuple[list[str], list[list[str]]]:
    """Sliding cis-window grouping using non-overlapping fixed-width windows.

    Delegates to build_ld_block_groups() with block_size_bp=window_bp.
    """
    return build_ld_block_groups(
        snp_meta,
        block_size_bp=window_bp,
        chrom_col=chrom_col,
        pos_col=pos_col,
        feature_col=feature_col,
    )


# ---------------------------------------------------------------------------
# Name-parsed window grouping (for PLINK-format SNP names)
# ---------------------------------------------------------------------------

def build_window_grouping(
    feature_names: list[str],
    window_size: int = 50_000,
) -> tuple[list[str], list[list[str]]]:
    """Group SNPs by physical window parsed from names of the form '<chr>_<pos>'.

    This is the lightweight fallback used by the CLI when no snp_meta
    DataFrame is available — position is extracted from the column name.
    SNPs that cannot be parsed fall into a single 'unknown' group.
    """
    parsed = []
    unknown = []
    for name in feature_names:
        parts = name.rsplit("_", 1)
        try:
            pos = int(parts[-1])
            parsed.append({"feature": name, "pos": pos})
        except (ValueError, IndexError):
            unknown.append(name)

    if not parsed:
        return ["unknown"], [feature_names]

    df = pd.DataFrame(parsed)
    df["win"] = (df["pos"] // window_size) * window_size
    df["group_name"] = "win_" + df["win"].astype(str)

    group_names = sorted(df["group_name"].unique())
    group_cols = [df.loc[df["group_name"] == g, "feature"].tolist() for g in group_names]

    if unknown:
        group_names.append("unknown")
        group_cols.append(unknown)

    return group_names, group_cols


# ---------------------------------------------------------------------------
# Correlation-based LD grouping (from ld_grouping.py)
# ---------------------------------------------------------------------------

def group_by_ld_correlation(
    X: pd.DataFrame,
    r2_threshold: float = 0.5,
) -> tuple[list[str], list[list[str]]]:
    """Group SNPs using agglomerative clustering on r² correlation distance.

    Parameters
    ----------
    X            : Genotype DataFrame (samples × SNPs).
    r2_threshold : SNPs with r² ≥ threshold are placed in the same cluster.

    Returns
    -------
    (group_names, group_cols)
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    corr = X.corr().values
    r2 = np.nan_to_num(corr ** 2)
    dist = np.clip(1 - r2, 0, 1)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="complete")
    labels = fcluster(Z, 1 - r2_threshold, criterion="distance")

    feature_names = list(X.columns)
    df = pd.DataFrame({"feature": feature_names, "cluster": labels})
    group_names, group_cols = [], []
    for c in sorted(df["cluster"].unique()):
        members = df.loc[df["cluster"] == c, "feature"].tolist()
        group_names.append(f"LD_corr_{members[0]}_n{len(members)}")
        group_cols.append(members)
    return group_names, group_cols


def _calculate_d_prime(X: np.ndarray) -> np.ndarray:
    """Vectorised D' matrix for a (n_samples, n_snps) 0/1/2 genotype array."""
    X_hap = X / 2.0
    p = np.mean(X_hap, axis=0)
    q = 1.0 - p
    p11 = (X_hap.T @ X_hap) / X.shape[0]
    D = p11 - np.outer(p, p)
    Dmax = np.where(D >= 0, np.minimum(np.outer(p, q), np.outer(q, p)),
                    np.minimum(np.outer(p, p), np.outer(q, q)))
    with np.errstate(divide="ignore", invalid="ignore"):
        dp = np.abs(D) / Dmax
        np.fill_diagonal(dp, 1.0)
        dp = np.nan_to_num(dp, nan=0.0)
    return dp


def group_by_plink_blocks(
    X: pd.DataFrame,
    d_prime_threshold: float = 0.9,
) -> tuple[list[str], list[list[str]]]:
    """Define contiguous LD blocks using D' threshold.

    SNP names are expected to contain a parseable position as the last
    underscore-separated token (e.g. ``chr1_12345``).  Columns that do not
    match this format default to position 0 and are placed at the start.

    Parameters
    ----------
    X                : Genotype DataFrame (samples × SNPs).
    d_prime_threshold: Minimum D' to keep two adjacent SNPs in the same block.

    Returns
    -------
    (group_names, group_cols)
    """
    if X.empty or len(X.columns) == 0:
        return [], []

    snp_info = []
    for f in X.columns:
        try:
            pos = int(f.split("_")[-1])
        except (ValueError, IndexError):
            pos = 0
        snp_info.append({"feature": f, "pos": pos})

    df = pd.DataFrame(snp_info, columns=["feature", "pos"]).sort_values("pos")
    sorted_features = df["feature"].tolist()
    X_sorted = X[sorted_features].values
    dp = _calculate_d_prime(X_sorted)

    blocks: list[list[str]] = []
    if not sorted_features:
        return [], []

    current_block = [sorted_features[0]]
    for i in range(1, len(sorted_features)):
        # Check D' against the immediately preceding SNP
        if dp[i, i - 1] >= d_prime_threshold:
            current_block.append(sorted_features[i])
        else:
            blocks.append(current_block)
            current_block = [sorted_features[i]]
    blocks.append(current_block)

    group_names = [f"LD_block_{b[0]}_n{len(b)}" for b in blocks]
    return group_names, blocks

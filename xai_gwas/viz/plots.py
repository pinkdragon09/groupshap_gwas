"""Plotting utilities for XAI-GWAS results.

All functions return (fig, ax) or (fig, axes) so callers can further
customise the plot or save with fig.savefig().
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def manhattan_plot(
    gwas_df: pd.DataFrame,
    chrom_col: str = "CHR",
    pos_col: str = "BP",
    pval_col: str = "P",
    snp_col: str | None = "SNP",
    sig_line: float = 5e-8,
    sug_line: float = 1e-5,
    title: str = "Manhattan Plot",
    figsize: tuple[int, int] = (14, 4),
    highlight_snps: list[str] | None = None,
):
    """Standard Manhattan plot from a GWAS summary statistics DataFrame.

    Parameters
    ----------
    gwas_df      : DataFrame with chromosome, position, and p-value columns.
    chrom_col    : Column name for chromosome.
    pos_col      : Column name for base-pair position.
    pval_col     : Column name for p-value.
    snp_col      : Column name for SNP ID (used for hover/highlight labelling).
    sig_line     : Genome-wide significance threshold (default 5e-8).
    sug_line     : Suggestive significance threshold (default 1e-5).
    title        : Plot title.
    figsize      : Figure size in inches.
    highlight_snps: List of SNP IDs to mark with a star.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    df = gwas_df.copy()
    df = df.dropna(subset=[pval_col])
    df[pval_col] = df[pval_col].astype(float).clip(lower=1e-300)
    df["log10p"] = -np.log10(df[pval_col])
    df[chrom_col] = df[chrom_col].astype(str)

    chroms = sorted(df[chrom_col].unique(), key=lambda c: (int(c) if c.isdigit() else 100, c))
    colors = ["#4477AA", "#EE7733"] * (len(chroms) // 2 + 1)
    chrom_color = {c: colors[i] for i, c in enumerate(chroms)}

    # Build cumulative x-axis positions
    chrom_offset: dict[str, int] = {}
    offset = 0
    for c in chroms:
        chrom_offset[c] = offset
        offset += int(df.loc[df[chrom_col] == c, pos_col].max()) + 1_000_000
    df["x"] = df.apply(lambda r: r[pos_col] + chrom_offset[r[chrom_col]], axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    for chrom in chroms:
        sub = df[df[chrom_col] == chrom]
        ax.scatter(sub["x"], sub["log10p"], c=chrom_color[chrom], s=6, alpha=0.7, linewidths=0)

    if highlight_snps and snp_col:
        hl = df[df[snp_col].isin(highlight_snps)]
        ax.scatter(hl["x"], hl["log10p"], c="red", s=30, marker="*", zorder=5, label="highlighted")

    if sig_line:
        ax.axhline(-np.log10(sig_line), color="red", linestyle="--", linewidth=0.8, label=f"p={sig_line:.0e}")
    if sug_line:
        ax.axhline(-np.log10(sug_line), color="blue", linestyle=":", linewidth=0.8, label=f"p={sug_line:.0e}")

    tick_pos = [
        df.loc[df[chrom_col] == c, "x"].median()
        for c in chroms
        if c in df[chrom_col].values
    ]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(chroms, fontsize=7, rotation=45)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("−log₁₀(p)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig, ax


def interaction_heatmap(
    interaction_df: pd.DataFrame,
    group_j_col: str = "group_j",
    group_k_col: str = "group_k",
    value_col: str = "mean_abs_interaction",
    title: str = "Pairwise Shapley Interaction Index",
    figsize: tuple[int, int] | None = None,
    cmap: str = "viridis",
    log_scale: bool = False,
):
    """Heatmap of pairwise Shapley Interaction Indices.

    Parameters
    ----------
    interaction_df : Output of grouped_interaction_all() — long-form DataFrame.
    group_j_col    : Column name for first group in each pair.
    group_k_col    : Column name for second group in each pair.
    value_col      : Column to visualise.
    log_scale      : If True, take log10 of absolute values before plotting.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt

    groups = sorted(set(interaction_df[group_j_col]) | set(interaction_df[group_k_col]))
    n = len(groups)
    idx = {g: i for i, g in enumerate(groups)}
    mat = np.full((n, n), np.nan)

    for _, row in interaction_df.iterrows():
        i, j = idx[row[group_j_col]], idx[row[group_k_col]]
        v = float(row[value_col])
        if log_scale:
            v = np.log10(max(abs(v), 1e-300))
        mat[i, j] = v
        mat[j, i] = v

    fs = figsize or (max(5, n // 2), max(5, n // 2))
    fig, ax = plt.subplots(figsize=fs)
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(groups, rotation=90, fontsize=7)
    ax.set_yticklabels(groups, fontsize=7)
    fig.colorbar(im, ax=ax, label=("log₁₀ |I|" if log_scale else value_col))
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def ale_plot(
    ale_df: pd.DataFrame,
    feature_name: str = "feature",
    x_col: str = "x",
    ale_col: str = "ale",
    n_obs_col: str = "n_obs",
    title: str | None = None,
    figsize: tuple[int, int] = (7, 4),
):
    """ALE (Accumulated Local Effects) line plot for a single feature.

    Parameters
    ----------
    ale_df       : Output of ale_effect() — DataFrame with x, ale, n_obs.
    feature_name : Feature name for axis label.
    title        : Plot title (defaults to 'ALE: <feature_name>').

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ale_df[x_col], ale_df[ale_col], color="steelblue", linewidth=1.8)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")

    # Rug plot scaled by observation counts
    max_obs = ale_df[n_obs_col].max()
    if max_obs > 0:
        rug_h = (ale_df[ale_col].max() - ale_df[ale_col].min()) * 0.04
        for _, row in ale_df.iterrows():
            height = rug_h * row[n_obs_col] / max_obs
            ax.vlines(row[x_col], -abs(rug_h), -abs(rug_h) + height, color="black", alpha=0.4, linewidth=0.8)

    ax.set_xlabel(feature_name)
    ax.set_ylabel("ALE")
    ax.set_title(title or f"ALE: {feature_name}")
    fig.tight_layout()
    return fig, ax


def shapley_barplot(
    summary_df: pd.DataFrame,
    group_col: str = "group",
    value_col: str = "mean_abs_phi",
    err_col: str | None = "sd_abs_phi",
    top_n: int = 20,
    title: str = "Grouped Shapley Values",
    figsize: tuple[int, int] = (8, 5),
):
    """Horizontal bar chart of grouped Shapley values.

    Parameters
    ----------
    summary_df : Output of summarize_grouped_shapley().
    top_n      : Show only the top N groups by |φ|.
    err_col    : Column for error bars (SD); set to None to omit.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt

    df = summary_df.head(top_n).copy()
    df = df.iloc[::-1]  # largest at top in horizontal bar

    fig, ax = plt.subplots(figsize=figsize)
    xerr = df[err_col].values if err_col and err_col in df.columns else None
    ax.barh(df[group_col], df[value_col], xerr=xerr, color="steelblue",
            ecolor="black", capsize=3, alpha=0.85)
    ax.set_xlabel(f"|φ| ({value_col})")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax

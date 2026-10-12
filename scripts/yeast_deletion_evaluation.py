#!/usr/bin/env python3
"""Evaluate frozen Bloom rankings against external, directly assayed deletion pairs.

This module never fits models or changes the historical ranks. Window-level
support means at least one significant direct gene pair in the mapping window;
it is not a causal SNP label. The null samples candidate pairs without replacement
within opportunity strata. Related windows/shared genes can violate exchangeability,
so its p-values and BH values are exploratory, not independent-fold validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


THRESHOLDS = ("lenient", "intermediate", "stringent")
DEFAULT_METHODS = ("rf", "kan", "marginal_product")


def bh_adjust(pvalues):
    """Benjamini-Hochberg adjustment, leaving unavailable values unavailable."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return out
    if np.any((p[valid] < 0) | (p[valid] > 1)):
        raise ValueError("P-values must be in [0, 1].")
    order = valid[np.argsort(p[valid], kind="stable")]
    adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
    out[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return out


def _boolean(series):
    if series.isna().any():
        raise ValueError(f"Missing boolean labels in {series.name}; use screened count for missingness.")
    mapped = series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if mapped.isna().any():
        raise ValueError(f"Invalid boolean labels in {series.name}.")
    return mapped.astype(bool)


def opportunity_strata(frame):
    """Coarse fixed opportunity bins, plus optional deletion degree and chromosome.

    Gene-pair counts use bins 1, 2-3, 4-7, 8-15, etc. Candidate and screened
    counts both enter, conditioning on interval size and screening opportunity.
    Optional assay counts also enter in log2 bins, distinguishing windows assayed
    once from windows with repeated temperatures/reciprocal crosses. Degree uses
    empirical tertiles in the common assayed universe. Edges are never chosen using
    method ranks. Optional degree_score is assayed degree; stringent_degree_score
    intentionally conditions on connectivity in the externally supported network.
    The latter is outcome-derived external information, so the null asks whether
    support exceeds reference-hub opportunity and remains exploratory. Stratum
    sizes and degenerate nulls are reported explicitly.
    """
    if frame.empty:
        names = ["n_candidate_gene_pairs", "n_screened_gene_pairs"]
        names += ["n_screened_assays"] if "n_screened_assays" in frame else []
        names += ["degree_bin"] if "degree_score" in frame else []
        names += ["stringent_degree_bin"] if "stringent_degree_score" in frame else []
        names += ["same_chromosome"] if "same_chromosome" in frame else []
        return pd.Series(index=frame.index, dtype="str"), names
    columns = {}
    for key in ("n_candidate_gene_pairs", "n_screened_gene_pairs"):
        columns[key] = np.floor(np.log2(frame[key].clip(lower=1))).astype(int).astype(str)
    if "n_screened_assays" in frame:
        columns["n_screened_assays"] = np.floor(np.log2(frame.n_screened_assays.clip(lower=1))).astype(int).astype(str)
    for key, bin_name in (("degree_score", "degree_bin"),
                          ("stringent_degree_score", "stringent_degree_bin")):
        if key not in frame:
            continue
        degree = pd.to_numeric(frame[key], errors="raise")
        if degree.isna().any() or not np.isfinite(degree).all():
            raise ValueError(f"{key} must be finite when provided.")
        if (degree < 0).any():
            raise ValueError(f"{key} must be nonnegative.")
        edges = np.unique(np.quantile(degree, [1 / 3, 2 / 3]))
        columns[bin_name] = pd.Series(np.searchsorted(edges, degree, side="right"), index=frame.index).astype(str)
    if "same_chromosome" in frame:
        columns["same_chromosome"] = _boolean(frame.same_chromosome).astype(str)
    bins = pd.DataFrame(columns, index=frame.index)
    return bins.agg("|".join, axis=1), list(columns)


def _seed(seed, *parts):
    payload = "|".join(map(str, (seed,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _matched_null(universe, top_ids, threshold, permutations, seed):
    """Vectorized stratified draws of support counts in matched random subsets."""
    column = f"supported_{threshold}"
    draws = np.zeros(permutations, dtype=np.int64)
    rng = np.random.default_rng(seed)
    variable_top = 0
    detail = []
    for name, group in universe.groupby("_stratum", sort=True):
        population = len(group)
        successes = int(group[column].sum())
        selected = int(group.pair_id.isin(top_ids).sum())
        if not selected:
            continue
        variable = selected < population and 0 < successes < population
        if variable:
            draws += rng.hypergeometric(successes, population - successes, selected, size=permutations)
            variable_top += selected
        else:
            draws += selected if successes == population else successes if selected == population else 0
        detail.append(dict(stratum=name, population=population, supported=successes,
                           selected=selected, variable=variable))
    return draws, variable_top, detail


def _validate(frame, methods):
    required = {"trait", "fold", "method", "pair_id", "rank", "score",
                "n_candidate_gene_pairs", "n_screened_gene_pairs"}
    required.update(f"supported_{t}" for t in THRESHOLDS)
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")
    result = frame.copy()
    result["pair_id"] = result.pair_id.astype(str)
    result = result[result.method.isin(methods)].copy()
    if result.duplicated(["trait", "fold", "method", "pair_id"]).any():
        raise ValueError("Duplicate candidate pair within a trait/fold/method.")
    integer_columns = ["rank", "n_candidate_gene_pairs", "n_screened_gene_pairs"]
    integer_columns += ["n_screened_assays"] if "n_screened_assays" in result else []
    for key in integer_columns:
        result[key] = pd.to_numeric(result[key], errors="raise")
        if result[key].isna().any() or not np.isfinite(result[key]).all():
            raise ValueError(f"Nonfinite values in {key}.")
        if (result[key] < 0).any() or (result[key] % 1 != 0).any():
            raise ValueError(f"Invalid integer values in {key}.")
    if (result["rank"] < 1).any():
        raise ValueError("Frozen ranks must start at 1.")
    if (result.n_screened_gene_pairs > result.n_candidate_gene_pairs).any():
        raise ValueError("Screened pair count exceeds candidate count.")
    if "n_screened_assays" in result:
        if (result.n_screened_assays < result.n_screened_gene_pairs).any():
            raise ValueError("Screened assay count is smaller than screened gene-pair count.")
        if ((result.n_screened_gene_pairs == 0) & (result.n_screened_assays != 0)).any():
            raise ValueError("Unassayed gene pairs cannot have screened assay opportunity.")
    for threshold in THRESHOLDS:
        key = f"supported_{threshold}"
        result[key] = _boolean(result[key])
        if ((result.n_screened_gene_pairs == 0) & result[key]).any():
            raise ValueError("An unassayed candidate cannot have deletion support.")
    if ((result.supported_stringent & ~result.supported_intermediate) |
        (result.supported_intermediate & ~result.supported_lenient)).any():
        raise ValueError("Deletion significance thresholds must be nested.")
    return result


def evaluate_annotations(frame, ks=(10, 25, 50, 100), n_permutations=1000, seed=42,
                         methods=DEFAULT_METHODS):
    """Return by-fold metrics, descriptive fold summaries, and evaluation metadata.

    Top K means original frozen rank <= K, before excluding missing screening.
    Candidates outside the common method universe are also excluded, never promoted
    into top K. This avoids rewarding an incompletely assayed ranking by moving
    lower-ranked candidates upward. Equal ranks use pair ID only for output order;
    all candidates at a rank boundary are included and ties are reported.
    """
    if n_permutations < 1 or any(k < 1 for k in ks):
        raise ValueError("Positive K and number of permutations are required.")
    frame = _validate(frame, methods)
    rows, universes, null_details = [], [], {}
    annotation_columns = ["n_candidate_gene_pairs", "n_screened_gene_pairs"]
    annotation_columns += [f"supported_{t}" for t in THRESHOLDS]
    annotation_columns += [key for key in ("n_screened_assays", "degree_score", "stringent_degree_score", "same_chromosome", "mapping_radius_bp") if key in frame]
    for (trait, fold), block in frame.groupby(["trait", "fold"], sort=True):
        present = set(block.method)
        if present != set(methods):
            raise ValueError(f"Missing matched methods for {trait} fold {fold}: {set(methods) - present}")
        pair_sets = [set(block.loc[block.method == method, "pair_id"]) for method in methods]
        common = set.intersection(*pair_sets)
        canonical = block[block.method == methods[0]].set_index("pair_id").loc[sorted(common)]
        for method in methods[1:]:
            other = block[block.method == method].set_index("pair_id").loc[sorted(common)]
            if not canonical[annotation_columns].equals(other[annotation_columns]):
                raise ValueError(f"External annotations differ by method for {trait}, fold {fold}.")
        eligible = canonical[canonical.n_screened_gene_pairs > 0].reset_index()
        eligible["_stratum"], strata_columns = opportunity_strata(eligible)
        universe_id = f"{trait}|{fold}"
        universes.append(dict(trait=trait, fold=int(fold), common_pairs=len(common),
                              assayed_pairs=len(eligible), method_pair_counts=dict(zip(methods, map(len, pair_sets))),
                              strata_columns=strata_columns, strata=len(set(eligible._stratum))))
        eligible_ids = set(eligible.pair_id)
        for method in methods:
            ranking = block[block.method == method].sort_values(["rank", "pair_id"], kind="stable")
            for k in ks:
                original_top = ranking[ranking["rank"] <= k]
                common_top = original_top[original_top.pair_id.isin(common)]
                top = common_top[common_top.pair_id.isin(eligible_ids)]
                top_ids = set(top.pair_id)
                top_n = len(top)
                for threshold in THRESHOLDS:
                    supported = int(top[f"supported_{threshold}"].sum())
                    universe_supported = int(eligible[f"supported_{threshold}"].sum())
                    rate = supported / top_n if top_n else np.nan
                    background = universe_supported / len(eligible) if len(eligible) else np.nan
                    draws, variable_top, details = _matched_null(eligible, top_ids, threshold,
                        n_permutations, _seed(seed, trait, fold, k, threshold))
                    pvalue = (1 + int((draws >= supported).sum())) / (n_permutations + 1) if top_n else np.nan
                    cell = f"{universe_id}|{method}|{k}|{threshold}"
                    null_details[cell] = details
                    rows.append(dict(trait=trait, fold=int(fold), method=method, k=int(k), threshold=threshold,
                        original_top_pairs=len(original_top), boundary_ties=int((ranking["rank"] == k).sum()),
                        common_top_pairs=len(common_top), assayed_top_pairs=top_n,
                        screened_top_fraction=top_n / len(common_top) if len(common_top) else np.nan,
                        supported_top_pairs=supported, support_rate=rate, common_assayed_pairs=len(eligible),
                        common_supported_pairs=universe_supported, universe_support_rate=background,
                        support_enrichment=rate / background if background > 0 else np.nan,
                        matched_null_mean_count=float(draws.mean()),
                        matched_null_q025_count=float(np.quantile(draws, .025)),
                        matched_null_q975_count=float(np.quantile(draws, .975)),
                        matched_null_mean_rate=float(draws.mean() / top_n) if top_n else np.nan,
                        matched_variable_top_pairs=variable_top, matched_null_degenerate=bool(np.ptp(draws) == 0),
                        matched_empirical_p=pvalue))
    by_fold = pd.DataFrame(rows)
    if by_fold.empty:
        return by_fold, pd.DataFrame(), dict(universes=universes, null_strata=null_details)
    by_fold["matched_bh_q"] = bh_adjust(by_fold.matched_empirical_p)
    summaries = []
    for (trait, method, k, threshold), group in by_fold.groupby(["trait", "method", "k", "threshold"], sort=True):
        row = dict(trait=trait, method=method, k=int(k), threshold=threshold, folds=len(group),
                   folds_with_assayed_top=int((group.assayed_top_pairs > 0).sum()))
        for key in ("support_rate", "support_enrichment", "screened_top_fraction", "matched_null_mean_rate"):
            row[f"mean_{key}"] = float(group[key].mean())
            row[f"sd_{key}"] = float(group[key].std(ddof=1))
        row.update(supported_top_pairs_by_fold=group.supported_top_pairs.astype(int).tolist(),
                   assayed_top_pairs_by_fold=group.assayed_top_pairs.astype(int).tolist(),
                   matched_empirical_p_by_fold=group.matched_empirical_p.tolist(),
                   matched_bh_q_by_fold=group.matched_bh_q.tolist(),
                   degenerate_null_folds=int(group.matched_null_degenerate.sum()))
        summaries.append(row)
    metadata = dict(schema_version=1, seed=seed, permutations=n_permutations, methods=list(methods), ks=list(ks),
        thresholds=list(THRESHOLDS), universes=universes, null_strata=null_details,
        rank_policy="Original frozen rank <= K; no promotion after screening or common-universe exclusions; rank-boundary ties retained.",
        missingness_policy="No assayed gene pairs means unknown, excluded from support-rate denominator.",
        null_policy="Random candidate subsets without replacement within fixed log2 candidate/screen count bins, optional assay-count log2 bins, assayed-degree and stringent-network-degree tertiles and same-chromosome strata. No method ranks determine covariates or edges; stringent-network degree intentionally conditions on outcome-derived external reference hubs.",
        inference_limit="Exploratory exchangeability-based p-values and global BH adjustment; correlated windows/shared genes and folds limit interpretation. No pooled independent-fold inference.",
        summary_policy="Unweighted descriptive mean and sample SD across folds; repeated candidates are not independent replicates.",
        support_interpretation="At least one supported direct deletion gene pair per mapped locus pair; no causal SNP validation or expected sign agreement.")
    return by_fold, pd.DataFrame(summaries), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.annotations, sep="\t" if args.annotations.name.endswith((".tsv", ".tsv.gz")) else ",")
    rows, summary, metadata = evaluate_annotations(frame, n_permutations=args.n_permutations, seed=args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.outdir / "deletion_support_by_fold.tsv", sep="\t", index=False)
    summary.to_json(args.outdir / "deletion_support_summary.json", orient="records", indent=2)
    (args.outdir / "deletion_evaluation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Evaluated {len(rows)} frozen-rank cells without model fitting.")


if __name__ == "__main__":
    main()

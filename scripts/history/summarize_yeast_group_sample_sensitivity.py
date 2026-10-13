#!/usr/bin/env python3
"""Describe sample-size sensitivity using completed Bloom GroupSHAP arrays.

No models or Monte Carlo explanations are recomputed. Twenty-row subsets reuse
their exact saved local explanation rows; comparisons are descriptive, with no
independent Monte Carlo test, repeated-subset experiment or confidence interval.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
TRAITS = ("Neomycin", "Lactate")
METHODS = ("rf", "kan")
EXPECTED_MODELS = 20


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ids_unique(values, label):
    values = [str(value) for value in values]
    require(len(values) == len(set(values)), f"Duplicate sample IDs in {label}")
    return values


def extract_importance(design, local):
    """Validate identities and return full-cohort and matched-subset importance."""
    groups = [group["name"] for group in design["groups"]]
    require(groups and len(groups) == len(set(groups)), "Group names must be unique and nonempty")
    require(list(local["group_names"].astype(str)) == groups, "Saved group order differs from design")
    ids = {role: ids_unique(values, role) for role, values in design["sample_ids"].items()}
    train, heldout, background = (set(ids[role]) for role in ("train", "heldout", "background"))
    require(not train & heldout, "Training and held-out samples overlap")
    require(background <= train, "Background contains nontraining samples")
    require(set(ids["interaction_train"]) <= train - background,
            "Training interaction samples must exclude the background")
    require(set(ids["interaction_heldout"]) <= heldout, "Held-out subset contains non-held-out samples")
    require(len(ids["interaction_train"]) == len(ids["interaction_heldout"]) == 20,
            "Expected matched subsets of exactly 20 observations")
    vectors, sizes = {}, {}
    for role, full_name, subset_name in (("train", "fulltrain", "train20"),
                                         ("heldout", "allheldout", "heldout20")):
        actual_ids = ids_unique(local[f"{role}_sample_ids"].astype(str), f"saved {role}")
        require(actual_ids == ids[role], f"Saved {role} IDs differ from design")
        require(list(local[f"{role}_interaction_sample_ids"].astype(str)) == ids[f"interaction_{role}"],
                f"Saved {role} interaction subset differs from design")
        phi = local[f"{role}_phi"]
        require(phi.shape == (len(actual_ids), len(groups)), f"Invalid {role} phi shape")
        require(len(actual_ids) > 0 and np.isfinite(phi).all(), f"Nonfinite or empty {role} explanations")
        lookup = {sample: index for index, sample in enumerate(actual_ids)}
        subset = np.array([lookup[sample] for sample in ids[f"interaction_{role}"]])
        vectors[full_name] = np.abs(phi).mean(axis=0)
        vectors[subset_name] = np.abs(phi[subset]).mean(axis=0)
        sizes[full_name], sizes[subset_name] = len(phi), len(subset)
    return groups, vectors, sizes


def compare_vectors(a, b, group_names, top_k=5):
    require(a.shape == b.shape == (len(group_names),), "Importance vectors differ in shape")
    rho = np.nan if np.ptp(a) == 0 or np.ptp(b) == 0 else float(spearmanr(a, b).statistic)
    k = min(top_k, len(group_names))
    # Stable ties resolve in the same saved group order for both sample sets.
    a_top, b_top = (np.argsort(-values, kind="stable")[:k] for values in (a, b))
    shared = len(set(a_top) & set(b_top))
    return dict(spearman_group_importance=rho, top5_group_overlap=shared / k,
                top5_shared_groups=shared, top_k_effective=k,
                top_groups_a=";".join(group_names[i] for i in a_top),
                top_groups_b=";".join(group_names[i] for i in b_top))


def summarize(outdir, allow_partial=False):
    outdir = Path(outdir).resolve()
    manifest_path = outdir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    input_hashes = {str(manifest_path): digest(manifest_path)}
    rows, completed_models = [], []
    comparisons = (("train20", "fulltrain"), ("heldout20", "allheldout"),
                   ("train20", "heldout20"), ("fulltrain", "allheldout"))
    for trait, fold, method in itertools.product(TRAITS, range(1, 6), METHODS):
        fold_dir = outdir / trait / f"fold_{fold}"
        model_dir = fold_dir / method
        receipt_path = model_dir / "COMPLETE.json"
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text())
        require(receipt["fingerprint"] == manifest["fingerprint"], f"Fingerprint mismatch: {model_dir}")
        for name, expected in receipt["outputs_sha256"].items():
            require(digest(model_dir / name) == expected, f"Completed artifact changed: {model_dir / name}")
        design_path, local_path = fold_dir / "design.json", model_dir / "local_explanations.npz"
        for path in (receipt_path, design_path, local_path):
            input_hashes[str(path)] = digest(path)
        design = json.loads(design_path.read_text())
        with np.load(local_path, allow_pickle=False) as local:
            groups, vectors, sizes = extract_importance(design, local)
        completed_models.append(dict(trait=trait, fold=fold, method=method))
        for a, b in comparisons:
            rows.append(dict(trait=trait, fold=fold, method=method, comparison=f"{a}_vs_{b}",
                             sample_set_a=a, sample_set_b=b, n_eval_a=sizes[a], n_eval_b=sizes[b],
                             n_groups=len(groups), **compare_vectors(vectors[a], vectors[b], groups)))
    require(completed_models, "No completed models are available")
    require(allow_partial or len(completed_models) == EXPECTED_MODELS,
            f"Only {len(completed_models)}/{EXPECTED_MODELS} completed models; use --allow-partial explicitly")
    complete_path = outdir / "COMPLETE.json"
    parent_complete = complete_path.exists() and len(completed_models) == EXPECTED_MODELS
    require(allow_partial or parent_complete, "Runner completion receipt is absent; use --allow-partial explicitly")
    if parent_complete:
        complete = json.loads(complete_path.read_text())
        require(complete["fingerprint"] == manifest["fingerprint"] and complete["models"] == EXPECTED_MODELS,
                "Runner completion receipt does not match manifest")
        input_hashes[str(complete_path)] = digest(complete_path)
    # Keep partial outputs outside the runner's root-level *.tsv receipt scope.
    report_dir = outdir if parent_complete else outdir / ".partial-report-checks"
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["trait", "method", "comparison"], sort=False).agg(
        n_folds=("fold", "count"), n_defined_spearman=("spearman_group_importance", "count"),
        spearman_mean=("spearman_group_importance", "mean"),
        spearman_sd=("spearman_group_importance", "std"),
        top5_overlap_mean=("top5_group_overlap", "mean"),
        top5_overlap_sd=("top5_group_overlap", "std"),
        n_eval_a_min=("n_eval_a", "min"), n_eval_a_max=("n_eval_a", "max"),
        n_eval_b_min=("n_eval_b", "min"), n_eval_b_max=("n_eval_b", "max")).reset_index()
    for path, expected in input_hashes.items():
        require(digest(path) == expected, f"Source changed while summarizing: {path}")
    output_paths = [report_dir / "group_sample_sensitivity_by_fold.tsv", report_dir / "group_sample_sensitivity_summary.tsv"]
    for data, path in zip((frame, summary), output_paths):
        data.to_csv(path, sep="\t", index=False)
    provenance = dict(
        status="complete" if parent_complete else "partial",
        built_at_utc=datetime.now(timezone.utc).isoformat(), models=len(completed_models),
        expected_models=EXPECTED_MODELS, completed_models=completed_models,
        explanation_source="Exact saved local GroupSHAP rows; no recomputation or new Monte Carlo draws",
        importance="Mean absolute local phi by saved chromosome group, in fold training-target SD",
        subset_selection="One fixed 20-row training subset excluding background and one fixed historical held-out subset per fold",
        interpretation="Descriptive single-subset sensitivity. Shared-row estimates are dependent. Fold SD is descriptive, not a confidence interval; folds share training samples. No significance test or stability guarantee.",
        tie_policy="Stable saved group order; top5 denominator min(5, number of groups)",
        undefined_spearman="Missing when either importance vector is constant",
        input_sha256=input_hashes, code_sha256={str(Path(__file__).resolve()): digest(__file__)},
        output_sha256={path.name: digest(path) for path in output_paths})
    (report_dir / "group_sample_sensitivity_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(f"Saved sensitivity comparisons for {len(completed_models)}/{EXPECTED_MODELS} models ({provenance['status']}).")
    return frame, summary, provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs/yeast_bloom_training_groupshap")
    parser.add_argument("--allow-partial", action="store_true", help="Explicitly allow an incomplete diagnostic snapshot")
    args = parser.parse_args()
    summarize(args.outdir, args.allow_partial)

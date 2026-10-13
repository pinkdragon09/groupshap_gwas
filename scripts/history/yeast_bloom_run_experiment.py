"""End-to-end Bloom epistasis-recovery experiment.

For each phenotype:
  1. Load the trained RF + filtered (X, y) from outputs/yeast_bloom/<pheno>/.
  2. Pick top-N markers by marginal QTL p-value.
  3. Sample bg/eval rows.
  4. Compute pairwise Shapley interaction (fast batched implementation).
  5. Map published interactions (Bloom 2015 SuppData4) to the closest
     filtered marker on each chromosome; restrict to pairs where BOTH
     loci map to a top-N marker.
  6. Compute recall@K for K in {10, 25, 50, 100} for two methods:
       (a) interaction-Shapley ranking
       (b) marginal-product baseline: rank pairs by min(-log10 p_j +
           -log10 p_k); equivalent to ranking by |p_j * p_k| ascending.
  7. Save: pairwise_interactions.csv, recall_at_k.csv, panel_summary.json.
  8. Run chromosome-level grouped Shapley.

CLI: ``python3 yeast_bloom_run_experiment.py <pheno1> [<pheno2> ...]``
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from load_yeast_bloom import load_interactions_2015  # noqa: E402
from yeast_bloom_fast_interaction import pairwise_interaction_fast  # noqa: E402

OUT = REPO / "outputs" / "yeast_bloom"

# Mapping from interaction-list trait names to Bloom 2013 phenotype columns.
TRAIT_TO_PHENO = {
    "Neomycin": "Neomycin",
    "Lactate": "Lactate",
    "YPD": "YPD",
    "Formamide": "Formamide",
    "Raffinose": "Raffinose",
    "Lactose": "Lactose",
    "Zeocin": "Zeocin",
    "CobaltChloride": "Cobalt_Chloride",
    "Ethanol": "Ethanol",
    "Diamide": "Diamide",
    "Menadione": "Menadione",
    "Xylose": "Xylose",
    "YNB": "YNB",
    "MagnesiumChloride": "Magnesium_Chloride",
    "Trehalose": "Trehalose",
}
PHENO_TO_TRAIT = {v: k for k, v in TRAIT_TO_PHENO.items()}


def _to_chrr_label(s: str) -> str:
    """Standardize chromosome labels: 'chr1' / 'chrI' -> 'chrI' (Roman)."""
    s = str(s).strip()
    if s.startswith("chr"):
        rest = s[3:]
        if rest.isdigit():
            roman = _arabic_to_roman(int(rest))
            return f"chr{roman}"
        return s
    return s


_ROMAN = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def _arabic_to_roman(n: int) -> str:
    out = []
    for v, sym in _ROMAN:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def _roman_to_arabic(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    s = s.upper()
    n = 0
    prev = 0
    for ch in reversed(s):
        v = vals.get(ch, 0)
        if v < prev:
            n -= v
        else:
            n += v
        prev = v
    return n


def _normalize_chrom(s: str) -> str:
    """Normalize a chromosome label to 'chrI', 'chrII', ..., 'chrXVI'."""
    s = str(s).strip()
    if s.startswith("chr"):
        rest = s[3:]
        if rest.isdigit():
            return f"chr{_arabic_to_roman(int(rest))}"
        return f"chr{rest.upper()}"
    if s.isdigit():
        return f"chr{_arabic_to_roman(int(s))}"
    return f"chr{s.upper()}"


def select_top_markers(
    marker_meta: pd.DataFrame, n_top: int,
    min_distance_bp: int = 100_000,
) -> pd.DataFrame:
    """Return top-N markers by marginal p, LD-pruned across the genome.

    Greedy distance pruning: walk markers in ascending-p order and admit
    each marker iff it is at least ``min_distance_bp`` from every
    already-admitted marker on the same chromosome. This avoids the
    pathology where a single mega-QTL (e.g. chrXIII for Neomycin) absorbs
    the entire panel and zero published epistasis pairs land in panel.

    With ``min_distance_bp=100_000`` and the yeast 12-Mb genome divided
    across 16 chromosomes, the pruned panel typically spans 12-15
    chromosomes, which is what makes recall@K against published
    cross-chromosome epistatic pairs informative.
    """
    # Drop markers with unparseable IDs (indels with multi-base alleles).
    valid = marker_meta.dropna(subset=["chr", "pos"]).copy()
    sorted_m = valid.sort_values("marginal_pvalue", ascending=True)
    chosen: list[int] = []
    chosen_records: dict[str, list[int]] = {}
    for _, row in sorted_m.iterrows():
        chrom = str(row["chr"])
        pos = int(row["pos"])
        prior_positions = chosen_records.get(chrom, [])
        if any(abs(pos - p) < min_distance_bp for p in prior_positions):
            continue
        chosen.append(int(row["filtered_index"]))
        chosen_records.setdefault(chrom, []).append(pos)
        if len(chosen) >= n_top:
            break
    panel = sorted_m.set_index("filtered_index").loc[chosen].reset_index()
    return panel.reset_index(drop=True)


def map_published_interactions_to_panel(
    pub_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    distance_kb: int = 10,
) -> pd.DataFrame:
    """For each published interaction, find the closest top-N panel marker
    on each locus. A pair is "in panel" iff BOTH loci map within distance_kb
    of a panel marker.
    """
    panel = panel_df.copy()
    panel["chr_norm"] = panel["chr"].map(_normalize_chrom)
    panel["pos"] = panel["pos"].astype(int)

    rows = []
    for _, r in pub_df.iterrows():
        q1c = _normalize_chrom(r["Q1_chr"])
        q2c = _normalize_chrom(r["Q2_chr"])
        q1p = int(r["Q1_pos"])
        q2p = int(r["Q2_pos"])
        # Locus 1
        cand1 = panel[panel["chr_norm"] == q1c]
        if cand1.empty:
            j_idx, j_dist, j_id = None, None, None
        else:
            d = (cand1["pos"] - q1p).abs()
            i = d.idxmin()
            j_dist = int(d.loc[i])
            j_idx = int(cand1.loc[i, "filtered_index"])
            j_id = cand1.loc[i, "marker_id"]
        # Locus 2
        cand2 = panel[panel["chr_norm"] == q2c]
        if cand2.empty:
            k_idx, k_dist, k_id = None, None, None
        else:
            d = (cand2["pos"] - q2p).abs()
            i = d.idxmin()
            k_dist = int(d.loc[i])
            k_idx = int(cand2.loc[i, "filtered_index"])
            k_id = cand2.loc[i, "marker_id"]
        in_panel = (
            j_idx is not None and k_idx is not None
            and j_dist <= distance_kb * 1000
            and k_dist <= distance_kb * 1000
            and j_idx != k_idx
        )
        rows.append({
            "pub_index": int(_.real if hasattr(_, "real") else 0),
            "trait": r.get("trait", ""),
            "Q1_chr": q1c, "Q1_pos": q1p,
            "Q2_chr": q2c, "Q2_pos": q2p,
            "panel_j_idx": j_idx, "panel_k_idx": k_idx,
            "panel_j_id": j_id, "panel_k_id": k_id,
            "panel_j_dist_bp": j_dist, "panel_k_dist_bp": k_dist,
            "in_panel": in_panel,
            "LOD": r.get("LOD", None),
            "lm_coefficient": r.get("lm_coefficient", None),
        })
    return pd.DataFrame(rows)


def compute_recall_at_k(
    inter_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    pub_in_panel: pd.DataFrame,
    Ks: list[int],
) -> pd.DataFrame:
    """Recall@K for two methods: interaction-Shapley vs marginal-product baseline.

    Ground truth: set of unordered (j_idx, k_idx) pairs from pub_in_panel
    with in_panel=True.

    Predictions:
      Shapley: rank by mean_abs_interaction descending.
      Baseline: rank by sum -log10 p_j + -log10 p_k descending (equivalent
                to ranking by |p_j * p_k| ascending).
    """
    # Ground-truth pair set as unordered frozensets of indices.
    gt_pairs = set()
    for _, r in pub_in_panel[pub_in_panel["in_panel"]].iterrows():
        a = int(r["panel_j_idx"])
        b = int(r["panel_k_idx"])
        gt_pairs.add(frozenset((a, b)))
    n_gt = len(gt_pairs)

    # Shapley ranking (already sorted desc by mean_abs_interaction)
    shap_sorted = inter_df.sort_values(
        "mean_abs_interaction", ascending=False).reset_index(drop=True)

    # Baseline ranking: marginal-product = -log10(p_j) + -log10(p_k), descending
    panel_neglog = dict(zip(panel_df["filtered_index"].astype(int),
                            panel_df["neg_log10_p"].astype(float)))
    base = inter_df.copy()
    base["neglog_sum"] = (
        base["j_idx"].astype(int).map(panel_neglog)
        + base["k_idx"].astype(int).map(panel_neglog)
    )
    base_sorted = base.sort_values("neglog_sum", ascending=False).reset_index(drop=True)

    rows = []
    for K in Ks:
        for method, df in [("shapley_interaction", shap_sorted),
                           ("marginal_product_baseline", base_sorted)]:
            top = df.head(K)
            top_pairs = {frozenset((int(j), int(k))) for j, k in
                         zip(top["j_idx"], top["k_idx"])}
            tp = len(top_pairs & gt_pairs)
            recall = tp / n_gt if n_gt > 0 else float("nan")
            precision = tp / K if K > 0 else float("nan")
            rows.append({
                "k": K,
                "method": method,
                "recall": recall,
                "precision": precision,
                "n_predicted": len(top_pairs),
                "n_ground_truth_in_panel": n_gt,
                "tp": tp,
            })
    return pd.DataFrame(rows)


def chromosome_level_shapley(
    rf, X_filtered: np.ndarray, marker_meta: pd.DataFrame,
    bg_idx: np.ndarray, eval_idx: np.ndarray,
    mc_samples: int, random_state: int,
) -> pd.DataFrame:
    """Compute grouped Shapley with one group per yeast chromosome.

    For each eval point and each MC permutation, build all 17 cumulative
    pinning states (empty, +1 chrom, +2 chroms, ..., +all 16 chroms) into
    one (17 * n_bg, p) batch and run ONE rf.predict; then take pairwise
    differences to get per-chromosome Shapley contributions for that perm.
    """
    chrom = marker_meta["chr"].map(_normalize_chrom).to_numpy()
    chroms = sorted(np.unique(chrom).tolist(),
                    key=lambda c: _roman_to_arabic(c[3:]))
    chrom_to_indices = {c: np.where(chrom == c)[0] for c in chroms}
    m = len(chroms)
    n_bg = len(bg_idx)
    bg = X_filtered[bg_idx].astype(np.float32)
    ev = X_filtered[eval_idx].astype(np.float32)
    n_eval = ev.shape[0]
    p = bg.shape[1]

    # Restore RF parallelism for chromosome Shapley (no outer loky here)
    if hasattr(rf, "n_jobs"):
        rf.n_jobs = -1

    phi = np.zeros((n_eval, m), dtype=np.float64)
    n_states = m + 1
    Xb = np.empty((n_states * n_bg, p), dtype=np.float32)

    for e_i in range(n_eval):
        rng = np.random.default_rng(random_state + e_i + 99991)
        eval_x = ev[e_i]
        for s_i in range(mc_samples):
            perm = rng.permutation(m)
            # Build all 17 cumulative pinning states into one batch.
            # State 0 = empty (just bg); state t (1..16) pins chroms[perm[0..t)].
            Xb[0:n_bg] = bg
            for t in range(1, n_states):
                # Reuse previous state
                Xb[t * n_bg:(t + 1) * n_bg] = Xb[(t - 1) * n_bg:t * n_bg]
                cols = chrom_to_indices[chroms[perm[t - 1]]]
                Xb[t * n_bg:(t + 1) * n_bg, cols] = eval_x[cols]
            preds = rf.predict(Xb)  # shape (n_states * n_bg,)
            v = preds.reshape(n_states, n_bg).mean(axis=1)  # (17,)
            # Pairwise differences -> contribution of each chromosome by perm position
            d = np.diff(v)  # (16,) where d[t] = v(state t+1) - v(state t)
            for t in range(m):
                phi[e_i, perm[t]] += d[t]
        phi[e_i] /= mc_samples

    rows = []
    for c_idx, c in enumerate(chroms):
        v = phi[:, c_idx]
        rows.append({
            "chromosome": c,
            "mean_phi": float(v.mean()),
            "mean_abs_phi": float(np.mean(np.abs(v))),
            "sd_abs_phi": float(np.std(np.abs(v))),
            "n_markers_in_chrom": int(len(chrom_to_indices[c])),
            "n_eval_points": int(n_eval),
        })
    return pd.DataFrame(rows).sort_values("mean_abs_phi", ascending=False).reset_index(drop=True)


def run_phenotype(
    phenotype: str,
    n_top: int = 100,
    n_bg: int = 15,
    n_eval: int = 15,
    mc_samples: int = 30,
    distance_kb: int = 10,
    chrom_mc: int = 50,
    random_state: int = 42,
    panel_prune_bp: int = 100_000,
) -> dict:
    out_dir = OUT / phenotype
    if not (out_dir / "model.joblib").exists():
        raise FileNotFoundError(
            f"{out_dir / 'model.joblib'} missing. Run yeast_bloom_train_rf.py first."
        )
    rf = joblib.load(out_dir / "model.joblib")
    X_filtered = np.load(out_dir / "X_filtered.npy")
    marker_meta = pd.read_parquet(out_dir / "marker_meta.parquet")
    n_samples_total, p = X_filtered.shape

    panel = select_top_markers(marker_meta, n_top, min_distance_bp=panel_prune_bp)
    panel.to_parquet(out_dir / "panel_topN.parquet")
    print(f"[panel] {len(panel)} markers across {panel.chr.nunique()} chroms "
          f"(prune={panel_prune_bp} bp); chrom counts: "
          f"{dict(sorted(panel.chr.value_counts().to_dict().items()))}", flush=True)

    rng = np.random.default_rng(random_state)
    bg_idx = rng.choice(n_samples_total, n_bg, replace=False)
    eval_idx = rng.choice(np.setdiff1d(np.arange(n_samples_total), bg_idx),
                          n_eval, replace=False)
    np.save(out_dir / "bg_idx.npy", bg_idx)
    np.save(out_dir / "eval_idx.npy", eval_idx)

    cand_filt_idx = panel["filtered_index"].astype(int).to_numpy()
    cand_labels = panel["marker_id"].tolist()

    print(f"\n=== {phenotype}: pairwise interaction sweep ===", flush=True)
    print(f"  N={n_top} markers, {n_top * (n_top - 1) // 2} pairs", flush=True)
    print(f"  bg={n_bg}, eval={n_eval}, mc={mc_samples}", flush=True)
    t0 = time.time()
    inter_df = pairwise_interaction_fast(
        rf, X_filtered[bg_idx], X_filtered[eval_idx],
        cand_filt_idx, cand_labels,
        n_samples=mc_samples, random_state=random_state, n_jobs=-1,
    )
    sweep_time = time.time() - t0
    inter_df.to_csv(out_dir / "pairwise_interactions.csv", index=False)

    # Map published interactions to panel
    pub_all = load_interactions_2015()
    trait_name = PHENO_TO_TRAIT.get(phenotype, phenotype.replace("_", ""))
    pub = pub_all[pub_all["trait"] == trait_name].copy().reset_index(drop=True)
    pub_mapped = map_published_interactions_to_panel(
        pub, panel, distance_kb=distance_kb,
    )
    pub_mapped.to_csv(out_dir / "published_interactions_mapped.csv", index=False)
    n_pub_total = len(pub_mapped)
    n_pub_in_panel = int(pub_mapped["in_panel"].sum())

    # Recall@K
    Ks = [10, 25, 50, 100]
    recall_df = compute_recall_at_k(inter_df, panel, pub_mapped, Ks)
    recall_df.to_csv(out_dir / "recall_at_k.csv", index=False)

    # Chromosome-level grouped Shapley
    print(f"\n=== {phenotype}: chromosome-level grouped Shapley ===", flush=True)
    t1 = time.time()
    chrom_df = chromosome_level_shapley(
        rf, X_filtered, marker_meta, bg_idx, eval_idx,
        mc_samples=chrom_mc, random_state=random_state,
    )
    chrom_time = time.time() - t1
    chrom_df.to_csv(out_dir / "chromosome_shapley.csv", index=False)

    summary = {
        "phenotype": phenotype,
        "trait_in_pub_list": trait_name,
        "n_top_markers": n_top,
        "n_pairs_evaluated": int(len(inter_df)),
        "n_published_interactions_total": int(n_pub_total),
        "n_published_in_panel": n_pub_in_panel,
        "panel_distance_threshold_kb": distance_kb,
        "n_bg": n_bg, "n_eval": n_eval,
        "mc_samples_pairwise": mc_samples,
        "mc_samples_chromosome": chrom_mc,
        "random_state": random_state,
        "sweep_seconds": sweep_time,
        "chrom_seconds": chrom_time,
        "recall_at_k": recall_df.to_dict(orient="records"),
        "top5_chromosomes_by_mean_abs_phi": chrom_df.head(5).to_dict(orient="records"),
    }
    (out_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Recall@K ===")
    print(recall_df.to_string(index=False))
    print("\n=== Top chromosomes ===")
    print(chrom_df.head(5).to_string(index=False))
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("phenotypes", nargs="+", help="phenotypes to run")
    parser.add_argument("--n-top", type=int, default=100)
    parser.add_argument("--n-bg", type=int, default=15)
    parser.add_argument("--n-eval", type=int, default=15)
    parser.add_argument("--mc-samples", type=int, default=30)
    parser.add_argument("--chrom-mc", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distance-kb", type=int, default=10)
    parser.add_argument("--panel-prune-kb", type=int, default=100,
                        help="LD-prune panel: min bp distance between admitted "
                             "top-N markers on the same chromosome (default 100 kb)")
    args = parser.parse_args()
    summaries = []
    for p in args.phenotypes:
        s = run_phenotype(
            p, n_top=args.n_top, n_bg=args.n_bg, n_eval=args.n_eval,
            mc_samples=args.mc_samples, chrom_mc=args.chrom_mc,
            random_state=args.seed, distance_kb=args.distance_kb,
            panel_prune_bp=args.panel_prune_kb * 1000,
        )
        summaries.append(s)
    print("\n=== ALL DONE ===")
    print(json.dumps(summaries, indent=2))

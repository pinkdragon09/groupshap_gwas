"""Sanity-check helper: print summary of one phenotype's experiment outputs."""

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "yeast_bloom"


def inspect(phenotype: str) -> None:
    d = OUT / phenotype
    print(f"\n=== {phenotype} ===")
    if (d / "model_meta.json").exists():
        m = json.loads((d / "model_meta.json").read_text())
        print(f"n={m['n_samples']} markers={m['n_markers_post_maf']} CV_R2={m['mean_cv_r2']:.3f}")
    if (d / "experiment_summary.json").exists():
        s = json.loads((d / "experiment_summary.json").read_text())
        print(f"panel size: 100 (target), pairs evaluated: {s['n_pairs_evaluated']}")
        print(f"pub interactions: total={s['n_published_interactions_total']}, in panel={s['n_published_in_panel']}")
        print(f"sweep: {s['sweep_seconds']:.0f}s, chrom: {s['chrom_seconds']:.0f}s")
        print("\nRecall@K:")
        for r in s["recall_at_k"]:
            print(f"  K={r['k']:3d}  {r['method']:30s}  recall={r['recall']:.3f}  precision={r['precision']:.3f}  tp={r['tp']}/{r['n_ground_truth_in_panel']}")
        print("\nTop-5 chromosomes by mean_abs_phi:")
        for r in s["top5_chromosomes_by_mean_abs_phi"]:
            print(f"  {r['chromosome']:8s}  mean_abs_phi={r['mean_abs_phi']:.4f}  n_markers={r['n_markers_in_chrom']}")
    if (d / "pairwise_interactions.csv").exists():
        pi = pd.read_csv(d / "pairwise_interactions.csv")
        print(f"\nTop-10 pairs by |I_jk|:")
        print(pi.head(10)[["marker_j", "marker_k", "mean_abs_interaction", "mean_interaction"]].to_string(index=False))


if __name__ == "__main__":
    phens = sys.argv[1:] or ["Neomycin", "Lactate"]
    for p in phens:
        inspect(p)

"""Recall@K, GT-pair |I_jk| diagnostic, and head-to-head figure.

For each phenotype:
  - Loads KAN ``pairwise_interactions.csv`` (this experiment) and the
    RF baseline ``recall_at_k.csv`` + ``pairwise_interactions.csv``.
  - Loads the published-interactions-mapped CSV from the RF baseline
    (same panel + same mapping).
  - Recomputes recall@K for KAN-Shapley using the same logic as the
    RF baseline.
  - Reads RF-Shapley and marginal-product baseline numbers from the
    RF baseline outputs.
  - Saves: recall_at_k.csv (3 methods), gt_pair_iValues.csv (KAN vs
    RF |I_jk| at every ground-truth panel pair), recall_summary.{md,tex},
    figures/kan_vs_rf_recall.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "yeast_bloom_kan"
DATA = ROOT / "data_cache"

PHENOS = ["Neomycin", "Lactate"]
KS = [10, 25, 50, 100]


def kan_panel_index_map(phenotype: str) -> dict[int, int]:
    """Map filtered_index → panel_idx (0..p_panel-1).

    KAN's pairwise_interactions.csv uses panel_idx (0..p_panel-1) in
    j_idx/k_idx columns; RF uses filtered_index. We need a way to align
    them when comparing GT pairs.
    """
    panel_meta = json.loads((OUT / phenotype / "panel_meta.json").read_text())
    panel_cols_in_full = panel_meta["panel_cols_in_full_dim"]
    return {int(fi): int(pi) for pi, fi in enumerate(panel_cols_in_full)}


def load_panel_df(phenotype: str) -> pd.DataFrame:
    return pd.read_csv(DATA / phenotype / "panel_topN.csv")


def load_pub_mapped(phenotype: str) -> pd.DataFrame:
    return pd.read_csv(DATA / phenotype / "published_interactions_mapped.csv")


def load_rf_pairwise(phenotype: str) -> pd.DataFrame:
    """RF baseline pairwise_interactions.csv (j_idx/k_idx are filtered_index)."""
    return pd.read_csv(DATA / phenotype / "pairwise_interactions.csv")


def load_rf_recall(phenotype: str) -> pd.DataFrame:
    return pd.read_csv(DATA / phenotype / "recall_at_k.csv")


def load_kan_pairwise(phenotype: str) -> pd.DataFrame:
    """KAN pairwise_interactions.csv (j_idx/k_idx are panel_idx)."""
    return pd.read_csv(OUT / phenotype / "pairwise_interactions.csv")


def kan_pairwise_with_full_idx(phenotype: str) -> pd.DataFrame:
    """KAN pairwise + the equivalent filtered_index columns for RF alignment."""
    df = load_kan_pairwise(phenotype).copy()
    panel_meta = json.loads((OUT / phenotype / "panel_meta.json").read_text())
    panel_cols_in_full = np.asarray(panel_meta["panel_cols_in_full_dim"], dtype=int)
    df["j_idx_full"] = panel_cols_in_full[df["j_idx"].astype(int).to_numpy()]
    df["k_idx_full"] = panel_cols_in_full[df["k_idx"].astype(int).to_numpy()]
    return df


def kan_recall_at_k(
    inter_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    pub_in_panel: pd.DataFrame,
    Ks: list[int],
    panel_idx_map: dict[int, int],
) -> pd.DataFrame:
    """Compute recall@K using KAN-Shapley ranking only.

    inter_df has j_idx/k_idx in panel-local indexing (0..p-1).
    pub_in_panel has panel_j_idx / panel_k_idx in filtered_index (full).
    We map both to panel-local indexing for the comparison.
    """
    gt_pairs = set()
    for _, r in pub_in_panel[pub_in_panel["in_panel"]].iterrows():
        a_full = int(r["panel_j_idx"])
        b_full = int(r["panel_k_idx"])
        a = panel_idx_map.get(a_full)
        b = panel_idx_map.get(b_full)
        if a is None or b is None:
            continue
        gt_pairs.add(frozenset((a, b)))
    n_gt = len(gt_pairs)

    shap_sorted = inter_df.sort_values(
        "mean_abs_interaction", ascending=False).reset_index(drop=True)
    rows = []
    for K in Ks:
        top = shap_sorted.head(K)
        top_pairs = {frozenset((int(j), int(k))) for j, k in
                     zip(top["j_idx"], top["k_idx"])}
        tp = len(top_pairs & gt_pairs)
        recall = tp / n_gt if n_gt > 0 else float("nan")
        rows.append({
            "k": K,
            "method": "shapley_interaction_kan",
            "recall": recall,
            "precision": tp / K if K > 0 else float("nan"),
            "n_predicted": len(top_pairs),
            "n_ground_truth_in_panel": n_gt,
            "tp": tp,
        })
    return pd.DataFrame(rows)


def gt_pair_diagnostic(
    phenotype: str,
    kan_inter: pd.DataFrame,
    rf_inter: pd.DataFrame,
    pub_in_panel: pd.DataFrame,
    panel_idx_map: dict[int, int],
) -> pd.DataFrame:
    """Per-GT-pair |I_jk| from KAN and RF.

    pub_in_panel uses filtered_index (j,k) as panel_j_idx / panel_k_idx.
    KAN uses panel-local indexing in j_idx/k_idx.
    RF uses filtered_index in j_idx/k_idx.
    """
    rf_lookup: dict[frozenset, dict] = {}
    for _, r in rf_inter.iterrows():
        a, b = int(r["j_idx"]), int(r["k_idx"])
        rf_lookup[frozenset((a, b))] = {
            "rf_mean_abs_I": float(r["mean_abs_interaction"]),
            "rf_mean_I": float(r["mean_interaction"]),
        }

    kan_lookup: dict[frozenset, dict] = {}
    for _, r in kan_inter.iterrows():
        a, b = int(r["j_idx"]), int(r["k_idx"])  # panel-local
        kan_lookup[frozenset((a, b))] = {
            "kan_mean_abs_I": float(r["mean_abs_interaction"]),
            "kan_mean_I": float(r["mean_interaction"]),
        }

    rows = []
    for _, r in pub_in_panel[pub_in_panel["in_panel"]].iterrows():
        a_full = int(r["panel_j_idx"])
        b_full = int(r["panel_k_idx"])
        a_local = panel_idx_map.get(a_full)
        b_local = panel_idx_map.get(b_full)
        rf_vals = rf_lookup.get(frozenset((a_full, b_full)), {})
        if a_local is None or b_local is None:
            kan_vals: dict = {}
        else:
            kan_vals = kan_lookup.get(frozenset((a_local, b_local)), {})
        rows.append({
            "phenotype": phenotype,
            "panel_j_full": a_full, "panel_k_full": b_full,
            "panel_j_local": a_local, "panel_k_local": b_local,
            "panel_j_id": r["panel_j_id"], "panel_k_id": r["panel_k_id"],
            "Q1_chr": r["Q1_chr"], "Q1_pos": r["Q1_pos"],
            "Q2_chr": r["Q2_chr"], "Q2_pos": r["Q2_pos"],
            "kan_mean_abs_I": kan_vals.get("kan_mean_abs_I"),
            "kan_mean_I":     kan_vals.get("kan_mean_I"),
            "rf_mean_abs_I":  rf_vals.get("rf_mean_abs_I"),
            "rf_mean_I":      rf_vals.get("rf_mean_I"),
        })
    return pd.DataFrame(rows)


def merge_recall(phenotype: str) -> pd.DataFrame:
    rf_recall = load_rf_recall(phenotype).rename(
        columns={"method": "method"})
    rf_recall["method"] = rf_recall["method"].replace({
        "shapley_interaction": "shapley_interaction_rf",
    })
    return rf_recall


def main() -> None:
    OUT_FIG = OUT / "figures"
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    per_pheno_summaries = []
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)

    for ax, ph in zip(axes, PHENOS):
        print(f"=== {ph}: assembling recall ===", flush=True)
        panel_df = load_panel_df(ph)
        pub_mapped = load_pub_mapped(ph)
        kan_inter = load_kan_pairwise(ph)
        rf_inter = load_rf_pairwise(ph)
        idx_map = kan_panel_index_map(ph)

        kan_rec = kan_recall_at_k(kan_inter, panel_df, pub_mapped, KS, idx_map)
        rf_rec = merge_recall(ph)
        # Combine
        combined = pd.concat([rf_rec, kan_rec], ignore_index=True, sort=False)
        combined.to_csv(OUT / ph / "recall_at_k.csv", index=False)

        # GT-pair diagnostic
        diag = gt_pair_diagnostic(ph, kan_inter, rf_inter, pub_mapped, idx_map)
        diag.to_csv(OUT / ph / "gt_pair_iValues.csv", index=False)

        # Plot
        for method, lbl, color, marker in [
            ("shapley_interaction_kan",   "KAN-Shapley",        "#1f77b4", "o"),
            ("shapley_interaction_rf",    "RF-Shapley",         "#d62728", "s"),
            ("marginal_product_baseline", "Marginal-product",   "#2ca02c", "^"),
        ]:
            sub = combined[combined["method"] == method].sort_values("k")
            ax.plot(sub["k"], sub["recall"], color=color, marker=marker,
                    label=lbl, linewidth=1.4, markersize=4.5)
        ax.set_xscale("log")
        ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
        ax.set_xlabel("K (top pairs)")
        if ax is axes[0]:
            ax.set_ylabel("Recall")
        ax.set_title(ph, fontsize=10)
        ax.set_ylim(0, max(0.7, combined["recall"].max() * 1.1))
        ax.grid(alpha=0.25, linewidth=0.5)

    axes[0].legend(loc="upper left", fontsize=7, frameon=False)
    fig.suptitle("Bloom 2015 epistasis recovery — KAN vs RF vs marginal-product",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "kan_vs_rf_recall.png", dpi=200,
                bbox_inches="tight")
    fig.savefig(OUT_FIG / "kan_vs_rf_recall.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {OUT_FIG / 'kan_vs_rf_recall.png'}", flush=True)

    # Aggregate summary
    rows = []
    means_per_method: dict[str, list[float]] = {}
    for ph in PHENOS:
        df = pd.read_csv(OUT / ph / "recall_at_k.csv")
        for K in KS:
            row = {"phenotype": ph, "k": K}
            for method, key in [
                ("shapley_interaction_kan", "kan"),
                ("shapley_interaction_rf",  "rf"),
                ("marginal_product_baseline", "base"),
            ]:
                m = df[(df["method"] == method) & (df["k"] == K)]
                if len(m):
                    val = float(m["recall"].iloc[0])
                else:
                    val = float("nan")
                row[f"recall_{key}"] = val
                means_per_method.setdefault((key, K), []).append(val)
            rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT / "recall_summary.csv", index=False)

    # Per-method means across phenotypes for each K
    mean_rows = []
    for K in KS:
        row = {"phenotype": "MEAN", "k": K}
        for key in ("kan", "rf", "base"):
            vals = means_per_method.get((key, K), [])
            row[f"recall_{key}"] = float(np.nanmean(vals)) if vals else float("nan")
        mean_rows.append(row)
    summary_with_mean = pd.concat(
        [summary_df, pd.DataFrame(mean_rows)], ignore_index=True)

    # Markdown summary
    md_lines = ["| phenotype | K | KAN-Shapley | RF-Shapley | Marginal-product |",
                "|---|---:|---:|---:|---:|"]
    for _, r in summary_with_mean.iterrows():
        md_lines.append(
            f"| {r['phenotype']} | {int(r['k'])} | "
            f"{r['recall_kan']:.3f} | {r['recall_rf']:.3f} | {r['recall_base']:.3f} |"
        )
    (OUT / "recall_summary.md").write_text("\n".join(md_lines) + "\n")

    # LaTeX summary (booktabs-style)
    tex_lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Phenotype & $K$ & KAN-Shapley & RF-Shapley & Marginal-product \\\\",
        "\\midrule",
    ]
    for _, r in summary_with_mean.iterrows():
        tex_lines.append(
            f"{r['phenotype']} & {int(r['k'])} & "
            f"{r['recall_kan']:.3f} & {r['recall_rf']:.3f} & "
            f"{r['recall_base']:.3f} \\\\"
        )
    tex_lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "recall_summary.tex").write_text("\n".join(tex_lines) + "\n")

    print("recall_summary.{md,tex,csv} saved")
    print(summary_with_mean.to_string(index=False))


if __name__ == "__main__":
    main()

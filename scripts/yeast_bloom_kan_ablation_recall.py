"""Aggregate recall@K and GT-pair |I_jk| diagnostics across the cross-
shape KAN ablation, then write the summary table and figure.

Reads:
    outputs/yeast_bloom_kan/ablation/<phenotype>/h<H>_g<G>/
        pairwise_interactions.csv
        cv_metrics.json
        panel_meta.json
    outputs/yeast_bloom_kan/<phenotype>/      (baseline C0)
        pairwise_interactions.csv
        cv_metrics.json
        panel_meta.json
        recall_at_k.csv     (already contains marginal-product baseline +
                             RF-Shapley reference rows)

Writes:
    outputs/yeast_bloom_kan/ablation/recall_summary.{md,tex,csv}
    outputs/yeast_bloom_kan/ablation/cv_summary.{md,csv}
    outputs/yeast_bloom_kan/ablation/gt_pair_diagnostic.csv
    outputs/yeast_bloom_kan/ablation/figures/ablation_recall.png
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
ABL = ROOT / "outputs" / "yeast_bloom_kan" / "ablation"
DATA = ROOT / "data_cache"
BASELINE_OUT = ROOT / "outputs" / "yeast_bloom_kan"

PHENOS = ["Neomycin", "Lactate"]
KS = [10, 25, 50, 100]

# Cross-shape configs. C0 is baseline (read from outputs/yeast_bloom_kan).
CONFIGS = [
    ("C0", 8, 5, "baseline"),
    ("C1", 4, 5, "h-narrower"),
    ("C2", 16, 5, "h-wider"),
    ("C3", 8, 3, "g-coarser"),
    ("C4", 8, 10, "g-finer"),
]


def panel_meta_for(phenotype: str, hidden: int, grid: int) -> dict:
    if (hidden, grid) == (8, 5):
        path = BASELINE_OUT / phenotype / "panel_meta.json"
    else:
        path = ABL / phenotype / f"h{hidden}_g{grid}" / "panel_meta.json"
    return json.loads(path.read_text())


def panel_idx_map(phenotype: str, hidden: int, grid: int) -> dict[int, int]:
    """filtered_index -> panel-local idx (0..p_panel-1)."""
    meta = panel_meta_for(phenotype, hidden, grid)
    return {int(fi): int(pi) for pi, fi in enumerate(meta["panel_cols_in_full_dim"])}


def kan_pairwise_for(phenotype: str, hidden: int, grid: int) -> pd.DataFrame:
    if (hidden, grid) == (8, 5):
        return pd.read_csv(BASELINE_OUT / phenotype / "pairwise_interactions.csv")
    return pd.read_csv(ABL / phenotype / f"h{hidden}_g{grid}" / "pairwise_interactions.csv")


def cv_metrics_for(phenotype: str, hidden: int, grid: int) -> dict:
    if (hidden, grid) == (8, 5):
        path = BASELINE_OUT / phenotype / "cv_metrics.json"
    else:
        path = ABL / phenotype / f"h{hidden}_g{grid}" / "cv_metrics.json"
    return json.loads(path.read_text())


def published_for(phenotype: str) -> pd.DataFrame:
    return pd.read_csv(DATA / phenotype / "published_interactions_mapped.csv")


def baseline_recall_for(phenotype: str) -> pd.DataFrame:
    """The baseline RF / marginal-product / KAN recall table (C0 + refs)."""
    return pd.read_csv(BASELINE_OUT / phenotype / "recall_at_k.csv")


def kan_recall_at_k(
    inter_df: pd.DataFrame,
    pub_in_panel: pd.DataFrame,
    Ks: list[int],
    idx_map: dict[int, int],
) -> pd.DataFrame:
    """KAN recall@K — interactions panel-local, GT in filtered_index."""
    gt_pairs = set()
    for _, r in pub_in_panel[pub_in_panel["in_panel"]].iterrows():
        a_full = int(r["panel_j_idx"])
        b_full = int(r["panel_k_idx"])
        a = idx_map.get(a_full)
        b = idx_map.get(b_full)
        if a is None or b is None:
            continue
        gt_pairs.add(frozenset((a, b)))
    n_gt = len(gt_pairs)

    sorted_df = inter_df.sort_values(
        "mean_abs_interaction", ascending=False).reset_index(drop=True)
    rows = []
    for K in Ks:
        top = sorted_df.head(K)
        top_pairs = {frozenset((int(j), int(k))) for j, k in
                     zip(top["j_idx"], top["k_idx"])}
        tp = len(top_pairs & gt_pairs)
        recall = tp / n_gt if n_gt > 0 else float("nan")
        rows.append({
            "k": K,
            "recall": recall,
            "precision": tp / K if K > 0 else float("nan"),
            "n_predicted": len(top_pairs),
            "n_gt_in_panel": n_gt,
            "tp": tp,
        })
    return pd.DataFrame(rows)


def gt_pair_iValues(
    phenotype: str, hidden: int, grid: int,
    inter_df: pd.DataFrame,
    pub_in_panel: pd.DataFrame,
    idx_map: dict[int, int],
) -> pd.DataFrame:
    lookup = {}
    for _, r in inter_df.iterrows():
        a, b = int(r["j_idx"]), int(r["k_idx"])
        lookup[frozenset((a, b))] = float(r["mean_abs_interaction"])
    rows = []
    for _, r in pub_in_panel[pub_in_panel["in_panel"]].iterrows():
        a_full = int(r["panel_j_idx"])
        b_full = int(r["panel_k_idx"])
        a = idx_map.get(a_full)
        b = idx_map.get(b_full)
        if a is None or b is None:
            val = float("nan")
        else:
            val = lookup.get(frozenset((a, b)), float("nan"))
        rows.append({
            "phenotype": phenotype, "hidden": hidden, "grid": grid,
            "panel_j_id": r["panel_j_id"], "panel_k_id": r["panel_k_id"],
            "Q1_chr": r["Q1_chr"], "Q1_pos": r["Q1_pos"],
            "Q2_chr": r["Q2_chr"], "Q2_pos": r["Q2_pos"],
            "kan_mean_abs_I": val,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ABL.mkdir(parents=True, exist_ok=True)
    (ABL / "figures").mkdir(parents=True, exist_ok=True)

    cv_rows = []
    recall_rows = []
    gt_rows = []

    # Reference rows for the marginal-product baseline + RF (per phenotype).
    ref_rows = []
    for ph in PHENOS:
        bl = baseline_recall_for(ph)
        for method, label in [
            ("marginal_product_baseline", "marginal_product"),
            ("shapley_interaction_rf", "rf_shapley"),
        ]:
            sub = bl[bl["method"] == method]
            for _, r in sub.iterrows():
                ref_rows.append({
                    "phenotype": ph,
                    "config": label,
                    "hidden": "—", "grid": "—",
                    "k": int(r["k"]),
                    "recall": float(r["recall"]),
                    "tp": int(r["tp"]),
                    "n_gt_in_panel": int(r["n_ground_truth_in_panel"]),
                })
    ref_df = pd.DataFrame(ref_rows)

    # KAN rows across the 5 cross-shape configs per phenotype.
    for ph in PHENOS:
        pub = published_for(ph)
        for config_name, hidden, grid, role in CONFIGS:
            sub_dir = (
                BASELINE_OUT / ph if (hidden, grid) == (8, 5)
                else ABL / ph / f"h{hidden}_g{grid}"
            )
            cv_path = sub_dir / "cv_metrics.json"
            inter_path = sub_dir / "pairwise_interactions.csv"
            if not (cv_path.exists() and inter_path.exists()):
                print(f"  [SKIP missing] {ph} {config_name} (h={hidden},g={grid}): "
                      f"{sub_dir}", flush=True)
                cv_rows.append({
                    "phenotype": ph, "config": config_name,
                    "hidden": hidden, "grid": grid, "role": role,
                    "cv_mean_r2": float("nan"), "cv_sd_r2": float("nan"),
                    "cv_seconds": float("nan"), "status": "DEFERRED",
                })
                continue

            cv = json.loads(cv_path.read_text())
            inter_df = kan_pairwise_for(ph, hidden, grid)
            idx_map = panel_idx_map(ph, hidden, grid)
            rec_df = kan_recall_at_k(inter_df, pub, KS, idx_map)
            gt_df = gt_pair_iValues(ph, hidden, grid, inter_df, pub, idx_map)

            # CV row
            cv_rows.append({
                "phenotype": ph, "config": config_name,
                "hidden": hidden, "grid": grid, "role": role,
                "cv_mean_r2": float(cv["mean_r2"]),
                "cv_sd_r2": float(cv["sd_r2"]),
                "cv_seconds": float(cv.get("cv_seconds", float("nan"))),
                "status": "DONE",
            })
            # Recall rows
            for _, r in rec_df.iterrows():
                recall_rows.append({
                    "phenotype": ph, "config": config_name,
                    "hidden": hidden, "grid": grid, "role": role,
                    "k": int(r["k"]),
                    "recall": float(r["recall"]),
                    "tp": int(r["tp"]),
                    "n_gt_in_panel": int(r["n_gt_in_panel"]),
                })
            gt_rows.append(gt_df)

    cv_df = pd.DataFrame(cv_rows)
    recall_df = pd.DataFrame(recall_rows)
    gt_df_all = pd.concat(gt_rows, ignore_index=True) if gt_rows else pd.DataFrame()

    cv_df.to_csv(ABL / "cv_summary.csv", index=False)
    recall_df.to_csv(ABL / "recall_summary.csv", index=False)
    ref_df.to_csv(ABL / "recall_reference.csv", index=False)
    if not gt_df_all.empty:
        gt_df_all.to_csv(ABL / "gt_pair_diagnostic.csv", index=False)

    # GT-pair median |I_jk| per (phenotype, config)
    if not gt_df_all.empty:
        gt_summary = (gt_df_all
                      .groupby(["phenotype", "hidden", "grid"], as_index=False)
                      ["kan_mean_abs_I"]
                      .agg(["median", "max", "min", "count"]).reset_index())
    else:
        gt_summary = pd.DataFrame()
    if not gt_summary.empty:
        gt_summary.to_csv(ABL / "gt_pair_summary.csv", index=False)

    # ---------- Markdown summary ----------
    md = ["# KAN cross-shape ablation — recall@K + CV R²",
          "",
          "## CV R² (5-fold, seed=42, identical panel/split as baseline)",
          "",
          "| Phenotype | Config | hidden | grid | role | CV mean R² | CV sd | status |",
          "|---|---|---:|---:|---|---:|---:|---|"]
    for _, r in cv_df.iterrows():
        md.append(
            f"| {r['phenotype']} | {r['config']} | {r['hidden']} | {r['grid']} | "
            f"{r['role']} | {r['cv_mean_r2']:.4f} | {r['cv_sd_r2']:.4f} | {r['status']} |"
        )

    md.append("")
    md.append("## Recall@K (KAN-Shapley)")
    md.append("")
    md.append("| Phenotype | Config | hidden | grid | role | K=10 | K=25 | K=50 | K=100 |")
    md.append("|---|---|---:|---:|---|---:|---:|---:|---:|")
    for ph in PHENOS:
        for cfg, h, g, role in CONFIGS:
            sub = recall_df[(recall_df["phenotype"] == ph)
                            & (recall_df["config"] == cfg)]
            cells = []
            for K in KS:
                row = sub[sub["k"] == K]
                cells.append(f"{float(row['recall'].iloc[0]):.3f}"
                             if len(row) else "—")
            md.append(
                f"| {ph} | {cfg} | {h} | {g} | {role} | "
                f"{cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |"
            )
    md.append("")
    md.append("## Reference (non-KAN methods on the same panel)")
    md.append("")
    md.append("| Phenotype | Method | K=10 | K=25 | K=50 | K=100 |")
    md.append("|---|---|---:|---:|---:|---:|")
    for ph in PHENOS:
        for label in ["marginal_product", "rf_shapley"]:
            sub = ref_df[(ref_df["phenotype"] == ph)
                         & (ref_df["config"] == label)]
            cells = []
            for K in KS:
                row = sub[sub["k"] == K]
                cells.append(f"{float(row['recall'].iloc[0]):.3f}"
                             if len(row) else "—")
            md.append(
                f"| {ph} | {label} | "
                f"{cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |"
            )
    md.append("")
    if not gt_summary.empty:
        md.append("## GT-pair |I_jk| summary (KAN, per cell)")
        md.append("")
        md.append("| Phenotype | hidden | grid | n GT pairs | median |I_jk| | min | max |")
        md.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, r in gt_summary.iterrows():
            md.append(
                f"| {r['phenotype']} | {r['hidden']} | {r['grid']} | "
                f"{int(r['count'])} | {r['median']:.4f} | "
                f"{r['min']:.4f} | {r['max']:.4f} |"
            )
    md.append("")
    md.append("## Configs vs. baseline")
    md.append("")
    md.append("- C0 = (h=8, g=5) baseline — reused from `outputs/yeast_bloom_kan/`.")
    md.append("- C1 = (h=4, g=5) hidden-narrower (sanity check).")
    md.append("- C2 = (h=16, g=5) hidden-wider — load-bearing hypothesis test for Lactate.")
    md.append("- C3 = (h=8, g=3) grid-coarser (sanity check).")
    md.append("- C4 = (h=8, g=10) grid-finer.")

    (ABL / "recall_summary.md").write_text("\n".join(md) + "\n")

    # ---------- LaTeX summary ----------
    tex = ["\\begin{tabular}{llrrlrrrr}",
           "\\toprule",
           "Phenotype & Config & $h$ & $g$ & role & "
           "Recall@10 & Recall@25 & Recall@50 & Recall@100 \\\\",
           "\\midrule"]
    for ph in PHENOS:
        for cfg, h, g, role in CONFIGS:
            sub = recall_df[(recall_df["phenotype"] == ph)
                            & (recall_df["config"] == cfg)]
            cells = []
            for K in KS:
                row = sub[sub["k"] == K]
                cells.append(f"{float(row['recall'].iloc[0]):.3f}"
                             if len(row) else "—")
            tex.append(
                f"{ph} & {cfg} & {h} & {g} & {role} & "
                f"{cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\"
            )
    tex.append("\\midrule")
    for ph in PHENOS:
        for label in ["marginal_product", "rf_shapley"]:
            sub = ref_df[(ref_df["phenotype"] == ph)
                         & (ref_df["config"] == label)]
            cells = []
            for K in KS:
                row = sub[sub["k"] == K]
                cells.append(f"{float(row['recall'].iloc[0]):.3f}"
                             if len(row) else "—")
            tex.append(
                f"{ph} & {label} & --- & --- & ref & "
                f"{cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\"
            )
    tex += ["\\bottomrule", "\\end{tabular}"]
    (ABL / "recall_summary.tex").write_text("\n".join(tex) + "\n")

    # ---------- Figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    color_by_cfg = {
        "C0": "#1f77b4",
        "C1": "#9467bd",
        "C2": "#d62728",
        "C3": "#ff7f0e",
        "C4": "#2ca02c",
    }
    marker_by_cfg = {"C0": "o", "C1": "v", "C2": "*", "C3": "s", "C4": "^"}

    for ax, ph in zip(axes, PHENOS):
        for cfg, h, g, role in CONFIGS:
            sub = recall_df[(recall_df["phenotype"] == ph)
                            & (recall_df["config"] == cfg)].sort_values("k")
            if not len(sub):
                continue
            label = f"{cfg} (h={h},g={g})"
            ms = 9 if cfg == "C2" else 5
            lw = 2.0 if cfg == "C2" else 1.2
            ax.plot(sub["k"], sub["recall"],
                    color=color_by_cfg[cfg], marker=marker_by_cfg[cfg],
                    label=label, linewidth=lw, markersize=ms)
        # marginal-product reference (dashed)
        sub = ref_df[(ref_df["phenotype"] == ph)
                     & (ref_df["config"] == "marginal_product")].sort_values("k")
        ax.plot(sub["k"], sub["recall"], color="#444444", marker="x",
                linestyle="--", label="marginal_product",
                linewidth=1.0, markersize=5)
        ax.set_xscale("log")
        ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
        ax.set_xlabel("K (top pairs)")
        if ax is axes[0]:
            ax.set_ylabel("Recall")
        ax.set_title(ph, fontsize=10)
        ax.set_ylim(0, max(0.8, recall_df["recall"].max() * 1.1))
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].legend(loc="upper left", fontsize=7, frameon=False, ncol=1)
    fig.suptitle("Bloom 2015 KAN ablation — recall@K (load-bearing test: C2 vs C0 on Lactate)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(ABL / "figures" / "ablation_recall.png", dpi=200,
                bbox_inches="tight")
    fig.savefig(ABL / "figures" / "ablation_recall.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {ABL / 'figures' / 'ablation_recall.png'}", flush=True)

    print("\n===== CV summary =====")
    print(cv_df.to_string(index=False))
    print("\n===== Recall summary =====")
    print(recall_df.to_string(index=False))
    if not gt_summary.empty:
        print("\n===== GT-pair |I_jk| summary =====")
        print(gt_summary.to_string(index=False))
    print("\n===== Reference (non-KAN) =====")
    print(ref_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

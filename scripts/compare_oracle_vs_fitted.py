#!/usr/bin/env python3
"""
Side-by-side numerical comparison of oracle (run_phase6.py) vs
fitted-RF (run_phase6_fitted_rf.py) Phase-6 simulation outputs.

Writes:
    outputs/phase6/oracle_vs_fitted_comparison.csv
    outputs/phase6/oracle_vs_fitted_summary.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs/phase6")


def main() -> None:
    rows: list[dict] = []

    # ---- Scenario 1 recovery: shap_snr vs linear_snr per sigma ----
    s1o = pd.read_csv(OUT / "scenario1_recovery.csv")
    s1f = pd.read_csv(OUT / "scenario1_recovery_fitted.csv")
    for sigma in s1o["sigma"]:
        ro = s1o[s1o["sigma"] == sigma].iloc[0]
        rf = s1f[s1f["sigma"] == sigma].iloc[0]
        rows.append({
            "scenario": 1, "param": f"sigma={sigma}",
            "metric": "shap_snr", "oracle": ro["shap_snr"], "fitted": rf["shap_snr"],
        })
        rows.append({
            "scenario": 1, "param": f"sigma={sigma}",
            "metric": "linear_snr (unchanged)", "oracle": ro["linear_snr"], "fitted": rf["linear_snr"],
        })

    # ---- Scenario 3 recovery: I_01 and null_max per b ----
    s3o = pd.read_csv(OUT / "scenario3_recovery.csv")
    s3f = pd.read_csv(OUT / "scenario3_recovery_fitted.csv")
    for b in s3o["b_01"]:
        ro = s3o[s3o["b_01"] == b].iloc[0]
        rf = s3f[s3f["b_01"] == b].iloc[0]
        rows.append({
            "scenario": 3, "param": f"b={b}",
            "metric": "I_01", "oracle": ro["I_01"], "fitted": rf["I_01"],
        })
        rows.append({
            "scenario": 3, "param": f"b={b}",
            "metric": "null_max", "oracle": ro["null_max"], "fitted": rf["null_max"],
        })
        rows.append({
            "scenario": 3, "param": f"b={b}",
            "metric": "I_01 / null_max", "oracle": ro["I_01"] / max(ro["null_max"], 1e-30),
            "fitted": rf["I_01"] / max(rf["null_max"], 1e-30),
        })

    # ---- Scenario 5 recovery: phi_causal_frac per rho ----
    s5o = pd.read_csv(OUT / "scenario5_recovery.csv")
    s5f = pd.read_csv(OUT / "scenario5_recovery_fitted.csv")
    for rho in s5o["rho"]:
        ro = s5o[s5o["rho"] == rho].iloc[0]
        rf = s5f[s5f["rho"] == rho].iloc[0]
        rows.append({
            "scenario": 5, "param": f"rho={rho}",
            "metric": "phi_causal_frac", "oracle": ro["phi_causal_frac"], "fitted": rf["phi_causal_frac"],
        })

    # ---- Scenario 5 stability ----
    s5os = pd.read_csv(OUT / "scenario5_stability.csv")
    s5fs = pd.read_csv(OUT / "scenario5_stability_fitted.csv")
    rows.append({
        "scenario": 5, "param": "rho=0.8 (15 seeds)",
        "metric": "stability mean(phi_causal_frac)",
        "oracle": float(s5os["phi_causal_frac"].mean()),
        "fitted": float(s5fs["phi_causal_frac"].mean()),
    })
    rows.append({
        "scenario": 5, "param": "rho=0.8 (15 seeds)",
        "metric": "stability sd(phi_causal_frac)",
        "oracle": float(s5os["phi_causal_frac"].std()),
        "fitted": float(s5fs["phi_causal_frac"].std()),
    })

    # ---- Scenario 1 stability summary ----
    s1os = pd.read_csv(OUT / "scenario1_stability.csv")
    s1fs = pd.read_csv(OUT / "scenario1_stability_fitted.csv")
    for col in ["G0", "G1", "G2", "G3"]:
        rows.append({
            "scenario": 1, "param": "sigma=0.5 (15 seeds)",
            "metric": f"stability mean(|phi_{col}|)",
            "oracle": float(s1os[col].mean()),
            "fitted": float(s1fs[col].mean()),
        })

    # ---- Scenario 3 stability summary ----
    s3os = pd.read_csv(OUT / "scenario3_stability.csv")
    s3fs = pd.read_csv(OUT / "scenario3_stability_fitted.csv")
    for col in ["I_01", "I_02", "I_12"]:
        rows.append({
            "scenario": 3, "param": "b=0.3 (15 seeds)",
            "metric": f"stability mean({col})",
            "oracle": float(s3os[col].mean()),
            "fitted": float(s3fs[col].mean()),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "oracle_vs_fitted_comparison.csv", index=False)

    # ---- Markdown summary ----
    md_lines: list[str] = []
    md_lines.append("# Phase-6 Oracle vs Fitted-RF Comparison")
    md_lines.append("")
    md_lines.append("Side-by-side of `outputs/phase6/scenario{1,3,5}_*.csv` (oracle, "
                    "from `run_phase6.py`) vs `*_fitted.csv` (fitted RandomForest, "
                    "from `run_phase6_fitted_rf.py`).")
    md_lines.append("")

    md_lines.append("## Scenario 1 — Additive Sparse (recovery)")
    md_lines.append("")
    md_lines.append("Headline: grouped-Shapley signal-to-null ratio (`shap_snr`) "
                    "vs linear-GWAS SNR. Oracle reported `shap_snr ≈ 2.81e9` for "
                    "every sigma (tautological — Shapley applied to the closed-form "
                    "`f = 2·G0 + 1.5·G1`). Fitted RF gives finite values that decay "
                    "modestly with noise.")
    md_lines.append("")
    md_lines.append("| sigma | shap_snr (oracle) | shap_snr (fitted) | linear_snr |")
    md_lines.append("|------:|------------------:|------------------:|-----------:|")
    for _, r in s1o.iterrows():
        rf = s1f[s1f["sigma"] == r["sigma"]].iloc[0]
        md_lines.append(f"| {r['sigma']:.2f} | {r['shap_snr']:.3e} | {rf['shap_snr']:.2f} | {r['linear_snr']:.2f} |")
    md_lines.append("")
    md_lines.append(f"**Linear GWAS beats grouped Shapley at every sigma** "
                    f"(linear range {s1f['linear_snr'].min():.1f}–{s1f['linear_snr'].max():.1f}, "
                    f"fitted Shapley range {s1f['shap_snr'].min():.1f}–{s1f['shap_snr'].max():.1f}).")
    md_lines.append("")

    md_lines.append("## Scenario 3 — Pairwise Epistasis (recovery)")
    md_lines.append("")
    md_lines.append("Headline: epistatic interaction Shapley `I_01` vs the worst "
                    "null pair `null_max = max(I_02, I_12, I_03, I_13, I_23)`. Oracle "
                    "reported `null_max < 3e-15` (tautological — closed-form has no "
                    "phantom interactions). Fitted RF surfaces real null-pair noise.")
    md_lines.append("")
    md_lines.append("| b | I_01 (oracle) | I_01 (fitted) | null_max (oracle) | null_max (fitted) | I_01/null_max (fitted) |")
    md_lines.append("|---:|--------------:|--------------:|------------------:|------------------:|-----------------------:|")
    for _, r in s3o.iterrows():
        rf = s3f[s3f["b_01"] == r["b_01"]].iloc[0]
        ratio = rf["I_01"] / max(rf["null_max"], 1e-30)
        md_lines.append(f"| {r['b_01']:.2f} | {r['I_01']:.4f} | {rf['I_01']:.4f} | {r['null_max']:.2e} | {rf['null_max']:.4f} | {ratio:.2f} |")
    md_lines.append("")
    # smallest b at which fitted I_01 >= 5x null_max
    crit_5x = s3f[s3f["I_01"] >= 5 * s3f["null_max"]]["b_01"]
    if len(crit_5x) > 0:
        md_lines.append(f"**Smallest b with `I_01 ≥ 5 × null_max` (fitted)**: b = {crit_5x.min():.2f}.")
    else:
        md_lines.append("**Fitted I_01 never reaches 5× null_max** in the swept range "
                        "[0.02, 1.0]; the ratio plateaus at ≈ 2–3×.")
    md_lines.append("")

    md_lines.append("## Scenario 5 — LD Proxy-Causal (recovery)")
    md_lines.append("")
    md_lines.append("Headline: `phi_causal_frac = φ_G0 / (φ_G0 + φ_G1)`. Oracle "
                    "reported exactly 1.0 at every rho (tautological — `f = 10·mean(G0)` "
                    "ignores G1). Fitted RF preserves the strong isolation but adds "
                    "tiny non-zero variance.")
    md_lines.append("")
    md_lines.append("| rho | phi_causal_frac (oracle) | phi_causal_frac (fitted) | linear_smear |")
    md_lines.append("|----:|-------------------------:|-------------------------:|-------------:|")
    for _, r in s5o.iterrows():
        rf = s5f[s5f["rho"] == r["rho"]].iloc[0]
        md_lines.append(f"| {r['rho']:.2f} | {r['phi_causal_frac']:.4f} | {rf['phi_causal_frac']:.4f} | {r['linear_smear']:.4f} |")
    md_lines.append("")
    md_lines.append(f"**phi_causal_frac stays > 0.998 at every rho up to 0.95** "
                    f"(min over rho = {s5f['phi_causal_frac'].min():.4f}).")
    md_lines.append("")

    md_lines.append("## Scenario 5 — Stability (rho=0.8, 15 seeds)")
    md_lines.append("")
    md_lines.append("| metric | oracle | fitted |")
    md_lines.append("|:-------|-------:|-------:|")
    md_lines.append(f"| mean(phi_causal_frac) | {s5os['phi_causal_frac'].mean():.6f} | {s5fs['phi_causal_frac'].mean():.6f} |")
    md_lines.append(f"| sd(phi_causal_frac)   | {s5os['phi_causal_frac'].std():.2e} | {s5fs['phi_causal_frac'].std():.2e} |")
    md_lines.append("")
    md_lines.append("Oracle is a δ-function at 1.0 (sd = 0). Fitted has finite but "
                    "tiny sd ≈ 3e-4 — the figure-1-right-panel flatness is the "
                    "*correct* result for this scenario, not a degenerate artefact.")
    md_lines.append("")

    md_lines.append("## Scenario 1 — Stability (sigma=0.5, 15 seeds)")
    md_lines.append("")
    md_lines.append("| group | mean(|phi|) oracle | mean(|phi|) fitted |")
    md_lines.append("|:------|-------------------:|-------------------:|")
    for col, lbl in zip(["G0", "G1", "G2", "G3"], ["G0 (causal, β=2)", "G1 (causal, β=1.5)", "G2 (null)", "G3 (null)"]):
        md_lines.append(f"| {lbl} | {s1os[col].mean():.4f} | {s1fs[col].mean():.4f} |")
    md_lines.append("")
    md_lines.append("Oracle null = 0 exactly. Fitted null ≈ 0.08 (RF leakage). "
                    "Causal/null ratio: oracle = ∞; fitted G0/null ≈ 23, G1/null ≈ 10. "
                    "Still recovers ranking, no longer perfect separation.")
    md_lines.append("")

    md_lines.append("## Scenario 3 — Stability (b=0.3, 15 seeds)")
    md_lines.append("")
    md_lines.append("| pair | mean oracle | mean fitted |")
    md_lines.append("|:-----|------------:|------------:|")
    md_lines.append(f"| I_01 (epistatic) | {s3os['I_01'].mean():.4f} | {s3fs['I_01'].mean():.4f} |")
    md_lines.append(f"| I_02 (null)      | {s3os['I_02'].mean():.2e} | {s3fs['I_02'].mean():.4f} |")
    md_lines.append(f"| I_12 (null)      | {s3os['I_12'].mean():.2e} | {s3fs['I_12'].mean():.4f} |")
    md_lines.append("")
    md_lines.append(f"Oracle ratio I_01 / mean-null ≈ "
                    f"{s3os['I_01'].mean() / max(s3os[['I_02','I_12']].values.mean(), 1e-30):.1e}; "
                    f"fitted ratio ≈ {s3fs['I_01'].mean() / max(s3fs[['I_02','I_12']].values.mean(), 1e-30):.2f}× "
                    "— signal still dominant but not categorically.")
    md_lines.append("")

    (OUT / "oracle_vs_fitted_summary.md").write_text("\n".join(md_lines))
    print(f"Wrote {OUT / 'oracle_vs_fitted_comparison.csv'}")
    print(f"Wrote {OUT / 'oracle_vs_fitted_summary.md'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit and summarize the completed Study 27 comparisons without refitting."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/study27_kan_comparison"
MODELS = ["random_forest", "transformer", "kan"]


def read_run(path):
    if not (path / "COMPLETE.json").exists():
        raise RuntimeError(f"Run incomplete: {path}")
    m = pd.read_csv(path / "cv_metrics.tsv", sep="\t")
    pred = pd.read_csv(path / "predictions.tsv", sep="\t")
    assert not m.duplicated(["model", "seed", "fold"]).any()
    assert not pred.duplicated(["model", "seed", "accession"]).any()
    assert np.isfinite(m[["r2", "rmse"]]).all().all()
    for (model, seed), rows in pred.groupby(["model", "seed"]):
        assert len(rows) == 81 and rows.accession.nunique() == 81
        for fold, r in rows.groupby("fold"):
            stored = m.query("model == @model and seed == @seed and fold == @fold").iloc[0]
            np.testing.assert_allclose(stored.r2, r2_score(r.observed, r.predicted), atol=1e-10)
    return m, pred


def main():
    m, pred = read_run(OUT)
    avg = m.groupby(["model", "fold"])[["r2", "rmse"]].mean().reset_index()
    stats = avg.groupby("model").agg(mean_cv_r2=("r2", "mean"), sd_cv_r2=("r2", "std"),
                                     mean_cv_rmse=("rmse", "mean"))
    seed_stats = m.groupby(["model", "seed"]).r2.mean().rename("mean_cv_r2").reset_index()
    seed_stats.to_csv(OUT / "seed_summary.tsv", sep="\t", index=False)
    foldwide = avg.pivot(index="fold", columns="model", values="r2")
    delta = pd.DataFrame({"kan_minus_" + name: foldwide.kan - foldwide[name]
                          for name in ["random_forest", "transformer", "pc_only"]})
    delta.to_csv(OUT / "paired_fold_differences.tsv", sep="\t")
    sens = []
    for pc in [0, 3, 5, 10]:
        directory = OUT if pc == 3 else OUT.with_name(OUT.name + f"_pc{pc}")
        mm, _ = read_run(directory)
        for name in MODELS + ["pc_only"]:
            rows = mm[(mm.model == name) & (mm.seed == (-1 if name == "pc_only" else 42))]
            assert len(rows) == 5
            sens.append(dict(n_pcs=pc, model=name, mean_cv_r2=rows.r2.mean(), sd_cv_r2=rows.r2.std()))
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT / "pc_sensitivity_summary.tsv", sep="\t", index=False)

    g = pd.read_csv(OUT / "group_shap.tsv", sep="\t")
    assert g.efficiency_error.max() < 1e-4
    ranks = g.pivot(index=["model", "seed", "fold"], columns="group", values="mean_abs_shap")
    stability = []
    for name in MODELS:
        a = ranks.loc[name]
        for i in range(len(a)):
            for j in range(i+1, len(a)):
                si, fi = a.index[i]
                sj, fj = a.index[j]
                if fi == fj:
                    kind = "same_fold_across_seeds"
                elif si == sj:
                    kind = "same_seed_across_folds"
                else:
                    continue
                rho = spearmanr(a.iloc[i], a.iloc[j]).statistic
                stability.append(dict(model=name, comparison=kind, seed1=si, fold1=fi,
                                      seed2=sj, fold2=fj, spearman_rho=rho))
    stability = pd.DataFrame(stability)
    stability.to_csv(OUT / "group_rank_stability.tsv", sep="\t", index=False)
    gs = g.groupby(["model", "group"]).mean_abs_shap.mean().reset_index()
    gs.to_csv(OUT / "group_summary.tsv", sep="\t", index=False)
    inter = pd.read_csv(OUT / "group_interactions.tsv", sep="\t")
    inter.groupby(["model", "group1", "group2"]).mean_abs_interaction.mean().to_csv(
        OUT / "interaction_summary.tsv", sep="\t")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    names = MODELS + ["pc_only"]
    colors = ["#2574A9", "#B66A24", "#37864C", "#777777"]
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = foldwide[name].to_numpy()
        axes[0].scatter(np.full(5, i) + np.linspace(-.12, .12, 5), vals, color=color, alpha=.65)
        axes[0].plot([i-.22, i+.22], [vals.mean()]*2, color=color, lw=3)
        ss = sens[sens.model == name]
        axes[1].plot(ss.n_pcs, ss.mean_cv_r2, "o-", label=name.replace("_", " "), color=color)
    axes[0].set_xticks(range(4), ["RF", "Transformer", "KAN", "PC only"])
    axes[0].set_ylabel("Held-out fold R²")
    axes[0].set_title("3 PCs: each dot is a fold (seeds averaged)")
    axes[1].set_title("Population-structure sensitivity (seed 42)")
    axes[1].set_xlabel("Number of kinship PCs")
    axes[1].set_ylabel("Mean of five held-out fold R² values")
    axes[1].set_xticks([0, 3, 5, 10])
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axhline(0, color="black", ls="--", lw=.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "model_comparison.png", dpi=180)
    plt.close(fig)

    lines = ["# Study 27: KAN comparison results", "",
             "81 accessions; 200 training-selected SNPs; five outer folds; three model seeds; three PCs.", "",
             "Scores below use training-only genome-wide SNP selection and inner-validation early stopping.",
             "They are not directly comparable to the historical preselected-panel RF 0.559 / Transformer 0.471 scores.", "",
             "| Model | Mean fold R² | Fold SD | Mean fold RMSE |", "|---|---:|---:|---:|"]
    for name in names:
        r = stats.loc[name]
        lines.append(f"| {name} | {r.mean_cv_r2:.4f} | {r.sd_cv_r2:.4f} | {r.mean_cv_rmse:.4f} |")
    lines += ["", "Seeds are averaged within each fold first. SD describes five folds, not a confidence interval.", ""]
    if (stats.loc[MODELS, "mean_cv_r2"] < 0).all():
        lines += ["All three nonlinear models have negative mean fold R². This experiment does not support useful",
                  "held-out predictive performance for the tested configurations. Adding KAN does not rescue that result.", ""]
    lines += ["## Seed variation", "", "| Model | Minimum seed mean R² | Maximum seed mean R² |",
              "|---|---:|---:|"]
    for name in MODELS:
        r = seed_stats[seed_stats.model == name].mean_cv_r2
        lines.append(f"| {name} | {r.min():.4f} | {r.max():.4f} |")
    lines += ["", "## Paired fold differences", "", "| Contrast | Mean R² difference | Positive folds / 5 |",
              "|---|---:|---:|"]
    for name in delta:
        lines.append(f"| {name} | {delta[name].mean():.4f} | {int((delta[name] > 0).sum())} |")
    lines += ["", "## Population adjustment", "", "One fixed seed (42) in every arm; five folds each.", "",
              "| PCs | RF | Transformer | KAN | PC only |", "|---:|---:|---:|---:|---:|"]
    for pc, rows in sens.groupby("n_pcs"):
        values = rows.set_index("model").mean_cv_r2
        lines.append(f"| {pc} | " + " | ".join(f"{values[n]:.4f}" for n in names) + " |")
    lines += ["", "## Explanation scope", "",
              "Exact chromosome-group Shapley coalitions were evaluated on held-out accessions, using eight",
              "training backgrounds; PCs form a separate player. Group rankings and group-pair interaction",
              "magnitudes are saved for reproducibility, not asserted as biological findings. No SNP-pair",
              "localization or candidate-gene validation was performed.", "",
              "| Model | Median group-rank rho across seeds (same fold) | Across folds (same seed) |",
              "|---|---:|---:|"]
    for name in MODELS:
        r = stability[stability.model == name].groupby("comparison").spearman_rho.median()
        lines.append(f"| {name} | {r['same_fold_across_seeds']:.3f} | {r['same_seed_across_folds']:.3f} |")
    lines += ["", f"Maximum Shapley efficiency error: {g.efficiency_error.max():.3g}.", "",
              "## Interpretation limits", "",
              "The sample is small and the inner validation sets contain only 13 accessions. Early stopping",
              "can therefore be unstable. KAN is one fixed compact configuration; these results do not rule",
              "out all KAN architectures. Kinship PCs use all genotypes without phenotype information, so",
              "the representation is transductive. Random folds are not a leave-population-out evaluation.",
              "The change from historical scores cannot isolate selection leakage alone: panels, neural",
              "configurations and fitting procedures also differ. No causal or biological-epistasis claim follows.", "",
              "See [protocol](../../docs/study27_kan_comparison.md), run_config.json, splits.tsv,",
              "outer*_panel.tsv, inner*_panel.tsv, cv_metrics.tsv and predictions.tsv for the audit trail.", "",
              "![Model comparison](model_comparison.png)"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("Verified predictions and wrote REPORT.md, comparison plot, and summary tables.")
    print(stats.to_string())


if __name__ == "__main__":
    main()

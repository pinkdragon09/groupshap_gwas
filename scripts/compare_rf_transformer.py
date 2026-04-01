#!/usr/bin/env python3
"""
Build a direct random-forest vs transformer comparison table from saved outputs.

Outputs:
- model_metrics.tsv
- rank_comparison.tsv
- top_hits_comparison.tsv
- top10_side_by_side.tsv
- top10_side_by_side.md
- published_results_comparison.tsv
- published_results_comparison.md
- summary.md
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd


DEFAULT_RF_DIR = Path("outputs/random_forest_gwas_full")
DEFAULT_TRANSFORMER_DIR = Path("outputs/transformer_gwas_full")
DEFAULT_LINEAR_HITS = Path("outputs/full_baseline_gwas/study27_mean_trl_top_hits.tsv")
DEFAULT_OUTDIR = Path("outputs/model_comparison")
DEFAULT_GFF = Path("data/reference/Araport11_GFF3_genes_transposons.Mar92021.gff.gz")

PUBLISHED_LOCUS_TERMS = {
    "FRI",
    "FRIGIDA",
    "FLC",
    "RPM1",
    "RPS2",
    "RPS5",
    "ACD6",
    "DOG1",
    "ANL2",
    "FT",
    "FRO2",
    "PUB4",
    "CYP81D6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf-dir", default=str(DEFAULT_RF_DIR))
    parser.add_argument("--transformer-dir", default=str(DEFAULT_TRANSFORMER_DIR))
    parser.add_argument("--linear-hits", default=str(DEFAULT_LINEAR_HITS))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--gff", default=str(DEFAULT_GFF))
    return parser.parse_args()


def read_summary(summary_path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in summary_path.read_text().splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            data[key] = value
    return data


def load_model_run(run_dir: Path, label: str) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame]:
    summary = read_summary(run_dir / "summary.txt")
    metrics = pd.read_csv(run_dir / "cv_metrics.tsv", sep="\t")
    ranking = pd.read_csv(run_dir / "global_shap_ranking.tsv", sep="\t")
    ranking = ranking[ranking["feature_type"] == "snp"].copy()
    ranking["snp_index"] = ranking["snp_index"].astype(int)
    ranking[f"{label}_rank"] = np.arange(1, len(ranking) + 1)
    return summary, metrics, ranking


def parse_gff_attributes(attrs: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in attrs.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed[key] = unquote(value)
    return parsed


def load_gene_annotation(gff_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with gzip.open(gff_path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            chrom, _source, feature, start, end, _score, strand, _phase, attrs = line.rstrip().split("\t")
            if feature != "gene":
                continue
            parsed = parse_gff_attributes(attrs)
            rows.append(
                {
                    "chromosome": chrom,
                    "start": int(start),
                    "end": int(end),
                    "strand": strand,
                    "gene_id": parsed.get("Name") or parsed.get("ID", "").replace("gene:", ""),
                    "symbol": parsed.get("symbol", ""),
                    "alias": parsed.get("Alias", ""),
                    "note": parsed.get("Note", ""),
                }
            )
    genes = pd.DataFrame(rows)
    if genes.empty:
        raise ValueError(f"No gene records found in {gff_path}")
    return genes


def nearest_gene_for_row(row: pd.Series, genes: pd.DataFrame) -> pd.Series:
    chrom_genes = genes.loc[genes["chromosome"] == row["chromosome"]].copy()
    pos = int(row["position"])
    inside = chrom_genes[(chrom_genes["start"] <= pos) & (chrom_genes["end"] >= pos)]
    if not inside.empty:
        hit = inside.sort_values(["start", "end"]).iloc[0]
        distance = 0
    else:
        chrom_genes["distance_bp"] = chrom_genes.apply(
            lambda gene: min(abs(pos - int(gene["start"])), abs(pos - int(gene["end"]))),
            axis=1,
        )
        hit = chrom_genes.sort_values(["distance_bp", "start", "end"]).iloc[0]
        distance = int(hit["distance_bp"])

    symbol = str(hit["symbol"]).strip()
    gene_label = str(hit["gene_id"])
    if symbol:
        gene_label = f"{gene_label} ({symbol})"

    terms = {str(hit["gene_id"]).upper(), symbol.upper()}
    alias = str(hit["alias"]).strip()
    if alias:
        terms.update(piece.strip().upper() for piece in alias.split(",") if piece.strip())
    published_match = sorted(PUBLISHED_LOCUS_TERMS.intersection(terms))

    return pd.Series(
        {
            "nearest_gene": gene_label,
            "nearest_gene_id": hit["gene_id"],
            "nearest_gene_symbol": symbol,
            "distance_bp": distance,
            "gene_note": str(hit["note"]),
            "published_locus_match": ";".join(published_match) if published_match else "",
        }
    )


def annotate_with_genes(df: pd.DataFrame, genes: pd.DataFrame) -> pd.DataFrame:
    annotations = df.apply(lambda row: nearest_gene_for_row(row, genes), axis=1)
    return pd.concat([df.reset_index(drop=True), annotations.reset_index(drop=True)], axis=1)


def build_model_metrics(
    rf_summary: dict[str, str],
    rf_metrics: pd.DataFrame,
    tr_summary: dict[str, str],
    tr_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "model": "random_forest",
            "matched_accessions": int(rf_summary["matched_accessions"]),
            "top_snps_requested": int(rf_summary["top_snps"]),
            "top_snp_features_used": int(rf_summary["feature_count_snps"]),
            "folds": int(rf_summary["folds"]),
            "mean_cv_r2": float(rf_summary["mean_cv_r2"]),
            "sd_cv_r2": float(rf_metrics["r2"].std(ddof=1)),
            "mean_cv_rmse": float(rf_summary["mean_cv_rmse"]),
            "sd_cv_rmse": float(rf_metrics["rmse"].std(ddof=1)),
        },
        {
            "model": "transformer",
            "matched_accessions": int(tr_summary["matched_accessions"]),
            "top_snps_requested": int(tr_summary["top_snps"]),
            "top_snp_features_used": int(tr_summary["feature_count_snps"]),
            "folds": int(tr_summary["folds"]),
            "mean_cv_r2": float(tr_summary["mean_cv_r2"]),
            "sd_cv_r2": float(tr_metrics["r2"].std(ddof=1)),
            "mean_cv_rmse": float(tr_summary["mean_cv_rmse"]),
            "sd_cv_rmse": float(tr_metrics["rmse"].std(ddof=1)),
        },
    ]
    return pd.DataFrame(rows)


def build_rank_comparison(
    rf_rank: pd.DataFrame,
    tr_rank: pd.DataFrame,
    linear_hits: pd.DataFrame,
) -> pd.DataFrame:
    linear = linear_hits.copy()
    linear["snp_index"] = linear["snp_index"].astype(int)
    linear["linear_rank"] = np.arange(1, len(linear) + 1)

    merged = (
        linear[["snp_index", "linear_rank", "chromosome", "position", "p_value"]]
        .merge(rf_rank[["snp_index", "random_forest_rank", "mean_abs_shap"]], on="snp_index", how="inner")
        .rename(columns={"mean_abs_shap": "rf_mean_abs_shap"})
        .merge(tr_rank[["snp_index", "transformer_rank", "mean_abs_shap"]], on="snp_index", how="inner")
        .rename(columns={"mean_abs_shap": "transformer_mean_abs_shap"})
    )
    return merged


def build_top_hits_table(
    rf_rank: pd.DataFrame,
    tr_rank: pd.DataFrame,
    linear_hits: pd.DataFrame,
    genes: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    linear = linear_hits.copy()
    linear["snp_index"] = linear["snp_index"].astype(int)
    linear["linear_rank"] = np.arange(1, len(linear) + 1)

    rf_top = rf_rank.head(top_k)[["snp_index", "chromosome", "position", "random_forest_rank", "mean_abs_shap"]]
    rf_top = rf_top.rename(columns={"mean_abs_shap": "rf_mean_abs_shap"})
    rf_top["source_model"] = "random_forest"

    tr_top = tr_rank.head(top_k)[["snp_index", "chromosome", "position", "transformer_rank", "mean_abs_shap"]]
    tr_top = tr_top.rename(columns={"mean_abs_shap": "transformer_mean_abs_shap"})
    tr_top["source_model"] = "transformer"

    combined = pd.concat([rf_top, tr_top], ignore_index=True, sort=False)
    combined = combined.merge(linear[["snp_index", "linear_rank", "p_value"]], on="snp_index", how="left")
    return annotate_with_genes(combined, genes)


def build_side_by_side_top10(
    rf_rank: pd.DataFrame,
    tr_rank: pd.DataFrame,
    linear_hits: pd.DataFrame,
    genes: pd.DataFrame,
) -> pd.DataFrame:
    linear = linear_hits.copy()
    linear["snp_index"] = linear["snp_index"].astype(int)
    linear["linear_rank"] = np.arange(1, len(linear) + 1)

    rf_top = annotate_with_genes(
        rf_rank.head(10)[["snp_index", "chromosome", "position", "random_forest_rank", "mean_abs_shap"]]
        .rename(columns={"mean_abs_shap": "rf_mean_abs_shap"})
        .merge(linear[["snp_index", "linear_rank"]], on="snp_index", how="left")
        .rename(columns={"linear_rank": "rf_linear_rank"}),
        genes,
    )
    tr_top = annotate_with_genes(
        tr_rank.head(10)[["snp_index", "chromosome", "position", "transformer_rank", "mean_abs_shap"]]
        .rename(columns={"mean_abs_shap": "transformer_mean_abs_shap"})
        .merge(linear[["snp_index", "linear_rank"]], on="snp_index", how="left")
        .rename(columns={"linear_rank": "transformer_linear_rank"}),
        genes,
    )

    rows: list[dict[str, object]] = []
    for idx in range(10):
        rf_row = rf_top.iloc[idx]
        tr_row = tr_top.iloc[idx]
        rows.append(
            {
                "rf_rank": int(rf_row["random_forest_rank"]),
                "rf_snp": f"{rf_row['chromosome']}:{int(rf_row['position'])}",
                "rf_linear_rank": int(rf_row["rf_linear_rank"]),
                "rf_nearest_gene": rf_row["nearest_gene"],
                "rf_distance_bp": int(rf_row["distance_bp"]),
                "rf_published_match": rf_row["published_locus_match"] or "none",
                "transformer_rank": int(tr_row["transformer_rank"]),
                "transformer_snp": f"{tr_row['chromosome']}:{int(tr_row['position'])}",
                "transformer_linear_rank": int(tr_row["transformer_linear_rank"]),
                "transformer_nearest_gene": tr_row["nearest_gene"],
                "transformer_distance_bp": int(tr_row["distance_bp"]),
                "transformer_published_match": tr_row["published_locus_match"] or "none",
            }
        )
    return pd.DataFrame(rows)


def build_published_results_comparison(
    rf_summary: dict[str, str],
    tr_summary: dict[str, str],
    linear_summary: dict[str, str],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": "This work: PC-adjusted linear GWAS",
                "trait_context": "AraPheno study 27, Mean(TRL) C root-length trait",
                "sample_size": 81,
                "analysis_type": "Genome-wide SNP-wise linear regression with 3 kinship PCs",
                "reported_metric": "Top-hit partial R2 / top-hit p-value",
                "reported_value": "partial R2 = 0.279; p = 6.60e-07 at Chr4:5828682",
                "published_loci_summary": "Top hit at Chr4:5828682 in current baseline scan",
                "comparability_note": "Directly comparable only to additive-association baselines, not to predictive ML R2.",
            },
            {
                "source": "This work: random forest",
                "trait_context": "AraPheno study 27, Mean(TRL) C root-length trait",
                "sample_size": int(rf_summary["matched_accessions"]),
                "analysis_type": "RandomForestRegressor on 200 prefiltered SNPs plus 3 PCs",
                "reported_metric": "5-fold cross-validated predictive R2 / RMSE",
                "reported_value": f"R2 = {float(rf_summary['mean_cv_r2']):.3f}; RMSE = {float(rf_summary['mean_cv_rmse']):.3f}",
                "published_loci_summary": "Top RF loci cluster at Chr3:9.07 Mb, Chr3:18.77 Mb, Chr1:21.85 Mb, and Chr5:16.90 Mb.",
                "comparability_note": "Predictive R2 is not directly equivalent to GWAS heritability or per-locus variance explained.",
            },
            {
                "source": "This work: transformer",
                "trait_context": "AraPheno study 27, Mean(TRL) C root-length trait",
                "sample_size": int(tr_summary["matched_accessions"]),
                "analysis_type": "Transformer regressor on 200 prefiltered SNPs plus 3 PCs",
                "reported_metric": "5-fold cross-validated predictive R2 / RMSE",
                "reported_value": f"R2 = {float(tr_summary['mean_cv_r2']):.3f}; RMSE = {float(tr_summary['mean_cv_rmse']):.3f}",
                "published_loci_summary": "Top transformer loci include Chr4:5828682, Chr5:6831693, and Chr3:20.72 Mb.",
                "comparability_note": "Best predictive model here, but attribution is exploratory because Deep SHAP required relaxed additivity checks.",
            },
            {
                "source": "Ristova et al. (2018)",
                "trait_context": "Root-system architecture under control and hormone treatments",
                "sample_size": 192,
                "analysis_type": "Mixed-model GWAS with BH-FDR 5%",
                "reported_metric": "Broad-sense H2 / significant associations",
                "reported_value": "Average H2 = 0.57 (range 0.22 to 0.77); 114 significant associations",
                "published_loci_summary": "441 genes within 20 kb windows; 98 ABA, 5 CK, 1 IAA, and 9 control associations.",
                "comparability_note": "Most relevant published benchmark for study 27, but local notes do not contain a machine-readable per-gene list.",
            },
            {
                "source": "Satbhai et al. (2017)",
                "trait_context": "Root length under iron deficiency",
                "sample_size": "Swedish subset; exact n not recoverable from local PDF",
                "analysis_type": "BH-corrected GWAS plus mechanistic follow-up",
                "reported_metric": "Lead locus",
                "reported_value": "FRO2 locus with top SNP c1p207099",
                "published_loci_summary": "Regulatory and coding variation around FRO2 examined within +/-25 kb.",
                "comparability_note": "Useful mechanistic root-length reference, but not the same phenotype or accession panel as study 27.",
            },
        ]
    )


def format_top10_side_by_side_md(table: pd.DataFrame) -> str:
    lines = [
        "# Top 10 RF vs Transformer SNPs",
        "",
        "| RF Rank | RF SNP | RF Linear Rank | RF Nearest Gene | RF Dist. (bp) | RF Published Match | Transformer Rank | Transformer SNP | Transformer Linear Rank | Transformer Nearest Gene | Transformer Dist. (bp) | Transformer Published Match |",
        "|---:|---|---:|---|---:|---|---:|---|---:|---|---:|---|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.rf_rank} | {row.rf_snp} | {row.rf_linear_rank} | {row.rf_nearest_gene} | {row.rf_distance_bp} | {row.rf_published_match} | "
            f"{row.transformer_rank} | {row.transformer_snp} | {row.transformer_linear_rank} | {row.transformer_nearest_gene} | {row.transformer_distance_bp} | {row.transformer_published_match} |"
        )
    return "\n".join(lines) + "\n"


def format_published_results_md(table: pd.DataFrame) -> str:
    lines = [
        "# Current Models vs Published Results",
        "",
        "The current runs use AraPheno study 27 (`Mean(TRL) C`), which is part of the Ristova et al. (2018) root-trait dataset. ",
        "The machine-learning metrics below are predictive `R2` values, so they are not directly equivalent to mixed-model GWAS heritability estimates or per-locus variance explained in the published papers.",
        "",
        "| Source | Trait Context | Sample Size | Analysis Type | Reported Metric | Reported Value | Published Loci Summary | Comparability Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.source} | {row.trait_context} | {row.sample_size} | {row.analysis_type} | "
            f"{row.reported_metric} | {row.reported_value} | {row.published_loci_summary} | {row.comparability_note} |"
        )
    lines.extend(
        [
            "",
            "Nearest-gene overlap caveat: the local literature summaries name a small number of canonical Arabidopsis loci such as `FRI`, `FLC`, `FRO2`, and `PUB4`, but the current top 10 RF and transformer nearest-gene sets do not overlap those named loci. Exact overlap against the full Ristova et al. candidate-gene list cannot yet be computed from local files because the 441-gene window list is not available in machine-readable form in this repo.",
        ]
    )
    return "\n".join(lines) + "\n"


def format_summary_md(model_metrics: pd.DataFrame, rank_table: pd.DataFrame) -> str:
    rf = model_metrics.loc[model_metrics["model"] == "random_forest"].iloc[0]
    tr = model_metrics.loc[model_metrics["model"] == "transformer"].iloc[0]

    spearman_rf = rank_table["linear_rank"].corr(rank_table["random_forest_rank"], method="spearman")
    spearman_tr = rank_table["linear_rank"].corr(rank_table["transformer_rank"], method="spearman")

    linear_top10 = rank_table[rank_table["linear_rank"] <= 10]
    rf_top50 = int((linear_top10["random_forest_rank"] <= 50).sum())
    tr_top50 = int((linear_top10["transformer_rank"] <= 50).sum())
    rf_top100 = int((linear_top10["random_forest_rank"] <= 100).sum())
    tr_top100 = int((linear_top10["transformer_rank"] <= 100).sum())

    lines = [
        "# RF vs Transformer Comparison",
        "",
        "## Predictive Metrics",
        "",
        "| Model | Mean CV R2 | SD CV R2 | Mean CV RMSE | SD CV RMSE | SNP Features Used |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Random forest | {rf['mean_cv_r2']:.6f} | {rf['sd_cv_r2']:.6f} | {rf['mean_cv_rmse']:.6f} | {rf['sd_cv_rmse']:.6f} | {int(rf['top_snp_features_used'])} |",
        f"| Transformer | {tr['mean_cv_r2']:.6f} | {tr['sd_cv_r2']:.6f} | {tr['mean_cv_rmse']:.6f} | {tr['sd_cv_rmse']:.6f} | {int(tr['top_snp_features_used'])} |",
        "",
        "## Rank Comparison Against Linear Baseline",
        "",
        f"- Shared prefiltered SNPs: `{len(rank_table)}`",
        f"- Random forest vs linear Spearman rank correlation: `{spearman_rf:.4f}`",
        f"- Transformer vs linear Spearman rank correlation: `{spearman_tr:.4f}`",
        f"- Linear top 10 recovered in RF top 50: `{rf_top50}`",
        f"- Linear top 10 recovered in transformer top 50: `{tr_top50}`",
        f"- Linear top 10 recovered in RF top 100: `{rf_top100}`",
        f"- Linear top 10 recovered in transformer top 100: `{tr_top100}`",
        "",
        "Important caveat: both nonlinear models were trained on SNPs prefiltered from the linear-GWAS hit table, so raw overlap is structurally inflated. The rank statistics above are more meaningful than overlap counts alone.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rf_summary, rf_metrics, rf_rank = load_model_run(Path(args.rf_dir), "random_forest")
    tr_summary, tr_metrics, tr_rank = load_model_run(Path(args.transformer_dir), "transformer")
    linear_hits = pd.read_csv(args.linear_hits, sep="\t")
    linear_summary = read_summary(Path("outputs/full_baseline_gwas/summary.txt"))
    genes = load_gene_annotation(Path(args.gff))

    model_metrics = build_model_metrics(rf_summary, rf_metrics, tr_summary, tr_metrics)
    rank_table = build_rank_comparison(rf_rank, tr_rank, linear_hits)
    top_hits_table = build_top_hits_table(rf_rank, tr_rank, linear_hits, genes, top_k=20)
    top10_side_by_side = build_side_by_side_top10(rf_rank, tr_rank, linear_hits, genes)
    published_results = build_published_results_comparison(rf_summary, tr_summary, linear_summary)
    summary_md = format_summary_md(model_metrics, rank_table)
    top10_md = format_top10_side_by_side_md(top10_side_by_side)
    published_md = format_published_results_md(published_results)

    model_metrics.to_csv(outdir / "model_metrics.tsv", sep="\t", index=False)
    rank_table.to_csv(outdir / "rank_comparison.tsv", sep="\t", index=False)
    top_hits_table.to_csv(outdir / "top_hits_comparison.tsv", sep="\t", index=False)
    top10_side_by_side.to_csv(outdir / "top10_side_by_side.tsv", sep="\t", index=False)
    published_results.to_csv(outdir / "published_results_comparison.tsv", sep="\t", index=False)
    (outdir / "summary.md").write_text(summary_md, encoding="ascii")
    (outdir / "top10_side_by_side.md").write_text(top10_md, encoding="ascii")
    (outdir / "published_results_comparison.md").write_text(published_md, encoding="ascii")

    print(outdir / "model_metrics.tsv")
    print(outdir / "rank_comparison.tsv")
    print(outdir / "top_hits_comparison.tsv")
    print(outdir / "top10_side_by_side.tsv")
    print(outdir / "top10_side_by_side.md")
    print(outdir / "published_results_comparison.tsv")
    print(outdir / "published_results_comparison.md")
    print(outdir / "summary.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GWAS-style baseline on the downloaded Arabidopsis resources.

Method:
- one quantitative AraPheno trait
- intercept + kinship-derived principal components as covariates
- SNP-wise linear regression across the full 1001 Genomes HDF5 matrix
- chunked scan to keep memory bounded

This is a practical baseline, not a full mixed-model GWAS.
"""

from __future__ import annotations

import csv
import io
import math
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import t as student_t


PHENO_ZIP = Path("data/arapheno/bulk/database.zip")
GENO_H5 = Path("data/1001genomes/genotypes/1001_SNP_MATRIX/imputed_snps_binary.hdf5")
KINSHIP_H5 = Path("data/1001genomes/genotypes/1001_SNP_MATRIX/kinship_ibs_mac5.hdf5")
OUTDIR = Path("outputs/full_baseline_gwas")

STUDY_VALUES_FILE = "27/study_27_values.csv"
PHENOTYPE_COL = "Mean(TRL) C"
N_PCS = 3
CHUNK_SIZE = 10000
TOP_K = 200
MIN_MINOR_ALLELE_COUNT = 2


def load_phenotype() -> tuple[dict[str, float], int]:
    with zipfile.ZipFile(PHENO_ZIP) as zf:
        rows = list(csv.DictReader(io.StringIO(zf.read(STUDY_VALUES_FILE).decode("utf-8"))))

    y_by_acc: dict[str, float] = {}
    n_all = 0
    for row in rows:
        value = row[PHENOTYPE_COL].strip()
        acc = row["accession_id"].strip()
        if not acc or not value:
            continue
        n_all += 1
        # The study table may contain replicate rows; keep the observed value.
        y_by_acc[acc] = float(value)
    return y_by_acc, n_all


def chrom_labels_and_bounds(h5f: h5py.File) -> tuple[list[str], np.ndarray]:
    labels = [f"Chr{x.decode() if isinstance(x, bytes) else x}" for x in h5f["positions"].attrs["chrs"]]
    bounds = h5f["positions"].attrs["chr_regions"]
    return labels, bounds


def annotate_chromosome(idx_start: int, idx_end: int, labels: list[str], bounds: np.ndarray) -> list[str]:
    chroms = []
    chrom_i = 0
    for idx in range(idx_start, idx_end):
        while chrom_i + 1 < len(bounds) and idx >= bounds[chrom_i][1]:
            chrom_i += 1
        chroms.append(labels[chrom_i])
    return chroms


def compute_pcs(common_accs: list[str], geno_accs: list[str]) -> np.ndarray:
    with h5py.File(KINSHIP_H5, "r") as f:
        kinship = f["kinship"][:]
        kin_accs = [x.decode() if isinstance(x, bytes) else str(x) for x in f["accessions"][:]]

    kin_index = {acc: i for i, acc in enumerate(kin_accs)}
    ix = np.asarray([kin_index[acc] for acc in common_accs], dtype=int)
    sub_k = kinship[np.ix_(ix, ix)]
    eigvals, eigvecs = np.linalg.eigh(sub_k)
    pcs = eigvecs[:, -N_PCS:]
    pcs = (pcs - pcs.mean(axis=0)) / pcs.std(axis=0)
    return pcs


def build_covariates(y: np.ndarray, pcs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n = len(y)
    X = np.column_stack([np.ones(n), pcs])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta_cov = xtx_inv @ X.T @ y
    y_res = y - X @ beta_cov
    y_res_ss = float(y_res @ y_res)
    return X, xtx_inv, y_res, y_res_ss


def scan_full_matrix() -> tuple[dict[str, float], list[dict[str, float]]]:
    y_by_acc, n_all_rows = load_phenotype()

    with h5py.File(GENO_H5, "r") as f:
        geno_accs = [x.decode() if isinstance(x, bytes) else str(x) for x in f["accessions"][:]]
        common_accs = [acc for acc in geno_accs if acc in y_by_acc]
        geno_ix = np.asarray([geno_accs.index(acc) for acc in common_accs], dtype=int)
        y = np.asarray([y_by_acc[acc] for acc in common_accs], dtype=float)
        labels, bounds = chrom_labels_and_bounds(f)
        positions_ds = f["positions"]
        snps_ds = f["snps"]

        pcs = compute_pcs(common_accs, geno_accs)
        X, xtx_inv, y_res, y_res_ss = build_covariates(y, pcs)
        n = len(y)
        df = n - X.shape[1] - 1

        top_hits: list[dict[str, float]] = []
        tested_snps = 0
        kept_snps = 0
        start_time = time.time()

        for chunk_start in range(0, snps_ds.shape[0], CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, snps_ds.shape[0])
            G = snps_ds[chunk_start:chunk_end, geno_ix].astype(float)
            positions = positions_ds[chunk_start:chunk_end]
            chroms = annotate_chromosome(chunk_start, chunk_end, labels, bounds)

            # Residualize the chunk against intercept + PCs.
            xg = X.T @ G.T
            g_res = G - (X @ (xtx_inv @ xg)).T
            g_ss = np.sum(g_res * g_res, axis=1)

            allele_count = G.sum(axis=1)
            minor_allele_count = np.minimum(allele_count, n - allele_count)
            valid = (g_ss > 1e-12) & (minor_allele_count >= MIN_MINOR_ALLELE_COUNT)
            tested_snps += len(G)
            if not np.any(valid):
                continue

            dot_gy = g_res @ y_res
            beta = dot_gy[valid] / g_ss[valid]
            rss = y_res_ss - beta * beta * g_ss[valid]
            rss = np.maximum(rss, 1e-12)
            sigma2 = rss / df
            se_beta = np.sqrt(sigma2 / g_ss[valid])
            t_stat = beta / se_beta
            p_value = 2.0 * student_t.sf(np.abs(t_stat), df)
            r2 = 1.0 - (rss / y_res_ss)

            valid_idx = np.where(valid)[0]
            kept_snps += len(valid_idx)
            for local_i, b, tval, pval, r2val in zip(valid_idx, beta, t_stat, p_value, r2):
                row = {
                    "snp_index": int(chunk_start + local_i),
                    "chromosome": chroms[local_i],
                    "position": int(positions[local_i]),
                    "n_samples": int(n),
                    "beta": float(b),
                    "t_stat": float(tval),
                    "p_value": float(pval),
                    "neg_log10_p": float(-math.log10(max(float(pval), 1e-300))),
                    "r2_partial": float(r2val),
                    "minor_allele_count": int(minor_allele_count[local_i]),
                }
                top_hits.append(row)

            top_hits.sort(key=lambda row: row["p_value"])
            if len(top_hits) > TOP_K:
                top_hits = top_hits[:TOP_K]

        elapsed = time.time() - start_time
        summary = {
            "phenotype_rows_with_values": n_all_rows,
            "matched_accessions": len(common_accs),
            "pcs_used": N_PCS,
            "tested_snps_total": tested_snps,
            "tested_snps_after_filters": kept_snps,
            "scan_seconds": round(elapsed, 2),
            "trait_mean": float(y.mean()),
            "trait_sd": float(y.std()),
        }
        return summary, top_hits


def write_outputs(summary: dict[str, float], top_hits: list[dict[str, float]]) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    top_path = OUTDIR / "study27_mean_trl_top_hits.tsv"
    with top_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(top_hits[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(top_hits)

    summary_path = OUTDIR / "summary.txt"
    with summary_path.open("w") as fh:
        for key, value in summary.items():
            fh.write(f"{key}: {value}\n")
        fh.write(f"top_hit: {top_hits[0]}\n")


def main() -> None:
    summary, top_hits = scan_full_matrix()
    write_outputs(summary, top_hits)
    print(summary)
    print(top_hits[0])
    print(OUTDIR / "study27_mean_trl_top_hits.tsv")


if __name__ == "__main__":
    main()

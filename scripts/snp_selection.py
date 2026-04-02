from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def load_top_snp_indices(baseline_hits: Path, top_snps: int) -> list[int]:
    hits = pd.read_csv(baseline_hits, sep="\t")
    if "snp_index" not in hits.columns:
        raise SystemExit(f"Baseline hit table missing snp_index column: {baseline_hits}")
    hits = hits.sort_values("p_value", ascending=True)
    return hits["snp_index"].astype(int).head(top_snps).tolist()


def load_neutral_snp_indices(
    geno_h5: Path,
    geno_ix: np.ndarray,
    top_snps: int,
    min_mac: int = 2,
    chunk_size: int = 20000,
    step: int = 10000,
) -> list[int]:
    """
    Select a roughly neutral SNP panel without using linear GWAS p-values.

    Strategy:
    - thin the genome by an absolute-index step to reduce local LD redundancy
    - filter by minor allele count on the matched accessions
    - rank remaining candidates by genotype variance on the matched accessions
    """
    candidates: list[tuple[float, int]] = []
    n = len(geno_ix)
    if n == 0:
        return []

    with h5py.File(geno_h5, "r") as f:
        snps_ds = f["snps"]
        total_snps = int(snps_ds.shape[0])

        for chunk_start in range(0, total_snps, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_snps)
            chunk_indices = np.arange(chunk_start, chunk_end, dtype=int)
            keep_mask = (chunk_indices % step) == 0
            if not np.any(keep_mask):
                continue

            selected_idx = chunk_indices[keep_mask]
            G = snps_ds[selected_idx][:, geno_ix].astype(np.float32)
            allele_count = G.sum(axis=1)
            minor_allele_count = np.minimum(allele_count, n - allele_count)
            variance = G.var(axis=1)
            valid = (minor_allele_count >= min_mac) & (variance > 1e-8)
            if not np.any(valid):
                continue

            for idx, var in zip(selected_idx[valid], variance[valid]):
                candidates.append((float(var), int(idx)))

    if not candidates:
        raise SystemExit("Neutral SNP selection found no usable SNPs after MAC/variance filtering.")

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [idx for _var, idx in candidates[:top_snps]]

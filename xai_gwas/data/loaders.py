"""Data loading utilities for common genetic data formats.

Supported formats
-----------------
- Merged genotype-phenotype TSV (project-native format)
- 1001 Genomes / GEUVADIS HDF5 SNP matrices
- PLINK binary (.bed/.bim/.fam) via subprocess plink2 or pandas-plink fallback
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# TSV (project-native)
# ---------------------------------------------------------------------------

def load_tsv(
    path: str | Path,
    phenotype_col: str,
    sample_col: str | None = None,
    sep: str = "\t",
    drop_na_pheno: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a merged genotype-phenotype TSV.

    Returns
    -------
    X : DataFrame of SNP columns (all columns except phenotype and sample id)
    y : Series of phenotype values
    """
    df = pd.read_csv(path, sep=sep)
    if drop_na_pheno:
        df = df.dropna(subset=[phenotype_col])
    y = df[phenotype_col].copy()
    drop_cols = [phenotype_col]
    if sample_col and sample_col in df.columns:
        drop_cols.append(sample_col)
    X = df.drop(columns=drop_cols)
    return X, y


# ---------------------------------------------------------------------------
# HDF5 (1001 Genomes / AraPheno format)
# ---------------------------------------------------------------------------

def load_hdf5_snps(
    hdf5_path: str | Path,
    accession_ids: list[int] | None = None,
    snp_key: str = "snps",
    pos_key: str = "positions",
    chrom_key: str = "chr_index",
    acc_key: str = "accessions",
) -> tuple[pd.DataFrame, dict]:
    """Load a 1001-Genomes–style HDF5 SNP matrix.

    Returns
    -------
    X    : DataFrame (samples × SNPs), columns = '<chrom>_<pos>'
    meta : dict with keys 'positions', 'chromosomes', 'accessions'
    """
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        snps = f[snp_key][:]          # (n_snps, n_accessions) or (n_accessions, n_snps)
        positions = f[pos_key][:]
        chromosomes = f[chrom_key][:]
        accessions = f[acc_key][:]

    # Normalise orientation: we want (n_accessions, n_snps)
    if snps.shape[0] == len(positions):
        snps = snps.T

    col_names = [f"{c}_{p}" for c, p in zip(chromosomes, positions)]
    acc_ids = [int(a) for a in accessions]

    df = pd.DataFrame(snps.astype(np.float32), index=acc_ids, columns=col_names)

    if accession_ids is not None:
        keep = [a for a in accession_ids if a in df.index]
        df = df.loc[keep]

    meta = {
        "positions": positions,
        "chromosomes": chromosomes,
        "accessions": np.array(acc_ids),
    }
    return df, meta


# ---------------------------------------------------------------------------
# PLINK binary (.bed/.bim/.fam)
# ---------------------------------------------------------------------------

def load_plink_bfile(
    bfile: str | Path,
    phenotype_path: str | Path | None = None,
    phenotype_col: str = "PHENO",
    sample_col: str = "IID",
    plink_binary: str = "plink2",
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Load a PLINK binary dataset (.bed/.bim/.fam) into X, y DataFrames.

    Tries pandas-plink first; falls back to calling plink2 to export a raw
    dosage file, then reads that.

    Parameters
    ----------
    bfile         : Prefix path (without .bed/.bim/.fam extensions).
    phenotype_path: Optional separate phenotype file (PLINK format: FID IID PHENO...).
    phenotype_col : Column to use as phenotype (in phenotype file or .fam column 6).
    sample_col    : Sample ID column name for phenotype file merging.
    plink_binary  : Name/path of PLINK2 executable (falls back to plink).

    Returns
    -------
    X : DataFrame of SNP dosage values (samples × SNPs).
    y : Series of phenotype values, or None if no phenotype available.
    """
    bfile = Path(bfile)

    # Strategy 1: pandas-plink (pure Python, no binary required)
    try:
        return _load_via_pandas_plink(bfile, phenotype_path, phenotype_col, sample_col)
    except ImportError:
        pass

    # Strategy 2: shell out to plink2 / plink
    return _load_via_plink_binary(bfile, phenotype_path, phenotype_col, sample_col, plink_binary)


def _load_via_pandas_plink(bfile, phenotype_path, phenotype_col, sample_col):
    from pandas_plink import read_plink1_bin  # type: ignore

    G = read_plink1_bin(str(bfile) + ".bed", verbose=False)
    X = pd.DataFrame(
        G.values,
        index=G.sample.values,
        columns=G.variant.values,
    ).astype(np.float32)
    X.index.name = "IID"

    y = _load_phenotype(bfile, phenotype_path, phenotype_col, sample_col, X.index)
    return X, y


def _load_via_plink_binary(bfile, phenotype_path, phenotype_col, sample_col, plink_binary):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "out"
        cmd = [
            plink_binary,
            "--bfile", str(bfile),
            "--recode", "A",
            "--out", str(out_prefix),
            "--allow-no-sex",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except FileNotFoundError:
            # Try plain plink
            cmd[0] = "plink"
            subprocess.run(cmd, check=True, capture_output=True)

        raw_path = out_prefix.with_suffix(".raw")
        raw = pd.read_csv(raw_path, sep=r"\s+")

    sample_ids = raw["IID"].tolist()
    snp_cols = [c for c in raw.columns if c not in {"FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"}]
    X = raw[snp_cols].copy().astype(np.float32)
    X.index = sample_ids

    y = _load_phenotype(bfile, phenotype_path, phenotype_col, sample_col, pd.Index(sample_ids))
    return X, y


def _load_phenotype(
    bfile: Path,
    phenotype_path: Path | None,
    phenotype_col: str,
    sample_col: str,
    sample_index: pd.Index,
) -> pd.Series | None:
    if phenotype_path is not None:
        pheno_df = pd.read_csv(phenotype_path, sep=r"\s+")
        pheno_df = pheno_df.set_index(sample_col)
        if phenotype_col in pheno_df.columns:
            y = pheno_df[phenotype_col].reindex(sample_index)
            return y

    # Fall back to .fam column 6 (1=control, 2=case, -9/0=missing)
    fam_path = bfile.with_suffix(".fam")
    if fam_path.exists():
        fam = pd.read_csv(fam_path, sep=r"\s+", header=None,
                          names=["FID", "IID", "PAT", "MAT", "SEX", "PHENO"])
        fam = fam.set_index("IID")
        y = fam["PHENO"].replace({-9: np.nan, 0: np.nan}).reindex(sample_index)
        if y.notna().any():
            return y

    return None

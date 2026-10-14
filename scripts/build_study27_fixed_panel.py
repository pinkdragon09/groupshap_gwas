#!/usr/bin/env python3
"""Rebuild the frozen Study 27 fixed panel (81 accessions x 200 SNPs + 3 kinship PCs).

The panel is derived from 1001 Genomes genotypes, the 1001 Genomes kinship matrix
and AraPheno Study 27 phenotypes, none of which this archive redistributes. Run
``scripts/download_data.sh`` first (without ``--kinship-only``), then:

    python scripts/build_study27_fixed_panel.py

The construction reproduces the historical matrix loader of the Study 27 runs:
accessions with a "Mean(TRL) C" value, in genotype-file order; the 200 SNPs listed in
``data/processed/arapheno_study27/study27_mean_trl_top_hits.tsv`` (ascending p-value,
then sorted by index); three standardized eigenvectors of the accession-subset IBS kinship.
``X``/``y`` are float64 (RF runs); ``X_kan``/``y_kan`` are their float32 casts (KAN runs).

The npz container embeds timestamps, so its file checksum is not reproducible. The script
prints a content digest (SHA-256 over names, dtypes, shapes and bytes of ``X_kan``, ``y``,
``y_kan``, ``accessions``, ``snp_indices`` and ``feature_names``) that must equal the value
recorded in ``data/README.md``. The float64 PC columns of ``X`` are excluded because
eigen-decomposition rounding differs across LAPACK builds at the 1e-15 level.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/arapheno_study27/fixed_panel.npz"
FEATURE_METADATA = ROOT / "data/processed/arapheno_study27/feature_metadata.tsv"
TOP_HITS = ROOT / "data/processed/arapheno_study27/study27_mean_trl_top_hits.tsv"
STUDY_VALUES_FILE = "27/study_27_values.csv"
PHENOTYPE_COL = "Mean(TRL) C"
TOP_SNPS = 500   # historical setting; the hit table holds 200 rows, so 200 SNPs are used
N_PCS = 3
KEYS = ("X", "X_kan", "y", "y_kan", "accessions", "snp_indices", "feature_names")
# The float64 PC columns of X depend on the LAPACK build at the 1e-15 level, so the
# recorded content digest covers the bit-stable arrays only.
DIGEST_KEYS = ("X_kan", "y", "y_kan", "accessions", "snp_indices", "feature_names")


# --- historical loader, unchanged in substance -------------------------------------
def load_phenotype(pheno_zip: Path, study_values_file: str, phenotype_col: str) -> dict[str, float]:
    with zipfile.ZipFile(pheno_zip) as zf:
        rows = list(csv.DictReader(io.StringIO(zf.read(study_values_file).decode("utf-8"))))
    y_by_acc: dict[str, float] = {}
    for row in rows:
        acc = row["accession_id"].strip()
        value = row[phenotype_col].strip()
        if not acc or not value:
            continue
        y_by_acc[acc] = float(value)
    if not y_by_acc:
        raise SystemExit(f"No usable phenotype rows found in {study_values_file} for {phenotype_col}.")
    return y_by_acc


def compute_pcs(kinship_h5: Path, common_accs: list[str], n_pcs: int) -> np.ndarray:
    with h5py.File(kinship_h5, "r") as f:
        kinship = f["kinship"][:]
        kin_accs = [x.decode() if isinstance(x, bytes) else str(x) for x in f["accessions"][:]]
    kin_index = {acc: i for i, acc in enumerate(kin_accs)}
    ix = np.asarray([kin_index[acc] for acc in common_accs], dtype=int)
    sub_k = kinship[np.ix_(ix, ix)]
    eigvals, eigvecs = np.linalg.eigh(sub_k)
    pcs = eigvecs[:, -n_pcs:]
    pc_std = pcs.std(axis=0)
    pc_std[pc_std < 1e-8] = 1.0
    pcs = (pcs - pcs.mean(axis=0)) / pc_std
    return pcs


def load_top_snp_indices(baseline_hits: Path, top_snps: int) -> list[int]:
    hits = pd.read_csv(baseline_hits, sep="\t")
    if "snp_index" not in hits.columns:
        raise SystemExit(f"Baseline hit table missing snp_index column: {baseline_hits}")
    hits = hits.sort_values("p_value", ascending=True)
    return hits["snp_index"].astype(int).head(top_snps).tolist()


def chromosome_labels_and_bounds(h5f: h5py.File) -> tuple[list[str], np.ndarray]:
    labels = [f"Chr{x.decode() if isinstance(x, bytes) else x}" for x in h5f["positions"].attrs["chrs"]]
    bounds = h5f["positions"].attrs["chr_regions"]
    return labels, bounds


def chromosome_for_index(idx: int, labels: list[str], bounds: np.ndarray) -> str:
    for chrom_idx, (start, stop) in enumerate(bounds):
        if int(start) <= idx < int(stop):
            return labels[chrom_idx]
    return "Chr?"


def load_matrix(geno_h5: Path, kinship_h5: Path, pheno_zip: Path):
    y_by_acc = load_phenotype(pheno_zip, STUDY_VALUES_FILE, PHENOTYPE_COL)
    with h5py.File(geno_h5, "r") as f:
        geno_accs = [x.decode() if isinstance(x, bytes) else str(x) for x in f["accessions"][:]]
        geno_acc_index = {acc: i for i, acc in enumerate(geno_accs)}
        common_accs = [acc for acc in geno_accs if acc in y_by_acc]
        geno_ix = np.asarray([geno_acc_index[acc] for acc in common_accs], dtype=int)
        sorted_snp_indices = sorted(load_top_snp_indices(TOP_HITS, TOP_SNPS))
        positions = f["positions"][sorted_snp_indices]
        snps = f["snps"][sorted_snp_indices][:, geno_ix].astype(float)
        labels, bounds = chromosome_labels_and_bounds(f)
    pcs = compute_pcs(kinship_h5, common_accs, N_PCS)
    y = pd.Series([y_by_acc[acc] for acc in common_accs], name=PHENOTYPE_COL, dtype=float)
    snp_cols = [f"snp_{idx}" for idx in sorted_snp_indices]
    X = pd.concat([pd.DataFrame(snps.T, columns=snp_cols),
                   pd.DataFrame(pcs, columns=[f"pc{i}" for i in range(1, N_PCS + 1)])], axis=1)
    chroms = [chromosome_for_index(idx, labels, bounds) for idx in sorted_snp_indices]
    return X, y, common_accs, sorted_snp_indices, positions.astype(int), chroms


# --- panel assembly -----------------------------------------------------------------
def content_digest(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in DIGEST_KEYS:
        a = np.ascontiguousarray(arrays[k])
        h.update(f"{k}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def build(geno_h5: Path, kinship_h5: Path, pheno_zip: Path) -> dict[str, np.ndarray]:
    Xdf, ys, accessions, snp_indices, positions, chroms = load_matrix(geno_h5, kinship_h5, pheno_zip)
    if Xdf.shape != (81, 203):
        raise SystemExit(f"Unexpected panel shape {Xdf.shape}; expected (81, 203)")
    X = Xdf.to_numpy(dtype=np.float64)
    y = ys.to_numpy(dtype=np.float64)
    arrays = dict(X=X, X_kan=X.astype(np.float32), y=y, y_kan=y.astype(np.float32),
                  accessions=np.array(accessions, dtype=str), snp_indices=np.asarray(snp_indices, dtype=np.int64),
                  feature_names=np.array([str(c) for c in Xdf.columns], dtype=str))
    meta = pd.read_csv(FEATURE_METADATA, sep="\t")
    if meta.feature.astype(str).tolist() != list(arrays["feature_names"]):
        raise SystemExit("Rebuilt feature order disagrees with feature_metadata.tsv")
    snp_meta = meta[meta.feature_type == "snp"]
    if snp_meta.position.astype(int).tolist() != positions.tolist() or snp_meta.chromosome.tolist() != chroms:
        raise SystemExit("Rebuilt SNP positions or chromosomes disagree with feature_metadata.tsv")
    return arrays


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geno-h5", type=Path, default=ROOT / "data/1001genomes/genotypes/1001_SNP_MATRIX/imputed_snps_binary.hdf5")
    ap.add_argument("--kinship-h5", type=Path, default=ROOT / "data/1001genomes/genotypes/1001_SNP_MATRIX/kinship_ibs_mac5.hdf5")
    ap.add_argument("--pheno-zip", type=Path, default=None,
                    help="AraPheno export (default: data/arapheno/bulk/database.zip, else data/arapheno/database.zip)")
    ap.add_argument("--out", type=Path, default=PANEL)
    ap.add_argument("--expect-digest", default=None, help="fail unless the content digest equals this value")
    ap.add_argument("--compare", type=Path, default=None, help="an existing fixed_panel.npz to compare arrays against")
    args = ap.parse_args()
    pheno = args.pheno_zip or next((p for p in (ROOT / "data/arapheno/bulk/database.zip",
                                                 ROOT / "data/arapheno/database.zip") if p.exists()), None)
    for p in (args.geno_h5, args.kinship_h5, pheno):
        if p is None or not Path(p).exists():
            raise SystemExit(f"Missing input {p}; run scripts/download_data.sh first")
    arrays = build(args.geno_h5, args.kinship_h5, Path(pheno))
    digest = content_digest(arrays)
    print(f"content digest: {digest}")
    if args.expect_digest and digest != args.expect_digest:
        raise SystemExit("Digest mismatch; inputs differ from those behind the archived results")
    if args.compare:
        with np.load(args.compare, allow_pickle=False) as ref:
            for k in KEYS:
                same = ref[k].dtype == arrays[k].dtype and np.array_equal(ref[k], arrays[k])
                note = ""
                if k == "X" and not same:
                    note = f"  (max |diff| {np.abs(ref[k] - arrays[k]).max():.1e}; SNP block identical: {np.array_equal(ref[k][:, :200], arrays[k][:, :200])})"
                print(f"  {k:14s} {'identical' if same else 'differs'}{note}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

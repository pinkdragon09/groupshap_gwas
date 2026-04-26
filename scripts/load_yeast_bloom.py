"""Loader for the BYxRM yeast cross from Bloom et al. 2013, 2015.

Three things this module exposes:

- ``load_geno_2013()``    →  (geno_df, segregant_ids, marker_meta_df)
- ``load_pheno_2013()``   →  pheno_df  (segregants × 46 traits)
- ``load_interactions_2015()`` → interactions_df from Supp Data 4
                                 (CSV converted from the .xls)

The 2013 genotype/phenotype tables are plain TSVs from the Princeton
Yeast Cross BYxRM page; loading is fast (a few seconds end-to-end).
The 2015 interaction list is read from the LibreOffice-converted CSV
in ``processed/`` because no .xls reader is on the project Python env.

The 2015 expansion (4,390 segregants × 28,880 markers) lives in
``processed/Supplementary_Data_5/{genotypes,phenotypes}.RData`` and
needs ``pyreadr`` to load. A loader stub is provided that raises
ImportError with a clear message rather than failing silently.

Genotype encoding: Princeton TSV uses 'B' (BY allele) and 'R' (RM
allele). ``load_geno_2013`` recodes to 0/1 by default
(B=0, R=1; see ``encoding`` keyword for ±1).

See ``data/yeast-byxrm/DESCRIPTION.md`` for source URLs + SHA256s.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "yeast-byxrm"
RAW = DATA_ROOT / "raw"
PROCESSED = DATA_ROOT / "processed"

GENO_2013 = RAW / "BYxRM_GenoData.txt"
PHENO_2013 = RAW / "BYxRM_PhenoData.txt"
INTERACTIONS_2015_CSV = PROCESSED / "Bloom2015_SuppData4_QTLxQTL_interactions.csv"
ADDITIVE_QTL_2015_CSV = PROCESSED / "Bloom2015_SuppData3.csv"

GENOTYPES_RDATA_2015 = PROCESSED / "Supplementary_Data_5" / "genotypes.RData"
PHENOTYPES_RDATA_2015 = PROCESSED / "Supplementary_Data_5" / "phenotypes.RData"

_MARKER_RE = re.compile(r"^(?P<index>\d+)_(?P<chr>chr\d+)_(?P<pos>\d+)_(?P<ref>[ACGT])_(?P<alt>[ACGT])$")


def parse_marker_id(marker_id: str) -> dict[str, str | int]:
    m = _MARKER_RE.match(marker_id)
    if not m:
        return {"marker_id": marker_id, "chr": None, "pos": None, "ref": None, "alt": None}
    g = m.groupdict()
    return {
        "marker_id": marker_id,
        "chr": g["chr"],
        "pos": int(g["pos"]),
        "ref": g["ref"],
        "alt": g["alt"],
    }


def load_geno_2013(
    encoding: Literal["01", "pm1"] = "01",
    path: Path = GENO_2013,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Load the Bloom 2013 Princeton genotype matrix.

    Returns
    -------
    geno_df : pd.DataFrame
        Shape (n_segregants, n_markers). Index = segregant IDs
        (e.g. ``A01_01``), columns = marker IDs. Values are int8:
        ``encoding="01"``  → B=0, R=1
        ``encoding="pm1"`` → B=-1, R=+1
    segregant_ids : list[str]
        Order matches geno_df.index.
    marker_meta_df : pd.DataFrame
        One row per marker. Columns: marker_id, chr, pos, ref, alt.
    """
    raw = pd.read_csv(path, sep="\t", dtype=str)
    marker_ids = raw["marker"].to_numpy()
    segregant_ids = list(raw.columns[1:])

    geno_chars = raw.iloc[:, 1:].to_numpy()  # shape (n_markers, n_segregants)
    if encoding == "01":
        geno = np.where(geno_chars == "R", 1, 0).astype(np.int8)
    elif encoding == "pm1":
        geno = np.where(geno_chars == "R", 1, -1).astype(np.int8)
    else:
        raise ValueError(f"unknown encoding: {encoding!r}")

    geno_df = pd.DataFrame(geno.T, index=segregant_ids, columns=marker_ids)
    marker_meta_df = pd.DataFrame([parse_marker_id(m) for m in marker_ids])
    return geno_df, segregant_ids, marker_meta_df


def load_pheno_2013(path: Path = PHENO_2013) -> pd.DataFrame:
    """Load the Bloom 2013 Princeton phenotype matrix.

    Returns shape (n_segregants, 46). Index = segregant ID, columns =
    trait names. Some cells are NaN (missing measurements).
    """
    pheno = pd.read_csv(path, sep="\t", index_col=0)
    pheno.index.name = "segregant_id"
    return pheno


def load_interactions_2015(path: Path = INTERACTIONS_2015_CSV) -> pd.DataFrame:
    """Load the Bloom 2015 published QTL × QTL interaction list (Supp Data 4).

    205 interactions across 19 traits. Columns include trait,
    Q1_chr, Q1_pos, Q2_chr, Q2_pos, LOD, lm_coefficient,
    variance_explained.
    """
    return pd.read_csv(path)


def load_additive_qtls_2015(path: Path = ADDITIVE_QTL_2015_CSV) -> pd.DataFrame:
    """Load the Bloom 2015 additive-QTL list (Supp Data 3, post-conversion).

    Header row in this CSV starts at line 2 because the source XLS
    has a banner row. We skip it.
    """
    return pd.read_csv(path, header=1)


def load_geno_2015():
    """Stub for the 2015 expansion (4,390 segregants × 28,880 markers).

    Requires ``pyreadr`` (not currently on the project Python env).
    Raises ImportError with a clear message.
    """
    try:
        import pyreadr
    except ImportError as e:
        raise ImportError(
            "load_geno_2015() requires pyreadr to read the .RData file. "
            "Install the pyreadr dependency in the project environment and re-run."
        ) from e
    result = pyreadr.read_r(str(GENOTYPES_RDATA_2015))
    return result


def smoke_test_phenotype(name: str = "Raffinose") -> dict:
    """Return summary stats for a phenotype to confirm the loader works."""
    pheno = load_pheno_2013()
    if name not in pheno.columns:
        raise KeyError(f"trait {name!r} not in phenotype columns: {list(pheno.columns)}")
    s = pheno[name]
    return {
        "trait": name,
        "n_total": int(len(s)),
        "n_nonmissing": int(s.notna().sum()),
        "n_missing": int(s.isna().sum()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
    }


if __name__ == "__main__":
    import json
    print("Loading 2013 genotypes…")
    geno, ids, meta = load_geno_2013()
    print(f"  geno shape: {geno.shape} (segregants × markers)")
    print(f"  encoding: 0={int((geno.values == 0).sum())}  1={int((geno.values == 1).sum())}")
    print(f"  chromosomes: {sorted(meta['chr'].dropna().unique())}")

    print("Loading 2013 phenotypes…")
    pheno = load_pheno_2013()
    print(f"  pheno shape: {pheno.shape}")
    print(f"  traits: {list(pheno.columns)[:5]} … (total {pheno.shape[1]})")

    print("Loading 2015 interaction list…")
    inter = load_interactions_2015()
    print(f"  interactions: {len(inter)} rows, {inter['trait'].nunique()} traits")

    print("Smoke test on Raffinose:")
    print(json.dumps(smoke_test_phenotype("Raffinose"), indent=2))

"""Load the frozen Study 27 fixed panel used by the Study 27 analyses.

The archive includes the 81-accession, 200-SNP plus three-PC matrix so that
fixed-panel RF/KAN analyses can be rerun without the 919 MB genome-wide HDF5
matrix. ``X``/``y`` preserve the float64 values used by RF; ``X_kan``/``y_kan``
preserve the float32 values used by the saved KAN fits.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/arapheno_study27/fixed_panel.npz"
FEATURE_METADATA = ROOT / "data/processed/arapheno_study27/feature_metadata.tsv"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fixed_panel(*, kan_dtype: bool = False):
    """Return ``X, y, feature metadata, accessions`` for the frozen panel."""
    with np.load(PANEL, allow_pickle=False) as saved:
        x_key, y_key = ("X_kan", "y_kan") if kan_dtype else ("X", "y")
        names = saved["feature_names"].astype(str)
        X = pd.DataFrame(saved[x_key], columns=names)
        y = pd.Series(saved[y_key], name="Mean(TRL) C")
        accessions = saved["accessions"].astype(str)
    meta = pd.read_csv(FEATURE_METADATA, sep="\t")
    required = {"feature", "feature_type", "chromosome", "snp_index", "position"}
    if not required.issubset(meta.columns):
        raise ValueError("Invalid Study 27 feature metadata")
    if X.shape != (81, 203) or y.shape != (81,) or len(accessions) != 81:
        raise ValueError("Unexpected frozen Study 27 panel dimensions")
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(y.to_numpy()).all():
        raise ValueError("Frozen Study 27 panel contains non-finite values")
    if list(X.columns) != meta.feature.astype(str).tolist():
        raise ValueError("Frozen Study 27 feature order disagrees with metadata")
    if meta.feature_type.eq("snp").sum() != 200 or meta.feature_type.eq("covariate").sum() != 3:
        raise ValueError("Frozen Study 27 panel does not contain 200 SNPs and three PCs")
    return X, y, meta, accessions

#!/usr/bin/env python3
"""Compare frozen historical ALE with PyALE without overwriting old outputs.

Run: .venv-kan/bin/python scripts/audit_ale_migration.py [--phase5]
"""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests/fixtures"))
from ale_legacy import upper_only_ale
from ale_cases import Oracle, cohort
from xai_gwas.attribution.ale import ale_effect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase5", action="store_true")
    parser.add_argument("--outdir", type=Path, default=ROOT / "docs/ale_migration_evidence")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    records = []
    for levels in [2, 3]:
        for correlated in [False, True]:
            for interaction in [0, 4]:
                X = cohort(levels, correlated); model = Oracle(interaction)
                old = upper_only_ale(model, X, "snp")
                new = ale_effect(model, X, "snp")
                for k in range(levels):
                    records.append(dict(levels=levels, correlated=correlated,
                        interaction=interaction, x=k, legacy=old.ale[k], pyale=new.ale[k],
                        change=new.ale[k]-old.ale[k]))
    pd.DataFrame(records).to_csv(args.outdir / "synthetic_comparison.tsv", sep="\t", index=False)
    meta = {"PyALE": importlib.metadata.version("PyALE"),
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": importlib.metadata.version("scikit-learn"),
            "legacy_source_commit": "293d22074db0c35a4cc32ed603b102faa1843328"}
    if args.phase5:
        import run_phase5 as phase
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import KFold
        X, y, _ = phase.load_arabidopsis_matrix()
        tr, _ = next(iter(KFold(n_splits=5, shuffle=True, random_state=42).split(X)))
        model = RandomForestRegressor(n_estimators=500, max_features="sqrt",
            min_samples_leaf=2, random_state=43, n_jobs=1).fit(X.iloc[tr], y.iloc[tr])
        saved_path = ROOT / "outputs/phase5/ale_results.tsv"
        meta["historical_sha256"] = hashlib.sha256(saved_path.read_bytes()).hexdigest()
        saved = pd.read_csv(saved_path, sep="\t")
        comparisons = []
        for snp in saved.feature.unique():
            old = upper_only_ale(model, X, snp).rename(columns={"ale": "legacy_recomputed"})
            new = ale_effect(model, X, snp).rename(columns={"ale": "pyale"})
            row = saved[saved.feature == snp].rename(columns={"ale": "historical_saved"})
            row = row.merge(old[["x", "legacy_recomputed"]], on="x").merge(new[["x", "pyale"]], on="x")
            row["estimator_change"] = row.pyale-row.legacy_recomputed
            row["reproduction_difference"] = row.legacy_recomputed-row.historical_saved
            comparisons.append(row)
        result = pd.concat(comparisons, ignore_index=True)
        result.to_csv(args.outdir / "phase5_comparison.tsv", sep="\t", index=False)
        meta["max_historical_reproduction_difference"] = float(result.reproduction_difference.abs().max())
        print(result.to_string(index=False))
    (args.outdir / "provenance.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(f"Audit saved to {args.outdir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Study 27: training-fold GWAS selection and matched RF/Transformer/KAN CV.

Run from the repository root with .venv-kan/bin/python. No shap dependency.
The chromosome-level Shapley game treats all PCs as a separate player.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from kan import KAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from threadpoolctl import threadpool_limits
from full_baseline_gwas import load_phenotype, compute_pcs, chrom_labels_and_bounds
from transformer_mamba_gwas_template import TransformerGWASRegressor

ROOT = Path(__file__).resolve().parents[1]
GENO = ROOT / "data/1001genomes/genotypes/1001_SNP_MATRIX/imputed_snps_binary.hdf5"
KIN = GENO.with_name("kinship_ibs_mac5.hdf5")
PHENO = ROOT / "data/arapheno/bulk/database.zip"


def rank_chunk(G, y, pcs, min_mac=2):
    """PC-adjusted marginal ranking; callers supply training rows only."""
    C = np.column_stack([np.ones(len(y)), pcs])
    Q = np.linalg.svd(C, full_matrices=False)[0][:, :np.linalg.matrix_rank(C)]
    yr = y - Q @ (Q.T @ y)
    gr = G - (G @ Q) @ Q.T
    ss = np.einsum("ij,ij->i", gr, gr)
    mac = np.minimum(G.sum(1), len(y) - G.sum(1))
    scores = np.full(len(G), -np.inf)
    ok = (ss > 1e-10) & (mac >= min_mac)
    scores[ok] = (gr[ok] @ yr) ** 2 / ss[ok]
    return scores


def top_indices(indices, scores, k):
    # Deterministic SNP-index tie breaking, including LD-equivalent variants.
    order = np.lexsort((indices, -scores))[:k]
    return indices[order], scores[order]


def prepare(args, out):
    ymap, _ = load_phenotype(PHENO, "27/study_27_values.csv", "Mean(TRL) C")
    with h5py.File(GENO) as f:
        accessions = [a.decode() if isinstance(a, bytes) else str(a) for a in f["accessions"][:]]
        aix = np.array([i for i, a in enumerate(accessions) if a in ymap])
        acc = [accessions[i] for i in aix]
        n_snps = f["snps"].shape[0]
    y = np.array([ymap[a] for a in acc], dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Non-finite phenotype")
    pcs = compute_pcs(KIN, acc, args.n_pcs) if args.n_pcs else np.empty((len(y), 0))
    folds = list(KFold(5, shuffle=True, random_state=42).split(y))
    jobs = []
    for fold, (tr, te) in enumerate(folds):
        inner_tr, val = train_test_split(tr, test_size=0.2, random_state=1042 + fold)
        jobs.extend([(f"outer{fold}", tr), (f"inner{fold}", np.sort(inner_tr))])
    identity = dict(top_snps=args.top_snps, n_pcs=args.n_pcs, accessions=acc,
                    phenotype=y.tolist(), split_seed=42, inner_seed=1042,
                    scan_version=1, sources={str(p): [p.stat().st_size, p.stat().st_mtime_ns]
                                             for p in [GENO, KIN, PHENO]})
    cache = out / "panels.npz"
    manifest = out / "panel_manifest.json"
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text()) == identity:
        panels = dict(np.load(cache))
        print("Using verified training-fold panel cache", flush=True)
    else:
        best = {name: (np.array([], dtype=int), np.array([], dtype=float)) for name, _ in jobs}
        t0 = time.time()
        with h5py.File(GENO) as f:
            for start in range(0, n_snps, 10000):
                G = f["snps"][start:start + 10000, :][:, aix].astype(float)
                idx = np.arange(start, start + len(G))
                for name, tr in jobs:
                    score = rank_chunk(G[:, tr], y[tr], pcs[tr])
                    ii, ss = top_indices(idx, score, args.top_snps)
                    bi, bs = best[name]
                    best[name] = top_indices(np.r_[bi, ii], np.r_[bs, ss], args.top_snps)
                if start % 500000 == 0:
                    print(f"GWAS selection: {start:,}/{n_snps:,} SNPs, {time.time()-t0:.0f}s", flush=True)
            panels = {}
            labels, bounds = chrom_labels_and_bounds(f)
            for name, _ in jobs:
                ix, score = best[name]
                if len(ix) < args.top_snps or not np.isfinite(score).all():
                    raise ValueError("Insufficient polymorphic training SNPs")
                ix = np.sort(ix)
                panels[name] = f["snps"][ix, :][:, aix].T.astype(np.float32)
                panels[name + "_indices"] = ix
                positions = f["positions"][ix]
                chroms = [next(c for c, (lo, hi) in zip(labels, bounds) if lo <= i < hi) for i in ix]
                panels[name + "_chroms"] = np.array(chroms)
                pd.DataFrame(dict(snp_index=ix, chromosome=chroms, position=positions)).to_csv(
                    out / (name + "_panel.tsv"), sep="\t", index=False)
        np.savez_compressed(cache, **panels)
        manifest.write_text(json.dumps(identity, indent=2))
    pd.DataFrame(dict(accession=acc, phenotype=y)).to_csv(out / "accessions.tsv", sep="\t", index=False)
    splits = []
    for fold, (tr, te) in enumerate(folds):
        it, va = train_test_split(tr, test_size=0.2, random_state=1042 + fold)
        for role, ix in [("inner_train", it), ("inner_validation", va), ("outer_test", te)]:
            splits.extend(dict(fold=fold, role=role, accession=acc[i], row=int(i)) for i in ix)
    pd.DataFrame(splits).to_csv(out / "splits.tsv", sep="\t", index=False)
    return y, pcs, folds, panels, acc


def scale_inputs(G, pcs, tr):
    # Binary SNPs mapped into the fixed spline grid; covariates fitted on training rows.
    if pcs.shape[1]:
        sd = np.maximum(pcs[tr].std(0), 1e-8)
        P = (pcs - pcs[tr].mean(0)) / sd
        return np.column_stack([2 * G - 1, P / 3]).astype(np.float32)
    return (2 * G - 1).astype(np.float32)


def make_model(name, p, c, seed, out):
    torch.manual_seed(seed)
    if name == "kan":
        return KAN(width=[p+c, 4, 1], grid=3, k=3, seed=seed,
                   auto_save=False, symbolic_enabled=False, ckpt_path=str(out / "checkpoints"))
    return TransformerGWASRegressor(p, c, d_model=32, n_heads=4, n_layers=1, dropout=0.1)


def fit_neural(name, X, y, tr, seed, args, out, val=None, epochs=None):
    model = make_model(name, args.top_snps, args.n_pcs, seed, out)
    mu, sd = float(y[tr].mean()), max(float(y[tr].std()), 1e-8)
    Xt = torch.as_tensor(X)
    yt = torch.as_tensor((y - mu) / sd, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    best, best_epoch, stale = float("inf"), 1, 0
    count = epochs if epochs is not None else args.epochs
    for epoch in range(count):
        model.train()
        opt.zero_grad()
        pred = model(Xt[tr]).reshape(-1)
        loss = ((pred - yt[tr]) ** 2).mean()
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite training loss: {name}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if val is not None:
            model.eval()
            with torch.no_grad():
                score = float(((model(Xt[val]).reshape(-1) - yt[val]) ** 2).mean())
            if score < best - 1e-6:
                best, best_epoch, stale = score, epoch + 1, 0
            else:
                stale += 1
            if stale >= args.patience:
                break
    model.eval()
    return model, mu, sd, (best_epoch if val is not None else count)


def predictor(model, mu=0.0, sd=1.0):
    if hasattr(model, "predict"):
        return model.predict
    def predict(X):
        with torch.no_grad():
            return np.concatenate([model(torch.as_tensor(a, dtype=torch.float32)).numpy().reshape(-1)
                                   for a in np.array_split(X, max(1, math.ceil(len(X) / 128)))]) * sd + mu
    return predict


def exact_group_game(predict, X, tr, te, chroms, seed):
    """Exact coalition enumeration; empirical background sampled from training only.

    Interaction uses the Shapley interaction index with pair weight
    |S|!(m-|S|-2)!/(m-1)!, not the half-sized SHAP interaction matrix entry.
    """
    labels = sorted(set(chroms))
    groups = [np.flatnonzero(chroms == c) for c in labels]
    if X.shape[1] > len(chroms):
        labels.append("PCs")
        groups.append(np.arange(len(chroms), X.shape[1]))
    m = len(groups)
    bg = np.random.default_rng(seed).choice(tr, min(8, len(tr)), replace=False)
    values = []
    for mask in range(1 << m):
        z = np.tile(X[bg], (len(te), 1))
        for j, cols in enumerate(groups):
            if mask & (1 << j):
                z[:, cols] = np.repeat(X[te][:, cols], len(bg), axis=0)
        values.append(predict(z).reshape(len(te), len(bg)).mean(1))
    v = np.array(values)
    phi = np.zeros((len(te), m))
    pairs = []
    for j in range(m):
        for mask in range(1 << m):
            if mask & (1 << j):
                continue
            s = mask.bit_count()
            w = math.factorial(s) * math.factorial(m-s-1) / math.factorial(m)
            phi[:, j] += w * (v[mask | (1 << j)] - v[mask])
        for k in range(j+1, m):
            interaction = np.zeros(len(te))
            for mask in range(1 << m):
                if mask & ((1 << j) | (1 << k)):
                    continue
                s = mask.bit_count()
                w = math.factorial(s) * math.factorial(m-s-2) / math.factorial(m-1)
                interaction += w * (v[mask | (1 << j) | (1 << k)] - v[mask | (1 << j)]
                                    - v[mask | (1 << k)] + v[mask])
            pairs.append((labels[j], labels[k], float(np.abs(interaction).mean())))
    err = float(np.max(np.abs(phi.sum(1) - (v[-1] - v[0]))))
    if err > 1e-4:
        raise AssertionError(f"Shapley efficiency error {err}")
    return labels, phi, pairs, err


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=ROOT / "outputs/study27_kan_comparison")
    p.add_argument("--n-pcs", type=int, default=3)
    p.add_argument("--top-snps", type=int, default=200)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--models", nargs="+", choices=["random_forest", "transformer", "kan"],
                   default=["random_forest", "transformer", "kan"])
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    if args.n_pcs < 0 or args.top_snps < 1 or args.epochs < 1 or args.patience < 1:
        p.error("Invalid configuration")
    out = args.outdir.resolve()
    if (out / "cv_metrics.tsv").exists() or (out / "COMPLETE.json").exists():
        raise SystemExit("This output directory already contains fit results; choose a new --outdir. "
                         "Panel-only preparation directories may be reused.")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    config = vars(args).copy()
    config["outdir"] = str(out)
    config.update(torch_version=torch.__version__,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  selection="genome-wide training-only PC-adjusted marginal GWAS, MAC >= 2",
                  pcs="kinship PCs on all matched genotypes, no phenotype used; transductive",
                  uncertainty="fold SD is descriptive, not a confidence interval")
    (out / "run_config.json").write_text(json.dumps(config, indent=2))
    with threadpool_limits(limits=2):
        y, pcs, folds, panels, acc = prepare(args, out)
        if args.prepare_only:
            return
        metrics, predictions, groups, interactions, local = [], [], [], [], []
        for fold, (tr, te) in enumerate(folds):
            it, va = train_test_split(tr, test_size=0.2, random_state=1042 + fold)
            G = panels[f"outer{fold}"]
            X = scale_inputs(G, pcs, tr)
            Xi = scale_inputs(panels[f"inner{fold}"], pcs, it)
            if args.n_pcs:
                base = Ridge(alpha=1.0).fit(X[tr, args.top_snps:], y[tr]).predict(X[te, args.top_snps:])
            else:
                base = np.repeat(y[tr].mean(), len(te))
            metrics.append(dict(model="pc_only", seed=-1, fold=fold, r2=r2_score(y[te], base),
                                rmse=mean_squared_error(y[te], base)**0.5, selected_epochs=0))
            predictions.extend(dict(model="pc_only", seed=-1, fold=fold, accession=acc[i],
                                    observed=y[i], predicted=float(v)) for i, v in zip(te, base))
            for seed in args.seeds:
                for name in args.models:
                    t0 = time.time()
                    if name == "random_forest":
                        model = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                    min_samples_leaf=2, random_state=seed+fold, n_jobs=2).fit(X[tr], y[tr])
                        predict = predictor(model)
                        ep = 0
                    else:
                        _, _, _, ep = fit_neural(name, Xi, y, it, seed+fold, args, out, val=va)
                        model, mu, sd, _ = fit_neural(name, X, y, tr, seed+fold, args, out, epochs=ep)
                        predict = predictor(model, mu, sd)
                    pred = predict(X[te])
                    row = dict(model=name, seed=seed, fold=fold, r2=r2_score(y[te], pred),
                               rmse=mean_squared_error(y[te], pred)**0.5, selected_epochs=ep)
                    metrics.append(row)
                    predictions.extend(dict(model=name, seed=seed, fold=fold, accession=acc[i],
                                            observed=y[i], predicted=float(v)) for i, v in zip(te, pred))
                    labels, phi, pairs, err = exact_group_game(
                        predict, X, tr, te, panels[f"outer{fold}_chroms"], 9000+fold)
                    for j, label in enumerate(labels):
                        groups.append(dict(model=name, seed=seed, fold=fold, group=label,
                                           mean_abs_shap=float(np.abs(phi[:, j]).mean()), efficiency_error=err))
                        local.extend(dict(model=name, seed=seed, fold=fold, accession=acc[i],
                                          group=label, shap=float(phi[t, j])) for t, i in enumerate(te))
                    interactions.extend(dict(model=name, seed=seed, fold=fold, group1=a, group2=b,
                                             mean_abs_interaction=v) for a, b, v in pairs)
                    print(f"fold={fold} seed={seed} {name}: R2={row['r2']:.4f} epochs={ep} "
                          f"time={time.time()-t0:.1f}s", flush=True)
                    for filename, rows in [("cv_metrics", metrics), ("predictions", predictions),
                                           ("group_shap", groups), ("group_interactions", interactions),
                                           ("local_group_shap", local)]:
                        pd.DataFrame(rows).to_csv(out / (filename + ".tsv"), sep="\t", index=False)
        m = pd.DataFrame(metrics)
        # Average seeds within a fold first; folds are the evaluation units.
        mf = m.groupby(["model", "fold"])[["r2", "rmse"]].mean().reset_index()
        summary = mf.groupby("model").agg(mean_cv_r2=("r2", "mean"), sd_cv_r2=("r2", "std"),
                                           mean_cv_rmse=("rmse", "mean"), folds=("fold", "count"))
        summary.to_csv(out / "model_summary.tsv", sep="\t")
        print(summary.to_string(), flush=True)
        (out / "COMPLETE.json").write_text(json.dumps(dict(metrics_rows=len(metrics),
                n_accessions=len(y), models=args.models, seeds=args.seeds), indent=2))


if __name__ == "__main__":
    main()

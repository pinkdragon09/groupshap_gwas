"""Cross-shape KAN ablation: vary hidden width and B-spline grid around
the (hidden=8, grid=5) baseline on the Bloom 2015 epistasis benchmark.

Reuses the baseline pipeline functions in
``scripts/yeast_bloom_kan_train.py`` and
``scripts/yeast_bloom_kan_interaction.py`` for apples-to-apples
comparison (identical panel, identical CV split, identical pairwise-
interaction Monte-Carlo settings).

Per-config outputs land at::

    outputs/yeast_bloom_kan/ablation/<phenotype>/h<H>_g<G>/
        cv_metrics.json
        train_log.json
        in_sample_r2.json
        model.pt
        pairwise_interactions.csv
        panel_meta.json
        run.log         (caller redirects stdout/stderr here)

CLI::

    python scripts/yeast_bloom_kan_ablation.py \\
        --phenotype Lactate --hidden 16 --grid 5 \\
        --outdir outputs/yeast_bloom_kan/ablation/Lactate/h16_g5/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from yeast_bloom_kan_train import (  # noqa: E402
    load_phenotype, make_kan, train_kan, cv_r2,
)
from yeast_bloom_kan_interaction import (  # noqa: E402
    pairwise_interaction_kan_fast,
)

DATA = ROOT / "data_cache"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phenotype", required=True, choices=["Neomycin", "Lactate"])
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--grid", type=int, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--k", type=int, default=3, help="B-spline order (matches baseline)")
    p.add_argument("--seed", type=int, default=42)
    # Match baseline cv_metrics.json: epochs=200, patience=25.
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--mc-samples", type=int, default=30)
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()

    out_dir: Path = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "_pykan_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "hidden": args.hidden, "grid": args.grid, "k": args.k,
        "epochs": args.epochs, "patience": args.patience,
        "lr": args.lr, "batch_size": args.batch_size, "seed": args.seed,
        "phenotype": args.phenotype,
    }
    print(f"=== ablation cell: phen={args.phenotype} hidden={args.hidden} "
          f"grid={args.grid} k={args.k} ===", flush=True)
    print(f"  config: {config}", flush=True)
    print(f"  out_dir: {out_dir}", flush=True)

    # ---- Load panel data (same files the baseline used) ----
    print(f"=== Loading {args.phenotype} ===", flush=True)
    data = load_phenotype(args.phenotype)
    X = data["X_panel"]
    y = data["y"]
    n, p_in = X.shape
    print(f"  X_panel: {X.shape}, y: {y.shape}, panel cols: {p_in}", flush=True)

    panel_meta = {
        "phenotype": args.phenotype,
        "n_samples": int(n),
        "p_panel": int(p_in),
        "panel_cols_in_full_dim": data["panel_cols_in_full"].tolist(),
        "marker_ids": data["panel"]["marker_id"].tolist(),
        "panel_chr": data["panel"]["chr"].tolist(),
        "panel_pos": data["panel"]["pos"].astype(int).tolist(),
        "neg_log10_p": data["panel"]["neg_log10_p"].astype(float).tolist(),
        "bg_idx": data["bg_idx"].tolist(),
        "eval_idx": data["eval_idx"].tolist(),
        "config": config,
    }
    (out_dir / "panel_meta.json").write_text(json.dumps(panel_meta, indent=2))

    # ---- 5-fold CV (same KFold seed=42 as baseline) ----
    print("=== 5-fold CV ===", flush=True)
    t0 = time.time()
    cv = cv_r2(
        X, y,
        hidden=args.hidden, grid=args.grid, k=args.k, seed=args.seed,
        epochs=args.epochs, patience=args.patience, lr=args.lr,
        batch_size=args.batch_size, ckpt_dir=ckpt_dir,
    )
    cv["cv_seconds"] = time.time() - t0
    cv["config"] = config
    (out_dir / "cv_metrics.json").write_text(json.dumps(cv, indent=2))
    print(f"  CV R^2: mean={cv['mean_r2']:.4f}  sd={cv['sd_r2']:.4f}  "
          f"({cv['cv_seconds']:.1f}s)", flush=True)

    # ---- Final fit on full panel (90/10 holdout for early stop) ----
    print("=== Final fit on full panel ===", flush=True)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, n // 10)
    Xv = X[perm[:n_val]]; yv = y[perm[:n_val]]
    Xtr = X[perm[n_val:]]; ytr = y[perm[n_val:]]
    model = make_kan(p_in, args.hidden, args.grid, args.k, args.seed, ckpt_dir)
    t0 = time.time()
    log = train_kan(
        model, Xtr, ytr, Xv, yv,
        epochs=args.epochs, patience=args.patience,
        lr=args.lr, batch_size=args.batch_size, seed=args.seed,
    )
    log["fit_seconds"] = time.time() - t0
    log["config"] = config
    print(f"  best val MSE = {log['best_val_mse']:.4f} at epoch "
          f"{log['best_epoch']}/{log['n_epochs_run']}; fit took "
          f"{log['fit_seconds']:.1f}s", flush=True)

    model.eval()
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))

    # In-sample R^2 sanity
    with torch.no_grad():
        yhat_full = model(torch.as_tensor(X, dtype=torch.float32)).numpy().ravel()
    r2_full = float(r2_score(y, yhat_full))
    (out_dir / "in_sample_r2.json").write_text(
        json.dumps({"in_sample_r2": r2_full}, indent=2))
    print(f"  in-sample R^2: {r2_full:.4f}", flush=True)

    # ---- Pairwise interaction sweep ----
    print("=== Pairwise interaction sweep ===", flush=True)
    bg_idx = np.asarray(data["bg_idx"], dtype=np.int64)
    eval_idx = np.asarray(data["eval_idx"], dtype=np.int64)
    X_bg = X[bg_idx]
    X_eval = X[eval_idx]
    cand_idx = np.arange(p_in, dtype=np.int64)
    marker_ids = data["panel"]["marker_id"].tolist()

    t0 = time.time()
    inter_df = pairwise_interaction_kan_fast(
        out_dir / "model.pt",
        p_in=p_in, hidden=args.hidden, grid=args.grid, k=args.k, seed=args.seed,
        ckpt_dir=ckpt_dir,
        X_bg=X_bg, X_eval=X_eval,
        candidate_indices=cand_idx,
        candidate_labels=marker_ids,
        n_samples=args.mc_samples,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbose=True,
    )
    inter_seconds = time.time() - t0
    inter_df.to_csv(out_dir / "pairwise_interactions.csv", index=False)
    print(f"  pairwise sweep done in {inter_seconds:.1f}s "
          f"({len(inter_df)} pairs)", flush=True)
    print("  top 5 by |I_jk|:")
    print(inter_df.head(5)[
        ["marker_j", "marker_k", "mean_abs_interaction", "mean_interaction"]
    ].to_string(index=False))

    # Summary record (lightweight, for later aggregation)
    summary = {
        "phenotype": args.phenotype,
        "config": config,
        "cv_mean_r2": cv["mean_r2"],
        "cv_sd_r2": cv["sd_r2"],
        "cv_seconds": cv["cv_seconds"],
        "in_sample_r2": r2_full,
        "fit_seconds": log["fit_seconds"],
        "interaction_seconds": inter_seconds,
        "n_pairs": int(len(inter_df)),
        "max_abs_I": float(inter_df["mean_abs_interaction"].max()),
        "median_abs_I": float(inter_df["mean_abs_interaction"].median()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("=== ablation cell done ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

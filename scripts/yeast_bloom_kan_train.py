"""Train a 2-layer KAN on the Bloom2013 top-100 LD-pruned marker panel.

For each phenotype:
  1. Load X_filtered, y, panel_topN, bg_idx, eval_idx from the cached
     RF-baseline outputs (data_cache/<phenotype>/).
  2. Build X_panel = X_filtered[:, panel.filtered_index].
  3. Train KAN with (width=[p_panel, hidden, 1], grid, k) — torch loop,
     Adam, MSE, early stopping on val MSE.
  4. Compute 5-fold CV R^2 (KFold seed=42, identical to RF baseline).
  5. Save: outputs/yeast_bloom_kan/<pheno>/model.pt, cv_metrics.json,
     train_log.json, panel_meta.json.

CLI:
    python3 yeast_bloom_kan_train.py <phenotype> [--hidden 8] [--grid 5] [--k 3]
                                                 [--epochs 500] [--patience 50]
                                                 [--lr 1e-3] [--batch-size 64]
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
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_cache"
OUT = ROOT / "outputs" / "yeast_bloom_kan"

# Pin pykan checkpoint dir under outputs/ to avoid cwd pollution.
CKPT_BASE = OUT / "_pykan_ckpt"
CKPT_BASE.mkdir(parents=True, exist_ok=True)

from kan import KAN  # noqa: E402


def load_phenotype(phenotype: str):
    d = DATA / phenotype
    X_full = np.load(d / "X_filtered.npy")
    y = np.load(d / "y.npy").astype(np.float32)
    panel = pd.read_csv(d / "panel_topN.csv")
    bg_idx = np.load(d / "bg_idx.npy")
    eval_idx = np.load(d / "eval_idx.npy")
    panel_cols = panel["filtered_index"].astype(int).to_numpy()
    X_panel = X_full[:, panel_cols].astype(np.float32)
    return {
        "X_panel": X_panel,
        "y": y,
        "panel": panel,
        "panel_cols_in_full": panel_cols,
        "bg_idx": bg_idx,
        "eval_idx": eval_idx,
    }


def make_kan(p_in: int, hidden: int, grid: int, k: int, seed: int, ckpt_dir: Path):
    return KAN(
        width=[p_in, hidden, 1], grid=grid, k=k, seed=seed,
        ckpt_path=str(ckpt_dir), auto_save=False,
    )


def train_kan(
    model: torch.nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    epochs: int, patience: int, lr: float, batch_size: int, seed: int,
) -> dict:
    """Adam loop with MSE loss + early stopping on val MSE.

    Returns: dict with train_losses, val_losses, best_epoch, best_val_mse.
    """
    torch.manual_seed(seed)
    Xt = torch.as_tensor(X_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.float32).unsqueeze(1)
    Xv = torch.as_tensor(X_val, dtype=torch.float32)
    yv = torch.as_tensor(y_val, dtype=torch.float32).unsqueeze(1)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xt.shape[0]
    n_batches = max(1, n // batch_size)

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    no_improve = 0
    train_losses, val_losses = [], []

    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            xb = Xt[idx]; yb = yt[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = ((pred - yb) ** 2).mean()
            loss.backward()
            opt.step()
            epoch_loss += float(loss) * len(idx)
        epoch_loss /= n
        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_mse = float(((val_pred - yv) ** 2).mean())
        train_losses.append(epoch_loss)
        val_losses.append(val_mse)
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "n_epochs_run": len(train_losses),
    }


def cv_r2(
    X: np.ndarray, y: np.ndarray, *,
    hidden: int, grid: int, k: int, seed: int,
    epochs: int, patience: int, lr: float, batch_size: int,
    ckpt_dir: Path,
) -> dict:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2s = []
    for fold, (tr, te) in enumerate(kf.split(X)):
        Xtr_full, Xte = X[tr], X[te]
        ytr_full, yte = y[tr], y[te]
        # 90/10 split inside the train fold for early-stopping val.
        n_tr = len(Xtr_full)
        rng = np.random.default_rng(seed + fold)
        idx_perm = rng.permutation(n_tr)
        n_val = max(1, n_tr // 10)
        val_idx = idx_perm[:n_val]
        tr_idx = idx_perm[n_val:]
        Xtr, Xv = Xtr_full[tr_idx], Xtr_full[val_idx]
        ytr, yv = ytr_full[tr_idx], ytr_full[val_idx]

        model = make_kan(X.shape[1], hidden, grid, k, seed + fold, ckpt_dir)
        train_kan(model, Xtr, ytr, Xv, yv,
                  epochs=epochs, patience=patience, lr=lr,
                  batch_size=batch_size, seed=seed + fold)
        model.eval()
        with torch.no_grad():
            yhat = model(torch.as_tensor(Xte, dtype=torch.float32)).numpy().ravel()
        r2s.append(float(r2_score(yte, yhat)))
        print(f"  fold {fold}: R²={r2s[-1]:.4f}", flush=True)
    return {
        "fold_r2": r2s,
        "mean_r2": float(np.mean(r2s)),
        "sd_r2": float(np.std(r2s, ddof=0)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phenotype")
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--grid", type=int, default=5)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-cv", action="store_true",
                   help="Skip 5-fold CV (use only the final-fit model)")
    args = p.parse_args()

    ph = args.phenotype
    out_dir = OUT / ph
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = CKPT_BASE / ph
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Loading {ph} ===", flush=True)
    data = load_phenotype(ph)
    X = data["X_panel"]
    y = data["y"]
    n, p_in = X.shape
    print(f"  X_panel: {X.shape}, y: {y.shape}, panel cols: {p_in}", flush=True)

    panel_meta = {
        "phenotype": ph,
        "n_samples": int(n),
        "p_panel": int(p_in),
        "panel_cols_in_full_dim": data["panel_cols_in_full"].tolist(),
        "marker_ids": data["panel"]["marker_id"].tolist(),
        "panel_chr": data["panel"]["chr"].tolist(),
        "panel_pos": data["panel"]["pos"].astype(int).tolist(),
        "neg_log10_p": data["panel"]["neg_log10_p"].astype(float).tolist(),
        "bg_idx": data["bg_idx"].tolist(),
        "eval_idx": data["eval_idx"].tolist(),
    }
    (out_dir / "panel_meta.json").write_text(json.dumps(panel_meta, indent=2))

    if not args.skip_cv:
        print("=== 5-fold CV ===", flush=True)
        t0 = time.time()
        cv = cv_r2(X, y,
                   hidden=args.hidden, grid=args.grid, k=args.k, seed=args.seed,
                   epochs=args.epochs, patience=args.patience, lr=args.lr,
                   batch_size=args.batch_size, ckpt_dir=ckpt_dir)
        cv["cv_seconds"] = time.time() - t0
        cv["config"] = {
            "hidden": args.hidden, "grid": args.grid, "k": args.k,
            "epochs": args.epochs, "patience": args.patience,
            "lr": args.lr, "batch_size": args.batch_size, "seed": args.seed,
        }
        (out_dir / "cv_metrics.json").write_text(json.dumps(cv, indent=2))
        print(f"  CV R^2: mean={cv['mean_r2']:.4f}  sd={cv['sd_r2']:.4f}", flush=True)

    print("=== Final fit on full panel ===", flush=True)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, n // 10)
    Xv = X[perm[:n_val]]; yv = y[perm[:n_val]]
    Xtr = X[perm[n_val:]]; ytr = y[perm[n_val:]]
    model = make_kan(p_in, args.hidden, args.grid, args.k, args.seed, ckpt_dir)
    t0 = time.time()
    log = train_kan(model, Xtr, ytr, Xv, yv,
                    epochs=args.epochs, patience=args.patience,
                    lr=args.lr, batch_size=args.batch_size, seed=args.seed)
    log["fit_seconds"] = time.time() - t0
    log["config"] = {
        "hidden": args.hidden, "grid": args.grid, "k": args.k,
        "epochs": args.epochs, "patience": args.patience,
        "lr": args.lr, "batch_size": args.batch_size, "seed": args.seed,
    }
    print(f"  best val MSE = {log['best_val_mse']:.4f} at epoch "
          f"{log['best_epoch']}/{log['n_epochs_run']}; fit took "
          f"{log['fit_seconds']:.1f}s", flush=True)

    model.eval()
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))

    # Sanity: training-set R^2 on the full panel
    with torch.no_grad():
        yhat_full = model(torch.as_tensor(X, dtype=torch.float32)).numpy().ravel()
    r2_full = float(r2_score(y, yhat_full))
    print(f"  full-set R^2 (in-sample): {r2_full:.4f}", flush=True)
    (out_dir / "in_sample_r2.json").write_text(
        json.dumps({"in_sample_r2": r2_full}, indent=2))


if __name__ == "__main__":
    main()

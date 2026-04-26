#!/usr/bin/env python3
"""Affine-$\\Phi$ KAN ablation.

Spec: docs/kan_affine_phi_ablation_spec.md (sections 4 = Option A, 5 = output
schema, 6 = verdict criteria).

Tests empirically: pairwise Shapley $I_{p,p'}$ on a 2-layer KAN
vanishes for all $p \\ne p'$ iff every outer activation $\\Phi_q$ is affine
on the relevant range.

Implementation (Option A — KANLayer inner + nn.Linear outer):

    f(x) = W . phi(x) + b
        phi : R^P -> R^H,  phi_h(x) = sum_p phi_{h,p}(x_p)  (KANLayer, B-spline edges)
        W   : R^{1xH},  b : R                              (torch.nn.Linear, affine by construction)

Each marker x_p enters f only through phi_{h,p}(x_p), then a single linear
combination over h. Pairwise Shapley I_{p,p'} should collapse to the noise
floor (RF-level, ~1e-15 to 1e-17) on the published GT pairs from Bloom 2015
SuppData4.

Usage::

    python3 yeast_bloom_kan_affine_ablation.py Neomycin
    python3 yeast_bloom_kan_affine_ablation.py Lactate

Per-phenotype outputs land at::

    outputs/yeast_bloom_kan/affine_phi_ablation/<phenotype>/
        cv_metrics.json
        train_log.json
        in_sample_r2.json
        model.pt
        gt_pair_iValues.csv     <- comparison vs standard KAN + RF
        verdict.json            <- PASS/FAIL/MIXED per spec section 6

Verdict (per spec section 6):
    PASS:  median(|I|_affine on GT) <= 1e-10  AND  median(|I|_std / |I|_aff) >= 1e8
    FAIL:  median(|I|_affine on GT) >= 1e-3
    MIXED: anything else
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
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# Reuse infrastructure from the baseline standard-KAN trainer for parity.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from yeast_bloom_kan_train import load_phenotype, train_kan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_cache"
OUT = ROOT / "outputs" / "yeast_bloom_kan" / "affine_phi_ablation"
STD_OUT = ROOT / "outputs" / "yeast_bloom_kan"

# Pin pykan checkpoint dir under outputs/ to avoid cwd pollution
# (KANLayer auto-creates ./model/ on first instantiation in some versions).
CKPT_BASE = OUT / "_pykan_ckpt"
CKPT_BASE.mkdir(parents=True, exist_ok=True)

# pykan namespace can shift between versions; try both import paths.
try:
    from kan import KANLayer  # type: ignore
except ImportError:
    from kan.KANLayer import KANLayer  # type: ignore


# ----------------------------------------------------------------------------
# Model: KAN with outer activation forced affine.
# ----------------------------------------------------------------------------

class AffinePhiKAN(nn.Module):
    """KAN with $\\Phi_q$ forced affine via torch.nn.Linear outer layer.

    Inner edges remain learnable B-splines so per-feature nonlinearity is
    preserved; only the cross-feature interaction channel (curvature of
    $\\Phi_q$) is removed. The Shapley pairwise interaction functional
    $I_{p,p'}$ is bounded by $\\Phi_q$ curvature,
    so on this model $|I_{p,p'}|$ should collapse to numerical noise.
    """

    def __init__(self, n_features: int, hidden: int = 8, grid: int = 5,
                 k: int = 3, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        # pykan 0.2.8 KANLayer.__init__ does not accept `seed`; the
        # torch.manual_seed(seed) above seeds the RNG used for KANLayer
        # parameter init, matching baseline determinism.
        self.inner = KANLayer(
            in_dim=n_features, out_dim=hidden, num=grid, k=k,
        )
        self.outer = nn.Linear(hidden, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        # Some pykan versions return a tuple; first element is the activation.
        z = out[0] if isinstance(out, tuple) else out
        return self.outer(z)


def make_affine_kan(p_in: int, hidden: int, grid: int, k: int,
                    seed: int) -> AffinePhiKAN:
    return AffinePhiKAN(p_in, hidden=hidden, grid=grid, k=k, seed=seed)


# ----------------------------------------------------------------------------
# Cross-validation parallel to cv_r2 in yeast_bloom_kan_train.py.
# ----------------------------------------------------------------------------

def cv_r2_affine(
    X: np.ndarray, y: np.ndarray, *,
    hidden: int, grid: int, k: int, seed: int,
    epochs: int, patience: int, lr: float, batch_size: int,
) -> dict:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2s: list[float] = []
    for fold, (tr, te) in enumerate(kf.split(X)):
        Xtr_full, Xte = X[tr], X[te]
        ytr_full, yte = y[tr], y[te]
        n_tr = len(Xtr_full)
        rng = np.random.default_rng(seed + fold)
        idx_perm = rng.permutation(n_tr)
        n_val = max(1, n_tr // 10)
        val_idx = idx_perm[:n_val]
        tr_idx = idx_perm[n_val:]
        Xtr, Xv = Xtr_full[tr_idx], Xtr_full[val_idx]
        ytr, yv = ytr_full[tr_idx], ytr_full[val_idx]

        model = make_affine_kan(X.shape[1], hidden, grid, k, seed + fold)
        train_kan(model, Xtr, ytr, Xv, yv,
                  epochs=epochs, patience=patience, lr=lr,
                  batch_size=batch_size, seed=seed + fold)
        model.eval()
        with torch.no_grad():
            yhat = model(torch.as_tensor(Xte, dtype=torch.float32)).numpy().ravel()
        r2s.append(float(r2_score(yte, yhat)))
        print(f"  fold {fold}: R^2={r2s[-1]:.4f}", flush=True)
    return {
        "fold_r2": r2s,
        "mean_r2": float(np.mean(r2s)),
        "sd_r2": float(np.std(r2s, ddof=0)),
    }


# ----------------------------------------------------------------------------
# Pairwise Shapley interaction estimator (sequential — only ~12 GT pairs).
# Inlined from yeast_bloom_kan_interaction.py::_interaction_for_pair.
# ----------------------------------------------------------------------------

def shapley_interaction_one_pair(
    model: nn.Module,
    bg: np.ndarray, eval_arr: np.ndarray,
    j_idx: int, k_idx: int,
    candidate_indices: np.ndarray,
    n_samples: int = 30,
    base_seed: int = 42,
) -> np.ndarray:
    """Return per-eval-point pairwise Shapley I_{j,k} (length n_eval array).

    Mirrors yeast_bloom_kan_interaction.py's inner loop. No joblib because
    we run only ~12 GT pairs per phenotype; sequential is plenty fast.
    """
    n_bg = bg.shape[0]
    n_eval = eval_arr.shape[0]
    rest_mask = (candidate_indices != j_idx) & (candidate_indices != k_idx)
    rest_indices = candidate_indices[rest_mask]
    m_minus_2 = len(rest_indices)
    block = 4 * n_bg
    p = bg.shape[1]

    per_eval_I = np.empty(n_eval, dtype=np.float64)
    X_batch = np.empty((n_samples * block, p), dtype=bg.dtype)

    for e_i in range(n_eval):
        rng = np.random.default_rng(base_seed + e_i)
        eval_x = eval_arr[e_i]
        eval_j = eval_x[j_idx]
        eval_k = eval_x[k_idx]
        for s_i in range(n_samples):
            if m_minus_2 > 0:
                perm = rng.permutation(rest_indices)
                s_size = int(rng.integers(0, m_minus_2 + 1))
                S = perm[:s_size]
            else:
                S = np.empty(0, dtype=np.int64)
            base = s_i * block
            X_batch[base:base + n_bg] = bg
            if S.size:
                X_batch[base:base + n_bg, S] = eval_x[S]
            X_batch[base + n_bg:base + 2 * n_bg] = X_batch[base:base + n_bg]
            X_batch[base + n_bg:base + 2 * n_bg, j_idx] = eval_j
            X_batch[base + 2 * n_bg:base + 3 * n_bg] = X_batch[base:base + n_bg]
            X_batch[base + 2 * n_bg:base + 3 * n_bg, k_idx] = eval_k
            X_batch[base + 3 * n_bg:base + 4 * n_bg] = X_batch[base + n_bg:base + 2 * n_bg]
            X_batch[base + 3 * n_bg:base + 4 * n_bg, k_idx] = eval_k

        with torch.no_grad():
            preds = model(torch.as_tensor(X_batch, dtype=torch.float32)).cpu().numpy().ravel()
        preds = preds.reshape(n_samples, 4, n_bg)
        v = preds.mean(axis=2)            # shape (n_samples, 4)
        delta = v[:, 3] - v[:, 1] - v[:, 2] + v[:, 0]
        per_eval_I[e_i] = delta.mean()

    return per_eval_I


def compute_gt_pair_diagnostic(
    phenotype: str, model: nn.Module, *, n_samples: int = 30,
) -> pd.DataFrame:
    """Compute |I_{j,k}| on each in-panel published GT pair.

    Reads:
      data_cache/<phen>/X_filtered.npy
      outputs/yeast_bloom_kan/<phen>/panel_meta.json   (eval/bg, panel cols)
      outputs/yeast_bloom_kan/<phen>/gt_pair_iValues.csv (standard-KAN + RF baselines)

    Writes one row per GT pair with three columns:
      kan_standard_mean_abs_I  (from existing baseline)
      kan_affine_mean_abs_I    (this run)
      rf_mean_abs_I            (from existing baseline)
      kan_standard_over_affine_ratio  (diagnostic)
    """
    panel_meta_path = STD_OUT / phenotype / "panel_meta.json"
    if not panel_meta_path.exists():
        raise FileNotFoundError(
            f"{panel_meta_path} missing. Run yeast_bloom_kan_train.py first "
            "to produce the standard-KAN baseline panel_meta and gt_pair_iValues."
        )
    panel_meta = json.loads(panel_meta_path.read_text())
    p_in = int(panel_meta["p_panel"])
    bg_idx = np.asarray(panel_meta["bg_idx"], dtype=np.int64)
    eval_idx = np.asarray(panel_meta["eval_idx"], dtype=np.int64)
    panel_cols_in_full = np.asarray(panel_meta["panel_cols_in_full_dim"], dtype=np.int64)

    X_full = np.load(DATA / phenotype / "X_filtered.npy")
    X_panel = X_full[:, panel_cols_in_full].astype(np.float32)
    X_bg = X_panel[bg_idx]
    X_eval = X_panel[eval_idx]
    cand_idx = np.arange(p_in, dtype=np.int64)

    std_gt = pd.read_csv(STD_OUT / phenotype / "gt_pair_iValues.csv")

    rows: list[dict] = []
    model.eval()
    print(f"[affine-Phi] computing |I| on {len(std_gt)} GT pairs "
          f"(n_bg={X_bg.shape[0]}, n_eval={X_eval.shape[0]}, "
          f"n_samples={n_samples})...", flush=True)
    for ridx, row in std_gt.iterrows():
        j_local = int(row["panel_j_local"])
        k_local = int(row["panel_k_local"])
        per_eval = shapley_interaction_one_pair(
            model, X_bg, X_eval, j_local, k_local, cand_idx,
            n_samples=n_samples, base_seed=42 + ridx * 1009,
        )
        I_mean = float(per_eval.mean())
        I_abs_mean = float(np.mean(np.abs(per_eval)))
        out_row = {
            c: row[c] for c in row.index if c not in (
                "kan_mean_abs_I", "kan_mean_I", "rf_mean_abs_I", "rf_mean_I",
            )
        }
        std_abs = float(row["kan_mean_abs_I"])
        out_row["kan_standard_mean_abs_I"] = std_abs
        out_row["kan_standard_mean_I"] = float(row["kan_mean_I"])
        out_row["kan_affine_mean_abs_I"] = I_abs_mean
        out_row["kan_affine_mean_I"] = I_mean
        out_row["rf_mean_abs_I"] = float(row["rf_mean_abs_I"])
        out_row["rf_mean_I"] = float(row["rf_mean_I"])
        out_row["kan_standard_over_affine_ratio"] = std_abs / max(I_abs_mean, 1e-30)
        rows.append(out_row)
        print(f"  pair {ridx + 1}/{len(std_gt)}: "
              f"|I|_aff={I_abs_mean:.3e}  |I|_std={std_abs:.3e}  "
              f"ratio={std_abs / max(I_abs_mean, 1e-30):.2e}", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Verdict (per spec section 6).
# ----------------------------------------------------------------------------

def compute_verdict(df: pd.DataFrame) -> dict:
    aff_med = float(df["kan_affine_mean_abs_I"].median())
    ratio_med = float(df["kan_standard_over_affine_ratio"].median())
    if aff_med <= 1e-10 and ratio_med >= 1e8:
        verdict = "PASS"
    elif aff_med >= 1e-3:
        verdict = "FAIL"
    else:
        verdict = "MIXED"
    return {
        "n_gt_pairs": int(len(df)),
        "affine_median_abs_I": aff_med,
        "standard_over_affine_ratio_median": ratio_med,
        "verdict": verdict,
    }


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("phenotype")
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--grid", type=int, default=5)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=30,
                   help="MC samples per Shapley evaluation (matches baseline)")
    p.add_argument("--skip-cv", action="store_true",
                   help="Skip 5-fold CV and only do final fit + Shapley")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training; load model.pt and only compute Shapley")
    args = p.parse_args()

    ph = args.phenotype
    out_dir = OUT / ph
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Affine-Phi KAN ablation: {ph} ===", flush=True)
    data = load_phenotype(ph)
    X = data["X_panel"]
    y = data["y"]
    n, p_in = X.shape
    print(f"  X_panel={X.shape}  y={y.shape}  panel_cols={p_in}", flush=True)

    config = {
        "hidden": args.hidden, "grid": args.grid, "k": args.k,
        "epochs": args.epochs, "patience": args.patience,
        "lr": args.lr, "batch_size": args.batch_size, "seed": args.seed,
        "model": "AffinePhiKAN (Option A: KANLayer inner + nn.Linear outer)",
    }

    if args.skip_train:
        print("=== Loading saved model.pt ===", flush=True)
        model = make_affine_kan(p_in, args.hidden, args.grid, args.k, args.seed)
        sd = torch.load(out_dir / "model.pt", map_location="cpu",
                        weights_only=False)
        model.load_state_dict(sd)
    else:
        if not args.skip_cv:
            print("=== 5-fold CV ===", flush=True)
            t0 = time.time()
            cv = cv_r2_affine(X, y,
                              hidden=args.hidden, grid=args.grid, k=args.k,
                              seed=args.seed, epochs=args.epochs,
                              patience=args.patience, lr=args.lr,
                              batch_size=args.batch_size)
            cv["cv_seconds"] = time.time() - t0
            cv["config"] = config
            (out_dir / "cv_metrics.json").write_text(json.dumps(cv, indent=2))
            print(f"  CV R^2: mean={cv['mean_r2']:.4f}  sd={cv['sd_r2']:.4f}",
                  flush=True)

        print("=== Final fit ===", flush=True)
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(n)
        n_val = max(1, n // 10)
        Xv = X[perm[:n_val]]; yv = y[perm[:n_val]]
        Xtr = X[perm[n_val:]]; ytr = y[perm[n_val:]]
        model = make_affine_kan(p_in, args.hidden, args.grid, args.k, args.seed)
        t0 = time.time()
        log = train_kan(model, Xtr, ytr, Xv, yv,
                        epochs=args.epochs, patience=args.patience,
                        lr=args.lr, batch_size=args.batch_size, seed=args.seed)
        log["fit_seconds"] = time.time() - t0
        log["config"] = config
        print(f"  best val MSE = {log['best_val_mse']:.4f} at epoch "
              f"{log['best_epoch']}/{log['n_epochs_run']}; "
              f"fit took {log['fit_seconds']:.1f}s", flush=True)

        model.eval()
        torch.save(model.state_dict(), out_dir / "model.pt")
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))

        with torch.no_grad():
            yhat = model(torch.as_tensor(X, dtype=torch.float32)).numpy().ravel()
        r2_full = float(r2_score(y, yhat))
        (out_dir / "in_sample_r2.json").write_text(
            json.dumps({"in_sample_r2": r2_full}, indent=2))
        print(f"  in-sample R^2 = {r2_full:.4f}", flush=True)

    print("=== GT-pair |I| diagnostic ===", flush=True)
    t0 = time.time()
    gt_df = compute_gt_pair_diagnostic(ph, model, n_samples=args.n_samples)
    print(f"  diagnostic done in {time.time() - t0:.1f}s", flush=True)
    gt_df.to_csv(out_dir / "gt_pair_iValues.csv", index=False)

    verdict = compute_verdict(gt_df)
    verdict["phenotype"] = ph
    verdict["config"] = config
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))

    print(f"\n=== VERDICT: {ph} ===")
    print(f"  affine median |I|:                {verdict['affine_median_abs_I']:.3e}")
    print(f"  standard/affine ratio median:     {verdict['standard_over_affine_ratio_median']:.3e}")
    print(f"  verdict:                          {verdict['verdict']}")
    print(f"  (PASS: |I|<=1e-10 AND ratio>=1e8;  FAIL: |I|>=1e-3;  else MIXED)", flush=True)


if __name__ == "__main__":
    main()

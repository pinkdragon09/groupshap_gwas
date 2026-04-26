"""Pairwise Shapley-interaction sweep on a fitted KAN, batched across MC.

Mirrors the RF baseline's ``yeast_bloom_fast_interaction.py`` pattern
for an apples-to-apples comparison.

Differences vs the RF version:
  - Uses a torch KAN forward pass instead of sklearn predict.
  - Pins ``torch.set_num_threads(1)`` inside each worker to keep the
    16-core loky outer parallelism clean.
  - Uses spawn-friendly model-state passing: each worker reconstructs
    a KAN with the same hyperparameters, then loads the saved
    state_dict from ``model.pt``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "yeast_bloom_kan"
DATA = ROOT / "data_cache"

CKPT_BASE = OUT / "_pykan_ckpt"
CKPT_BASE.mkdir(parents=True, exist_ok=True)


def _build_kan(p_in: int, hidden: int, grid: int, k: int, seed: int, ckpt_dir: Path):
    from kan import KAN
    return KAN(
        width=[p_in, hidden, 1], grid=grid, k=k, seed=seed,
        ckpt_path=str(ckpt_dir), auto_save=False,
    )


def _interaction_for_pair(
    model_pkl_path: str,
    p_in: int,
    hidden: int,
    grid: int,
    k: int,
    seed: int,
    ckpt_dir: str,
    bg: np.ndarray,
    eval_arr: np.ndarray,
    j_idx: int,
    k_idx: int,
    pair_label_j: str,
    pair_label_k: str,
    candidate_indices: np.ndarray,
    n_samples: int,
    base_seed: int,
) -> dict:
    """Single-pair worker. Reconstructs KAN inside the worker."""
    # Stay deterministic and avoid CPU oversubscription inside each loky worker.
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = _build_kan(p_in, hidden, grid, k, seed, Path(ckpt_dir))
    sd = torch.load(model_pkl_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model.eval()

    n_bg, p = bg.shape
    n_eval = eval_arr.shape[0]
    rest_mask = (candidate_indices != j_idx) & (candidate_indices != k_idx)
    rest_indices = candidate_indices[rest_mask]
    m_minus_2 = len(rest_indices)
    block = 4 * n_bg

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
        v = preds.mean(axis=2)  # (n_samples, 4)
        delta = v[:, 3] - v[:, 1] - v[:, 2] + v[:, 0]
        per_eval_I[e_i] = delta.mean()

    return {
        "marker_j": pair_label_j,
        "marker_k": pair_label_k,
        "j_idx": int(j_idx),
        "k_idx": int(k_idx),
        "mean_interaction": float(per_eval_I.mean()),
        "mean_abs_interaction": float(np.mean(np.abs(per_eval_I))),
        "sd_interaction": float(per_eval_I.std(ddof=0)),
    }


def pairwise_interaction_kan_fast(
    model_state_path: Path,
    *, p_in: int, hidden: int, grid: int, k: int, seed: int, ckpt_dir: Path,
    X_bg: np.ndarray,
    X_eval: np.ndarray,
    candidate_indices: Sequence[int],
    candidate_labels: Sequence[str],
    n_samples: int = 30,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
) -> pd.DataFrame:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    bg = np.ascontiguousarray(X_bg, dtype=np.float32)
    ev = np.ascontiguousarray(X_eval, dtype=np.float32)
    m = len(candidate_indices)
    # Use (a, b) here -- not (j, k) -- so the comprehension does NOT shadow
    # the outer ``k`` (B-spline order) parameter that gets closed over.
    pairs = [(a, b) for a in range(m) for b in range(a + 1, m)]
    n_eval = ev.shape[0]
    if verbose:
        print(f"[KAN pairwise] {m} markers -> {len(pairs)} pairs, "
              f"n_bg={bg.shape[0]}, n_eval={n_eval}, mc={n_samples}", flush=True)

    def pair_seed(pi: int) -> int:
        return random_state + (pi + 1) * (n_eval + 1) * 1009

    # Cap outer workers — KAN forward on a 1800x97 batch is fast enough
    # that 8 workers comfortably saturate; more risks RAM blowup since
    # each worker materializes its own KAN graph + state_dict.
    if n_jobs == -1:
        n_jobs = min(8, os.cpu_count() or 1)

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_interaction_for_pair)(
            str(model_state_path), p_in, hidden, grid, k, seed, str(ckpt_dir),
            bg, ev,
            int(candidate_indices[a]), int(candidate_indices[b]),
            str(candidate_labels[a]), str(candidate_labels[b]),
            candidate_indices, n_samples, pair_seed(pi),
        )
        for pi, (a, b) in enumerate(pairs)
    )
    if verbose:
        print(f"[KAN pairwise] done in {time.time() - t0:.1f}s "
              f"({(time.time() - t0) / max(len(pairs), 1) * 1000:.1f} ms/pair)", flush=True)
    df = pd.DataFrame(results)
    return df.sort_values("mean_abs_interaction", ascending=False).reset_index(drop=True)


def run_phenotype(
    phenotype: str,
    *, hidden: int, grid: int, k: int, seed: int,
    n_samples: int = 30, n_jobs: int = -1,
) -> dict:
    out_dir = OUT / phenotype
    if not (out_dir / "model.pt").exists():
        raise FileNotFoundError(
            f"{out_dir / 'model.pt'} missing. Run yeast_bloom_kan_train.py first."
        )
    panel_meta = json.loads((out_dir / "panel_meta.json").read_text())
    p_in = int(panel_meta["p_panel"])
    bg_idx = np.asarray(panel_meta["bg_idx"], dtype=np.int64)
    eval_idx = np.asarray(panel_meta["eval_idx"], dtype=np.int64)
    panel_cols_in_full = np.asarray(
        panel_meta["panel_cols_in_full_dim"], dtype=np.int64)
    marker_ids = panel_meta["marker_ids"]

    X_full = np.load(DATA / phenotype / "X_filtered.npy")
    X_panel = X_full[:, panel_cols_in_full].astype(np.float32)
    X_bg = X_panel[bg_idx]
    X_eval = X_panel[eval_idx]

    # Candidate indices in the *panel* (0..p_panel-1).
    cand_idx = np.arange(p_in, dtype=np.int64)
    ckpt_dir = CKPT_BASE / phenotype
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {phenotype}: KAN pairwise interaction sweep ===", flush=True)
    print(f"  panel cols={p_in}, pairs={p_in * (p_in - 1) // 2}", flush=True)
    print(f"  bg={X_bg.shape[0]}, eval={X_eval.shape[0]}, mc={n_samples}", flush=True)

    inter_df = pairwise_interaction_kan_fast(
        out_dir / "model.pt",
        p_in=p_in, hidden=hidden, grid=grid, k=k, seed=seed,
        ckpt_dir=ckpt_dir,
        X_bg=X_bg, X_eval=X_eval,
        candidate_indices=cand_idx,
        candidate_labels=marker_ids,
        n_samples=n_samples,
        random_state=seed,
        n_jobs=n_jobs,
    )
    inter_df.to_csv(out_dir / "pairwise_interactions.csv", index=False)

    print(f"  saved -> {out_dir / 'pairwise_interactions.csv'} "
          f"({len(inter_df)} rows)", flush=True)
    print("  top 10 by |I_jk|:")
    print(inter_df.head(10)[
        ["marker_j", "marker_k", "mean_abs_interaction", "mean_interaction"]
    ].to_string(index=False))
    return {
        "phenotype": phenotype,
        "n_pairs": int(len(inter_df)),
        "max_abs_I": float(inter_df["mean_abs_interaction"].max()),
        "median_abs_I": float(inter_df["mean_abs_interaction"].median()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phenotypes", nargs="+")
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--grid", type=int, default=5)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mc-samples", type=int, default=30)
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()

    summaries = []
    for ph in args.phenotypes:
        s = run_phenotype(
            ph,
            hidden=args.hidden, grid=args.grid, k=args.k, seed=args.seed,
            n_samples=args.mc_samples, n_jobs=args.n_jobs,
        )
        summaries.append(s)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

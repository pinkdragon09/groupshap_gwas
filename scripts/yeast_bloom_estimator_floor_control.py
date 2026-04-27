#!/usr/bin/env python3
"""Estimator-floor control for the affine-Phi KAN interaction channel.

Computes pairwise Shapley |I_{j,k}| on a hand-coded purely additive oracle:

    f_add(x) = sum_p sin(x_p)

By the additivity axiom, I_{j,k}(f_add) = 0 exactly for all j != k.
Any nonzero |I| measured by the MC estimator on f_add is pure estimator noise.

Compares this noise floor to the affine-Phi KAN residuals from the ablation run to
determine whether the BOTH-MIXED verdict is a float32 artifact (FLOOR)
or genuine partial-mechanism signal (ABOVE_FLOOR).

Verdict criterion (from A-2 spec):
    FLOOR:       median floor_ratio < 5 on BOTH phenotypes
    ABOVE_FLOOR: median floor_ratio >= 5 on EITHER phenotype
    INCONCLUSIVE: edge case

Outputs:
    outputs/yeast_bloom_kan/estimator_floor/floor_results.csv
    outputs/yeast_bloom_kan/estimator_floor/summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from yeast_bloom_kan_train import load_phenotype  # noqa: E402

AFFINE_OUT = ROOT / "outputs" / "yeast_bloom_kan" / "affine_phi_ablation"
STD_OUT    = ROOT / "outputs" / "yeast_bloom_kan"
FLOOR_OUT  = ROOT / "outputs" / "yeast_bloom_kan" / "estimator_floor"

N_SAMPLES = 30
N_BG      = 15
N_EVAL    = 15
SEED_BASE = 42


# ---------------------------------------------------------------------------
# Oracle function — purely additive, I_{j,k} = 0 exactly for all j != k.
# ---------------------------------------------------------------------------

def f_add(X: np.ndarray) -> np.ndarray:
    return np.sin(X).sum(axis=1)


# ---------------------------------------------------------------------------
# Pairwise Shapley MC estimator (mirrors affine_ablation + interaction scripts).
# ---------------------------------------------------------------------------

def shapley_interaction_oracle(
    bg: np.ndarray,
    eval_arr: np.ndarray,
    j_idx: int,
    k_idx: int,
    candidate_indices: np.ndarray,
    n_samples: int = N_SAMPLES,
    base_seed: int = SEED_BASE,
) -> np.ndarray:
    """Return per-eval-point pairwise Shapley I_{j,k} for f_add."""
    n_bg  = bg.shape[0]
    n_eval = eval_arr.shape[0]
    p = bg.shape[1]
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

        preds = f_add(X_batch.astype(np.float32))
        preds = preds.reshape(n_samples, 4, n_bg)
        v = preds.mean(axis=2)
        delta = v[:, 3] - v[:, 1] - v[:, 2] + v[:, 0]
        per_eval_I[e_i] = delta.mean()

    return per_eval_I


# ---------------------------------------------------------------------------
# Per-phenotype computation.
# ---------------------------------------------------------------------------

def run_phenotype(phenotype: str) -> pd.DataFrame:
    data = load_phenotype(phenotype)
    panel_meta_path = STD_OUT / phenotype / "panel_meta.json"
    panel_meta = json.loads(panel_meta_path.read_text())
    p_in = int(panel_meta["p_panel"])
    bg_idx   = np.asarray(panel_meta["bg_idx"],   dtype=np.int64)
    eval_idx = np.asarray(panel_meta["eval_idx"], dtype=np.int64)
    panel_cols = np.asarray(panel_meta["panel_cols_in_full_dim"], dtype=np.int64)

    from pathlib import Path as _P
    X_full = np.load(ROOT / "data_cache" / phenotype / "X_filtered.npy")
    X_panel = X_full[:, panel_cols].astype(np.float32)
    X_bg   = X_panel[bg_idx]
    X_eval = X_panel[eval_idx]
    cand_idx = np.arange(p_in, dtype=np.int64)

    affine_df = pd.read_csv(AFFINE_OUT / phenotype / "gt_pair_iValues.csv")

    rows: list[dict] = []
    print(f"[floor-ctrl] {phenotype}: {len(affine_df)} GT pairs  "
          f"n_bg={X_bg.shape[0]}  n_eval={X_eval.shape[0]}", flush=True)
    for ridx, row in affine_df.iterrows():
        j_loc = int(row["panel_j_local"])
        k_loc = int(row["panel_k_local"])
        per_eval = shapley_interaction_oracle(
            X_bg, X_eval, j_loc, k_loc, cand_idx,
            n_samples=N_SAMPLES,
            base_seed=SEED_BASE + ridx * 1009,
        )
        I_oracle = float(np.mean(np.abs(per_eval)))
        I_affine  = float(row["kan_affine_mean_abs_I"])
        I_std     = float(row["kan_standard_mean_abs_I"])
        floor_ratio = I_affine / max(I_oracle, 1e-300)
        rows.append({
            "phenotype":      phenotype,
            "j_loc":          j_loc,
            "k_loc":          k_loc,
            "I_oracle_abs":   I_oracle,
            "I_affine_abs":   I_affine,
            "I_standard_abs": I_std,
            "floor_ratio":    floor_ratio,
        })
        print(f"  pair {ridx + 1}/{len(affine_df)}: "
              f"|I|_oracle={I_oracle:.3e}  |I|_affine={I_affine:.3e}  "
              f"floor_ratio={floor_ratio:.2f}", flush=True)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    FLOOR_OUT.mkdir(parents=True, exist_ok=True)

    all_rows = []
    phenotype_summaries = {}

    for ph in ("Neomycin", "Lactate"):
        print(f"\n=== {ph} ===", flush=True)
        df = run_phenotype(ph)
        all_rows.append(df)

        med_oracle = float(df["I_oracle_abs"].median())
        med_affine = float(df["I_affine_abs"].median())
        med_ratio  = float(df["floor_ratio"].median())
        verdict_ph = "FLOOR" if med_ratio < 5 else "ABOVE_FLOOR"
        phenotype_summaries[ph] = {
            "median_I_oracle":    med_oracle,
            "median_I_affine":    med_affine,
            "median_floor_ratio": med_ratio,
            "verdict":            verdict_ph,
        }
        print(f"  median |I|_oracle = {med_oracle:.3e}", flush=True)
        print(f"  median |I|_affine = {med_affine:.3e}", flush=True)
        print(f"  median floor_ratio = {med_ratio:.2f}", flush=True)
        print(f"  verdict: {verdict_ph}", flush=True)

    results_df = pd.concat(all_rows, ignore_index=True)
    results_df.to_csv(FLOOR_OUT / "floor_results.csv", index=False)
    print(f"\nWrote {FLOOR_OUT}/floor_results.csv", flush=True)

    # Combined verdict
    neo_v = phenotype_summaries["Neomycin"]["verdict"]
    lac_v = phenotype_summaries["Lactate"]["verdict"]
    if neo_v == "FLOOR" and lac_v == "FLOOR":
        combined = "FLOOR"
    elif neo_v == "ABOVE_FLOOR" or lac_v == "ABOVE_FLOOR":
        combined = "ABOVE_FLOOR"
    else:
        combined = "INCONCLUSIVE"

    summary = {**phenotype_summaries, "combined_verdict": combined}
    (FLOOR_OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {FLOOR_OUT}/summary.json", flush=True)

    print(f"\n=== COMBINED VERDICT: {combined} ===")
    if combined == "FLOOR":
        print("  Affine-Phi residuals are at the MC estimator noise floor.")
        print("  Noise floor fully explains the residual; BOTH-MIXED label was a float32 artifact.")
    elif combined == "ABOVE_FLOOR":
        print("  Affine-Phi residuals exceed the estimator floor.")
        print("  Noise floor only partially explains the residual; some residual mechanism active.")
    else:
        print("  Edge case.")


if __name__ == "__main__":
    main()

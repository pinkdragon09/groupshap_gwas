#!/usr/bin/env python3
"""
Arabidopsis-ready transformer / Mamba runner for the downloaded local data.

Default inputs:
- AraPheno bulk database zip
- 1001 Genomes imputed SNP HDF5 matrix
- 1001 Genomes kinship HDF5 for principal components
- baseline top-hit table from the local PC-adjusted linear GWAS scan

This script consumes the current local data layout directly and trains a
sequence-style nonlinear model on a prefiltered SNP set plus kinship PCs.

Supported models:
- cnn
- transformer
- mamba

Use --debug for a smaller smoke-test configuration.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

try:
    import shap
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: shap") from exc

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: torch") from exc

from grouped_shap_utils import (
    build_group_cols_from_meta,
    grouped_shapley_all,
    summarize_grouped_shapley,
)


PHENO_ZIP = Path("data/arapheno/bulk/database.zip")
GENO_H5 = Path("data/1001genomes/genotypes/1001_SNP_MATRIX/imputed_snps_binary.hdf5")
KINSHIP_H5 = Path("data/1001genomes/genotypes/1001_SNP_MATRIX/kinship_ibs_mac5.hdf5")
BASELINE_HITS = Path("outputs/full_baseline_gwas/study27_mean_trl_top_hits.tsv")
DEFAULT_OUTDIR = Path("outputs/sequence_gwas")

DEFAULT_STUDY_VALUES_FILE = "27/study_27_values.csv"
DEFAULT_PHENOTYPE_COL = "Mean(TRL) C"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pheno-zip", default=str(PHENO_ZIP))
    parser.add_argument("--geno-h5", default=str(GENO_H5))
    parser.add_argument("--kinship-h5", default=str(KINSHIP_H5))
    parser.add_argument("--baseline-hits", default=str(BASELINE_HITS))
    parser.add_argument("--study-values-file", default=DEFAULT_STUDY_VALUES_FILE)
    parser.add_argument("--phenotype-col", default=DEFAULT_PHENOTYPE_COL)
    parser.add_argument("--model", choices=["cnn", "transformer", "mamba"], default="transformer")
    parser.add_argument("--top-snps", type=int, default=500)
    parser.add_argument("--n-pcs", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--debug", action="store_true")
    # Grouped Shapley options
    parser.add_argument(
        "--group-by",
        default="chromosome",
        choices=["chromosome", "none"],
        help=(
            "Grouping scheme for cooperative-game grouped Shapley. "
            "'chromosome' groups SNPs by chromosome (Arabidopsis: 5 groups). "
            "'none' skips grouped Shapley and outputs per-SNP SHAP only."
        ),
    )
    parser.add_argument(
        "--n-background-grouped",
        type=int,
        default=100,
        help="Background samples used to marginalise the grouped Shapley value function.",
    )
    parser.add_argument(
        "--n-eval-grouped",
        type=int,
        default=50,
        help="Evaluation points per fold for grouped Shapley (subset of test set).",
    )
    return parser.parse_args()


def apply_debug_overrides(args: argparse.Namespace) -> None:
    if not args.debug:
        return
    args.top_snps = min(args.top_snps, 100)
    args.folds = min(args.folds, 3)
    args.epochs = min(args.epochs, 10)
    args.batch_size = min(args.batch_size, 8)
    args.d_model = min(args.d_model, 32)
    args.n_heads = min(args.n_heads, 2)
    args.n_layers = min(args.n_layers, 1)
    args.n_background_grouped = min(args.n_background_grouped, 30)
    args.n_eval_grouped = min(args.n_eval_grouped, 10)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def load_top_snp_indices(baseline_hits: Path, top_snps: int) -> list[int]:
    hits = pd.read_csv(baseline_hits, sep="\t")
    hits = hits.sort_values("p_value", ascending=True)
    return hits["snp_index"].astype(int).head(top_snps).tolist()


def compute_pcs(kinship_h5: Path, common_accs: list[str], n_pcs: int) -> np.ndarray:
    with h5py.File(kinship_h5, "r") as f:
        kinship = f["kinship"][:]
        kin_accs = [x.decode() if isinstance(x, bytes) else str(x) for x in f["accessions"][:]]

    kin_index = {acc: i for i, acc in enumerate(kin_accs)}
    ix = np.asarray([kin_index[acc] for acc in common_accs], dtype=int)
    sub_k = kinship[np.ix_(ix, ix)]
    eigvals, eigvecs = np.linalg.eigh(sub_k)
    pcs = eigvecs[:, -n_pcs:]
    pcs = (pcs - pcs.mean(axis=0)) / pcs.std(axis=0)
    return pcs


def chromosome_labels_and_bounds(h5f: h5py.File) -> tuple[list[str], np.ndarray]:
    labels = [f"Chr{x.decode() if isinstance(x, bytes) else x}" for x in h5f["positions"].attrs["chrs"]]
    bounds = h5f["positions"].attrs["chr_regions"]
    return labels, bounds


def chromosome_for_index(idx: int, labels: list[str], bounds: np.ndarray) -> str:
    for chrom_idx, (start, stop) in enumerate(bounds):
        if int(start) <= idx < int(stop):
            return labels[chrom_idx]
    return "Chr?"


def load_arabidopsis_matrix(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, int]:
    y_by_acc = load_phenotype(Path(args.pheno_zip), args.study_values_file, args.phenotype_col)
    snp_indices = load_top_snp_indices(Path(args.baseline_hits), args.top_snps)
    sorted_snp_indices = sorted(snp_indices)

    with h5py.File(args.geno_h5, "r") as f:
        geno_accs = [x.decode() if isinstance(x, bytes) else str(x) for x in f["accessions"][:]]
        common_accs = [acc for acc in geno_accs if acc in y_by_acc]
        geno_ix = np.asarray([geno_accs.index(acc) for acc in common_accs], dtype=int)
        positions = f["positions"][sorted_snp_indices]
        snps = f["snps"][sorted_snp_indices][:, geno_ix].astype(np.float32)
        labels, bounds = chromosome_labels_and_bounds(f)

    pcs = compute_pcs(Path(args.kinship_h5), common_accs, args.n_pcs).astype(np.float32)
    y = pd.Series([y_by_acc[acc] for acc in common_accs], name=args.phenotype_col, dtype=float)

    snp_cols = [f"snp_{idx}" for idx in sorted_snp_indices]
    X_snp = pd.DataFrame(snps.T, columns=snp_cols)
    X_cov = pd.DataFrame(pcs, columns=[f"pc{i}" for i in range(1, args.n_pcs + 1)])
    X = pd.concat([X_snp, X_cov], axis=1)

    snp_meta = pd.DataFrame(
        {
            "feature": snp_cols,
            "snp_index": sorted_snp_indices,
            "chromosome": [chromosome_for_index(idx, labels, bounds) for idx in sorted_snp_indices],
            "position": positions.astype(int),
            "feature_type": "snp",
        }
    )
    covar_meta = pd.DataFrame(
        {
            "feature": X_cov.columns,
            "snp_index": np.nan,
            "chromosome": "",
            "position": np.nan,
            "feature_type": "covariate",
        }
    )
    feature_meta = pd.concat([snp_meta, covar_meta], ignore_index=True)
    return X, y, feature_meta, len(snp_cols)


class TransformerGWASRegressor(nn.Module):
    def __init__(self, num_snps: int, num_covars: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.num_snps = num_snps
        self.num_covars = num_covars
        self.token_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_snps, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model + num_covars),
            nn.Linear(d_model + num_covars, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        snps = x[:, : self.num_snps].unsqueeze(-1)
        covars = x[:, self.num_snps :]
        seq = self.token_proj(snps) + self.pos_embed
        seq = self.encoder(seq)
        pooled = seq.mean(dim=1)
        joined = torch.cat([pooled, covars], dim=1)
        return self.head(joined).squeeze(-1)


class CNNGWASRegressor(nn.Module):
    def __init__(self, num_snps: int, num_covars: int, d_model: int, n_layers: int, dropout: float):
        super().__init__()
        self.num_snps = num_snps
        self.num_covars = num_covars
        hidden = d_model

        layers: list[nn.Module] = []
        in_channels = 1
        for _ in range(n_layers):
            layers.extend(
                [
                    nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = hidden
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden + num_covars, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        snps = x[:, : self.num_snps].unsqueeze(1)
        covars = x[:, self.num_snps :]
        seq = self.conv(snps)
        pooled = self.pool(seq).squeeze(-1)
        joined = torch.cat([pooled, covars], dim=1)
        return self.head(joined).squeeze(-1)


def build_mamba_model(num_snps: int, num_covars: int, d_model: int, n_layers: int, dropout: float) -> nn.Module:
    try:
        from mamba_ssm import Mamba
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: mamba_ssm. Install it before using --model mamba."
        ) from exc

    class MambaGWASRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_snps = num_snps
            self.num_covars = num_covars
            self.token_proj = nn.Linear(1, d_model)
            self.blocks = nn.ModuleList(
                [Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)]
            )
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model + num_covars),
                nn.Linear(d_model + num_covars, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            snps = x[:, : self.num_snps].unsqueeze(-1)
            covars = x[:, self.num_snps :]
            seq = self.token_proj(snps)
            for block in self.blocks:
                seq = block(seq)
            seq = self.norm(seq)
            pooled = seq.mean(dim=1)
            joined = torch.cat([pooled, covars], dim=1)
            return self.head(joined).squeeze(-1)

    return MambaGWASRegressor()


def build_model(args: argparse.Namespace, num_snps: int, num_covars: int) -> nn.Module:
    if args.model == "cnn":
        return CNNGWASRegressor(
            num_snps=num_snps,
            num_covars=num_covars,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
    if args.model == "transformer":
        return TransformerGWASRegressor(
            num_snps=num_snps,
            num_covars=num_covars,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
    if args.model == "mamba":
        return build_mamba_model(
            num_snps=num_snps,
            num_covars=num_covars,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
    raise SystemExit(f"Unsupported model: {args.model}")


def fit_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    num_snps: int,
    num_covars: int,
    seed: int,
) -> tuple[nn.Module, float, float]:
    set_seed(seed)
    device = torch.device(args.device)
    model = build_model(args, num_snps, num_covars).to(device)
    lr = args.learning_rate * 0.5 if args.model == "cnn" else args.learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss() if args.model == "cnn" else nn.MSELoss()

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std < 1e-8:
        y_std = 1.0
    y_train_scaled = (y_train - y_mean) / y_std
    y_val_scaled = (y_val - y_mean) / y_std

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32, device=device)

    best_state = None
    best_val = float("inf")
    stale = 0

    for _epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(X_train_t), device=device)
        for start in range(0, len(order), args.batch_size):
            batch_ix = order[start : start + args.batch_size]
            pred = model(X_train_t[batch_ix])
            loss = loss_fn(pred, y_train_t[batch_ix])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = float(loss_fn(val_pred, y_val_t).item())

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), y_mean, y_std


def compute_shap_values(model: nn.Module, background: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Per-SNP SHAP values via DeepExplainer.

    Note on check_additivity=False: DeepExplainer uses gradient-based hooks
    that may not satisfy exact SHAP additivity for attention/SSM layers (the
    backpropagated contributions can have small numerical discrepancies).
    check_additivity=False suppresses the assertion; the values are still useful
    as relative feature-importance estimates but should NOT be summed to recover
    the model output exactly.

    For theoretically grounded, additivity-exact attributions use the
    cooperative-game grouped Shapley below (SklearnWrapperForGroupedShap +
    grouped_shapley_all), which avoids backpropagation entirely and works for
    any black-box model.
    """
    class ShapWrapper(nn.Module):
        def __init__(self, base_model: nn.Module):
            super().__init__()
            self.base_model = base_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.base_model(x)
            if out.ndim == 1:
                out = out.unsqueeze(-1)
            return out

    model.eval()
    background_t = torch.tensor(background, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    explainer = shap.DeepExplainer(ShapWrapper(model), background_t)
    shap_values = explainer.shap_values(X_test_t, check_additivity=False)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 0]
    return shap_values


class SklearnWrapperForGroupedShap:
    """
    Wraps a trained PyTorch model + scaling parameters into a sklearn-style
    predictor that accepts a pd.DataFrame and returns a 1-D np.ndarray.

    Used by grouped_shapley_all() to call the model as a black box, so that
    the cooperative-game value function can be evaluated without backpropagation.
    The grouped Shapley computation is model-agnostic: it only calls .predict().
    """
    def __init__(
        self,
        model: nn.Module,
        feature_cols: list[str],
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: float,
        y_std: float,
    ):
        self._model = model
        self._feature_cols = feature_cols
        self._x_mean = x_mean
        self._x_std = x_std
        self._y_mean = y_mean
        self._y_std = y_std

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        X_np = X_df[self._feature_cols].to_numpy(dtype=np.float32)
        X_scaled = (X_np - self._x_mean) / self._x_std
        self._model.eval()
        with torch.no_grad():
            pred_scaled = self._model(
                torch.tensor(X_scaled, dtype=torch.float32)
            ).numpy()
        return pred_scaled * self._y_std + self._y_mean


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    feature_meta: pd.DataFrame,
    args: argparse.Namespace,
    num_snps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (metrics_df, per_snp_shap_df, grouped_shap_df).

    Per-SNP SHAP: shap.DeepExplainer — individual SNP players, gradient-based.
      See compute_shap_values() docstring for the check_additivity caveat.
    Grouped Shapley: cooperative-game interventional value function over
      chromosome groups (players = groups, not SNPs).  Model-agnostic: calls
      model.predict() as a black box via SklearnWrapperForGroupedShap.
      This is the theoretically grounded group-additive estimand.
    """
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    metrics = []
    shap_tables = []
    grouped_tables = []

    X_np = X.to_numpy(dtype=np.float32)
    y_np = y.to_numpy(dtype=np.float32)
    num_covars = X.shape[1] - num_snps
    feature_cols = list(X.columns)

    # Build group structure once (same for all folds)
    if args.group_by != "none":
        group_names, group_cols_list = build_group_cols_from_meta(
            feature_meta, group_col=args.group_by
        )

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_np), start=1):
        X_train = X_np[train_idx]
        X_test = X_np[test_idx]
        y_train = y_np[train_idx]
        y_test = y_np[test_idx]

        val_size = max(1, int(round(0.2 * len(X_train))))
        X_val = X_train[:val_size]
        y_val = y_train[:val_size]
        X_fit = X_train[val_size:]
        y_fit = y_train[val_size:]
        if len(X_fit) == 0:
            X_fit = X_train
            y_fit = y_train
            X_val = X_train
            y_val = y_train

        x_mean = X_fit.mean(axis=0)
        x_std = X_fit.std(axis=0)
        x_std[x_std < 1e-8] = 1.0
        X_fit_scaled = (X_fit - x_mean) / x_std
        X_val_scaled = (X_val - x_mean) / x_std
        X_test_scaled = (X_test - x_mean) / x_std

        model, y_mean, y_std = fit_one_fold(
            X_fit_scaled,
            y_fit,
            X_val_scaled,
            y_val,
            args,
            num_snps=num_snps,
            num_covars=num_covars,
            seed=args.random_state + fold_idx,
        )

        model.eval()
        with torch.no_grad():
            pred_scaled = model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
        pred = pred_scaled * y_std + y_mean
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        metrics.append(
            {
                "fold": fold_idx,
                "r2": r2_score(y_test, pred),
                "rmse": rmse,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )

        # ── Per-SNP SHAP (individual player, DeepExplainer) ──────────────────
        # Treats each SNP as a separate player.  See check_additivity caveat in
        # compute_shap_values().  Not the grouped-Shapley estimand.
        background = X_fit_scaled[: min(32, len(X_fit_scaled))]
        shap_values = compute_shap_values(model, background, X_test_scaled)
        fold_importance = feature_meta.copy()
        fold_importance["mean_abs_shap"] = np.abs(shap_values).mean(axis=0)
        fold_importance["fold"] = fold_idx
        shap_tables.append(fold_importance)

        # ── Cooperative-game grouped Shapley ─────────────────────────────────
        # Model-agnostic black-box approach: wraps the PyTorch model in a
        # sklearn-style predictor and evaluates v(S; x) via forward passes.
        # No backpropagation — avoids the additivity issues of DeepExplainer.
        if args.group_by != "none":
            wrapper = SklearnWrapperForGroupedShap(
                model=model,
                feature_cols=feature_cols,
                x_mean=x_mean,
                x_std=x_std,
                y_mean=float(y_mean),
                y_std=float(y_std),
            )
            rng = np.random.RandomState(args.random_state + fold_idx)
            # Build DataFrames with original column names (wrapper handles scaling)
            X_train_df = X.iloc[train_idx].reset_index(drop=True)
            X_test_df = X.iloc[test_idx].reset_index(drop=True)
            bg_idx = rng.choice(
                len(X_train_df),
                size=min(args.n_background_grouped, len(X_train_df)),
                replace=False,
            )
            eval_idx = rng.choice(
                len(X_test_df),
                size=min(args.n_eval_grouped, len(X_test_df)),
                replace=False,
            )
            X_bg = X_train_df.iloc[bg_idx].reset_index(drop=True)
            X_eval = X_test_df.iloc[eval_idx].reset_index(drop=True)

            phi_df = grouped_shapley_all(wrapper, X_bg, X_eval, group_cols_list, group_names)
            phi_df["fold"] = fold_idx
            grouped_tables.append(phi_df)

    grouped_df = pd.concat(grouped_tables, ignore_index=True) if grouped_tables else pd.DataFrame()
    return pd.DataFrame(metrics), pd.concat(shap_tables, ignore_index=True), grouped_df


def summarize_shap(shap_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = ["feature", "feature_type", "snp_index", "chromosome", "position"]
    return (
        shap_df.groupby(keep_cols, dropna=False, as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            sd_abs_shap=("mean_abs_shap", "std"),
            n_folds=("mean_abs_shap", "count"),
        )
        .sort_values("mean_abs_shap", ascending=False)
    )


def write_outputs(
    outdir: Path,
    args: argparse.Namespace,
    X: pd.DataFrame,
    y: pd.Series,
    metrics_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    shap_summary: pd.DataFrame,
    grouped_df: pd.DataFrame,
    num_snps: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(outdir / "cv_metrics.tsv", sep="\t", index=False)
    shap_df.to_csv(outdir / "fold_shap_importance.tsv", sep="\t", index=False)
    shap_summary.to_csv(outdir / "global_shap_ranking.tsv", sep="\t", index=False)

    if not grouped_df.empty:
        group_names = [c for c in grouped_df.columns if c != "fold"]
        summary_grp = summarize_grouped_shapley(
            grouped_df.drop(columns=["fold"]), group_names
        )
        grouped_df.to_csv(outdir / "fold_grouped_shapley.tsv", sep="\t", index=False)
        summary_grp.to_csv(outdir / "grouped_shapley_ranking.tsv", sep="\t", index=False)

    summary_lines = [
        f"study_values_file: {args.study_values_file}",
        f"phenotype_col: {args.phenotype_col}",
        f"model: {args.model}",
        f"matched_accessions: {len(y)}",
        f"top_snps: {args.top_snps}",
        f"n_pcs: {args.n_pcs}",
        f"folds: {args.folds}",
        f"epochs: {args.epochs}",
        f"d_model: {args.d_model}",
        f"n_layers: {args.n_layers}",
        f"group_by: {args.group_by}",
        f"n_background_grouped: {args.n_background_grouped}",
        f"n_eval_grouped: {args.n_eval_grouped}",
        f"debug: {args.debug}",
        f"feature_count_total: {X.shape[1]}",
        f"feature_count_snps: {num_snps}",
        f"mean_cv_r2: {metrics_df['r2'].mean():.6f}",
        f"mean_cv_rmse: {metrics_df['rmse'].mean():.6f}",
    ]
    (outdir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    apply_debug_overrides(args)
    set_seed(args.random_state)
    X, y, feature_meta, num_snps = load_arabidopsis_matrix(args)
    metrics_df, shap_df, grouped_df = run_cv(X, y, feature_meta, args, num_snps)
    shap_summary = summarize_shap(shap_df)
    outdir = Path(args.outdir)
    write_outputs(outdir, args, X, y, metrics_df, shap_df, shap_summary, grouped_df, num_snps)
    print(f"Saved outputs to: {outdir}")
    print(f"Model: {args.model}")
    print(f"Matched accessions: {len(y)}")
    print(f"Mean CV R2: {metrics_df['r2'].mean():.4f}")
    if not grouped_df.empty:
        group_names = [c for c in grouped_df.columns if c != "fold"]
        summary_grp = summarize_grouped_shapley(grouped_df.drop(columns=["fold"]), group_names)
        print(f"Top chromosome group: {summary_grp.iloc[0]['group']}")


if __name__ == "__main__":
    main()

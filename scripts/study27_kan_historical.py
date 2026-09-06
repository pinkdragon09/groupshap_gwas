#!/usr/bin/env python3
"""Add KAN to the historical Study 27 preselected-panel comparison.

Uses the original matrix loader and five folds. Historical RF/Transformer scores
are read unchanged from saved files. Feature selection used all phenotypes.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from kan import KAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from transformer_mamba_gwas_template import load_arabidopsis_matrix
from study27_kan_comparison import exact_group_game, predictor

ROOT = Path(__file__).resolve().parents[1]


def fit_kan(X, y, fit, val, seed, out):
    """Match historical Transformer scaling, minibatches and validation protocol."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = KAN(width=[X.shape[1], 4, 1], grid=3, k=3, seed=seed,
                auto_save=False, symbolic_enabled=False, ckpt_path=str(out / "checkpoints"))
    mean, sd = float(y[fit].mean()), max(float(y[fit].std()), 1e-8)
    xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor((y-mean)/sd, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    best, state, stale, best_epoch = np.inf, None, 0, 0
    trace = []
    for epoch in range(80):
        model.train()
        order = torch.randperm(len(fit)).numpy()
        for start in range(0, len(order), 16):
            ix = fit[order[start:start+16]]
            optimizer.zero_grad()
            loss = ((model(xt[ix]).reshape(-1)-yt[ix])**2).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite KAN loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = float(((model(xt[val]).reshape(-1)-yt[val])**2).mean())
            tl = float(((model(xt[fit]).reshape(-1)-yt[fit])**2).mean())
        trace.append(dict(epoch=epoch+1, train_mse_standardized=tl, validation_mse_standardized=vl))
        if vl < best:
            best, stale, best_epoch = vl, 0, epoch+1
            state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(state)
    model.eval()
    return predictor(model, mean, sd), trace, best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs/study27_kan_historical")
    parser.add_argument("--report-only", action="store_true", help="Summarize existing saved fits without refitting")
    args = parser.parse_args()
    out = args.outdir.resolve()
    if args.report_only:
        summarize(out)
        return
    if (out / "cv_metrics.tsv").exists():
        raise SystemExit("Choose a fresh output directory; fit results already exist.")
    out.mkdir(parents=True, exist_ok=True)
    base = ROOT / "outputs/full_baseline_gwas/study27_mean_trl_top_hits.tsv"
    cfg = SimpleNamespace(pheno_zip=ROOT / "data/arapheno/bulk/database.zip",
         geno_h5=ROOT / "data/1001genomes/genotypes/1001_SNP_MATRIX/imputed_snps_binary.hdf5",
         kinship_h5=ROOT / "data/1001genomes/genotypes/1001_SNP_MATRIX/kinship_ibs_mac5.hdf5",
         baseline_hits=base, study_values_file="27/study_27_values.csv", phenotype_col="Mean(TRL) C",
         snp_selection_mode="linear_hits", top_snps=500, n_pcs=3)
    Xdf, ys, feature_meta, p = load_arabidopsis_matrix(cfg)
    assert Xdf.shape == (81, 203) and p == 200
    X, y = Xdf.to_numpy(np.float32), ys.to_numpy(np.float32)
    feature_meta.to_csv(out / "feature_meta.tsv", sep="\t", index=False)
    with __import__("h5py").File(cfg.geno_h5) as f:
        ga = [a.decode() if isinstance(a, bytes) else str(a) for a in f["accessions"][:]]
    from full_baseline_gwas import load_phenotype
    ym, _ = load_phenotype(cfg.pheno_zip, cfg.study_values_file, cfg.phenotype_col)
    accessions = [a for a in ga if a in ym]
    np.testing.assert_allclose(y, [ym[a] for a in accessions], rtol=1e-6)
    config = dict(protocol="historical full-sample preselected panel", accessions=81, snps=200,
        n_pcs=3, folds=5, split_seed=42, model_seed="42 + one-based fold",
        kan_width=[203,4,1], grid=3, spline_degree=3, epochs=80, patience=10,
        batch_size=16, learning_rate=.001, weight_decay=.0001,
        preprocessing="historical Transformer fit-only z-score scaling",
        selection_caveat="all phenotypes entered panel selection; CV conditions on that panel",
        panel_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions={k:metadata.version(k) for k in ["torch","pykan","numpy","scipy","scikit-learn","h5py"]})
    (out / "run_config.json").write_text(json.dumps(config, indent=2))
    metrics, predictions, traces, groups, pairs, splitrows, rfcheck = [],[],[],[],[],[],[]
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=42).split(X), start=1):
            perm = np.random.RandomState(42+fold).permutation(len(tr))
            shuffled = tr[perm]
            nval = max(1, int(round(.2 * len(tr))))
            val, fit = shuffled[:nval], shuffled[nval:]
            mean, sd = X[fit].mean(0), X[fit].std(0)
            sd[sd < 1e-8] = 1.
            Z = (X-mean)/sd
            predict, trace, ep = fit_kan(Z, y, fit, val, 42+fold, out)
            yh = predict(Z[te])
            metrics.append(dict(fold=fold, r2=r2_score(y[te], yh), rmse=mean_squared_error(y[te], yh)**.5,
                                n_train=len(tr), n_test=len(te), n_fit=len(fit), best_epoch=ep))
            predictions.extend(dict(fold=fold, accession=accessions[i], row=int(i), observed=float(y[i]),
                                    predicted=float(v)) for i,v in zip(te,yh))
            traces.extend(dict(fold=fold, **r) for r in trace)
            for role, ix in [("fit",fit),("validation",val),("test",te)]:
                splitrows.extend(dict(fold=fold, role=role, row=int(i), accession=accessions[i]) for i in ix)
            labels, phi, interaction, err = exact_group_game(predict, Z, fit, te,
                feature_meta.loc[feature_meta.feature_type == "snp", "chromosome"].to_numpy(), 9000+fold)
            groups.extend(dict(fold=fold, group=label, mean_abs_shap=float(np.abs(phi[:,j]).mean()),
                               efficiency_error=err) for j,label in enumerate(labels))
            pairs.extend(dict(fold=fold, group1=a, group2=b, mean_abs_interaction=v) for a,b,v in interaction)
            # Independent RF rerun checks historical panel/accession/fold alignment.
            rf = RandomForestRegressor(n_estimators=500, max_features="sqrt", min_samples_leaf=2,
                                       random_state=42+fold, n_jobs=2).fit(X[tr], ys.to_numpy()[tr])
            rfcheck.append(dict(fold=fold, r2=r2_score(ys.to_numpy()[te], rf.predict(X[te]))))
            print(f"fold {fold}: KAN R2={metrics[-1]['r2']:.6f}, epoch={ep}; RF check={rfcheck[-1]['r2']:.6f}", flush=True)
    for name, rows in [("cv_metrics",metrics),("predictions",predictions),("training_trace",traces),
                       ("group_shap",groups),("group_interactions",pairs),("splits",splitrows),("rf_reproduction",rfcheck)]:
        pd.DataFrame(rows).to_csv(out / (name+".tsv"), sep="\t", index=False)
    summarize(out)


def summarize(out):
    metrics = pd.read_csv(out / "cv_metrics.tsv", sep="\t").to_dict("records")
    rfcheck = pd.read_csv(out / "rf_reproduction.tsv", sep="\t")
    predictions = pd.read_csv(out / "predictions.tsv", sep="\t")
    assert len(predictions) == 81 and predictions.accession.nunique() == 81
    for r in metrics:
        v = predictions[predictions.fold == r["fold"]]
        np.testing.assert_allclose(r["r2"], r2_score(v.observed, v.predicted), atol=1e-6)
    historical = {}
    for name, folder in [("random_forest","random_forest_gwas_full"),("transformer","transformer_gwas_full")]:
        historical[name] = pd.read_csv(ROOT / "outputs" / folder / "cv_metrics.tsv", sep="\t")
    rf_difference = float(np.max(np.abs(rfcheck.r2 - historical["random_forest"].r2)))
    current_panel = pd.read_csv(out / "feature_meta.tsv", sep="\t")
    for folder in ["random_forest_gwas_full", "transformer_gwas_full"]:
        old_panel = pd.read_csv(ROOT / "outputs" / folder / "global_shap_ranking.tsv", sep="\t")
        np.testing.assert_array_equal(
            np.sort(current_panel.loc[current_panel.feature_type == "snp", "snp_index"].astype(int)),
            np.sort(old_panel.loc[old_panel.feature_type == "snp", "snp_index"].astype(int)))
    summary = []
    for name, frame in list(historical.items()) + [("kan", pd.DataFrame(metrics))]:
        summary.append(dict(model=name, mean_cv_r2=frame.r2.mean(), sd_cv_r2=frame.r2.std(),
                            mean_cv_rmse=frame.rmse.mean(), source="historical saved" if name != "kan" else "new KAN run"))
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "model_comparison.tsv", sep="\t", index=False)
    text = ["# Study 27: historical-panel RF–Transformer–KAN comparison", "",
            "Same historical 200 SNPs, 81 accessions, three kinship PCs and five outer folds.",
            "RF and Transformer values are retained from their saved runs; KAN is newly fitted.", "",
            "| Model | Mean fold R² | Fold SD | Mean RMSE | Source |", "|---|---:|---:|---:|---|"]
    for r in summary.itertuples():
        text.append(f"| {r.model} | {r.mean_cv_r2:.4f} | {r.sd_cv_r2:.4f} | {r.mean_cv_rmse:.4f} | {r.source} |")
    text += ["", "KAN uses widths [203,4,1], grid 3, cubic splines, and the historical Transformer",
             "inner split, fit-only input/target standardization, 80-epoch maximum, minibatches of 16,",
             "and validation early stopping (patience 10). The best validation checkpoint is evaluated",
             "on the outer test fold. No test-fold tuning was performed. Hyperparameters were fixed before fitting.", "",
             f"RF rerun mean R² is {rfcheck.r2.mean():.6f}; historical RF mean is {historical["random_forest"].r2.mean():.6f}.",
             f"Maximum fold difference is {rf_difference:.6f}; this is not bit-for-bit reproduction in the current runtime.",
             "The SNP indices match both historical model panels exactly. The original split algorithm/seed is reused.",
             "Historical RF and Transformer rows remain unchanged. KAN predictions, training traces, splits, feature metadata",
             "and chromosome-group attributions are saved alongside this report.", "",
             "## Why these scores differ from the training-selected comparison", "",
             "Historical selection ranked SNPs using all 81 phenotypes before CV. The other experiment",
             "reselected SNPs using only each training split and also changed the neural fitting configuration.",
             "R² = 1 - SSE/SST is negative when prediction errors exceed test-fold variation around its mean.",
             "Negative R² is possible and does not mean a negative squared correlation.", "",
             "This is the requested comparison conditional on the historical panel. Because held-out",
             "phenotypes participated in panel selection, these scores are not unbiased estimates of",
             "the full selection-plus-prediction pipeline on new accessions. The separate training-selected",
             "experiment is retained as a sensitivity analysis; it does not replace these historical rows.", "",
             "Fold SD is descriptive, not a confidence interval. Chromosome-group interactions do not",
             "identify SNP-pair epistasis or establish causal genes.", "",
             "Reproduce with `.venv-kan/bin/python scripts/study27_kan_historical.py --outdir outputs/NEW_DIRECTORY`.",
             "Regenerate this report without refitting using `--report-only`."]
    (out / "REPORT.md").write_text("\n".join(text)+"\n")
    (out / "COMPLETE.json").write_text(json.dumps(dict(historical_panel_match=True, rf_max_abs_fold_r2_difference=rf_difference, folds=5, accessions=81)))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

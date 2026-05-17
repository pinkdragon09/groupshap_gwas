# Target-normalization follow-up plan

**Note date:** 2026-05-17 10:38:06

The proposed comparison would hold model, data splits and interaction estimator
fixed while comparing KAN training with raw targets against standardized targets.
The first gate was a Lactate normalization experiment. If that gate did not improve
the selected recovery metric, the further yeast-Shapley and Study 43 tasks were to stop.

For the yeast-Shapley task, inspect `scripts/history/yeast_bloom_kan_train.py`
and confirm it fits KAN weights using raw targets. If so, compute the target
mean and standard deviation from fitting rows, standardize the fitting targets,
and apply that same transform to validation/test targets. The first proposed
phenotype was Neomycin. Compare recovery and prediction metrics using the same
Shapley pipeline and clearly consistent target units.

For the AraPheno Study 43 task, inspect `scripts/history/run_joint_kan_study43.py`.
Only attempt the corresponding raw-versus-standardized comparison if the script
actually trains KAN weights; skip it if it only loads weights or uses another model.

Write each comparison to a separate output location and preserve existing numerical
results. These are planned steps, not a record of a completed patch or rerun.

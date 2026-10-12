# FT10 RF training-data GroupSHAP

**Note date:** 2026-10-10

The extension applied GroupSHAP to all 15 saved RF checkpoints from the FT10
five-fold analysis: five folds and three seeds. Each fold explained all 108 training
rows without refitting. It reused the corresponding held-out analysis's 16 training
background donors and 128 gene-game paths/orders. The chromosome game was
enumerated exactly, and training evaluation included the background donors.

The retained runner is `scripts/history/run_ft10_training_groupshap.py`.
Numerical outputs, local arrays and sensitivity diagnostics are under
`outputs/history/ft10/ft10_training_groupshap_20261010/`.

All 15 checkpoints reproduced saved test predictions. Checks covered group
partitions, preprocessing, sample membership, identical donor/path orders and
efficiency; the maximum gene/chromosome efficiency error was 1.42e-14.
Three FT10 estimator tests passed, and 97 input/output hashes were independently
verified in the recorded session.

Recorded training mean fold R² was 0.7896 versus 0.3957 held out. The median
within-model training/held-out gene-rank correlation was 0.9677, with range
0.7854–0.9855. Most leading genes were represented in only one fold; DOG1 was
assigned a gene player only in fold 4.

Training explanations describe the fitted models on their training data.
They do not add missing canonical genes to fold-specific panels or establish
causal effects or additional held-out predictive value from SNPs.

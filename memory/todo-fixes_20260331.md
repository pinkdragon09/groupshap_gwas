# Reproducibility engineering tasks

**Note date:** 2026-03-31

The genotype-to-phenotype workflow needed a clearer way to run its stages and
reconstruct its environment. The planned engineering work was:

- Provide one entry point that executes the pipeline stages in order.
- Pin versions of PyTorch, SHAP, scikit-learn and pandas.
- Separate generated outputs from source files through `.gitignore` rules.

These were proposed tasks at the note date. This record does not claim that an
entry point or dependency lock file had been completed.

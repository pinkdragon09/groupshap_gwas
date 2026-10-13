# Column-name-based grouped Shapley summary fix

**Note date:** 2026-04-14 14:32:02

The grouped Shapley summary previously risked interpreting concatenated fold
outputs using the final fold's group-name list. That failed when different folds
contained different groups.

`summarize_grouped_shapley` in `scripts/grouped_shap_utils.py` was changed to
summarize actual named columns and handle missing values. The retained helper
excludes the fold column and computes statistics from finite values for each group.

Matching names alone are insufficient when an LD-derived label refers to different
SNP sets across folds. Those cases still require separate fold summaries or an
explicitly stable group definition.

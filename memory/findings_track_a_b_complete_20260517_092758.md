# Target normalization and interaction-estimator comparability

**Note date:** 2026-05-17 09:27:58

A target-normalization experiment ranked SNP pairs using mixed second derivatives
computed with automatic differentiation. The retained Bloom KAN benchmark instead
ranked pairs with Shapley interaction values, computed by
`scripts/history/yeast_bloom_kan_interaction.py`.

Mixed-Hessian importance and Shapley interaction values are different quantities.
Consequently, a comparison that changes both target normalization and the
interaction estimator cannot isolate the effect of normalization.
An otherwise matched target-normalized run of the Shapley pipeline had not been
completed at this note date.

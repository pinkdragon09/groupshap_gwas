# Bloom yeast RF interaction benchmark configuration

**Note date:** 2026-04-26 01:24:58

The RF used 200 trees, `max_features="sqrt"`, `min_samples_leaf=2`, seed 42
and parallel tree fitting. It fitted the 2013 matrix of 11,623 BY×RM markers.
Five-fold shuffled CV with seed 42 measured prediction; a subsequent fit on all
rows supplied the model for attribution. Thus the attribution evaluation in this
early experiment was not an outer-fold held-out explanation design.

Each marker was a single-feature group. Pairwise Shapley interactions used the
batched NumPy implementation in `scripts/history/yeast_bloom_fast_interaction.py`,
with 30 coalition samples, 15 background rows, 15 evaluation rows and seed 42.
Eight process workers each used an RF predictor with `n_jobs=1` to avoid nested
tree parallelism. A sweep over 4,656 or 4,753 candidate pairs took about 12 minutes
per phenotype in the recorded run.

The reference set was the Bloom 2015 supplementary QTL×QTL table, containing
205 pairs across 19 traits. For a phenotype, its reference loci were matched to
the closest panel markers on the same chromosome after normalizing chromosome
labels. Both endpoints had to be within 50 kb of their matched markers for a
reference pair to count as represented in the panel.

Candidate pairs were ranked either by mean absolute Shapley interaction or by
the sum of the two marginal-GWAS negative log10 p-values. Recall at K=10, 25, 50
and 100 was the fraction of represented reference pairs appearing among the
top K ranked pairs. The reference list defines a recovery benchmark; it is not
a claim that every other pair is biologically noninteracting.

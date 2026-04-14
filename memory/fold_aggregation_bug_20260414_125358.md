# Fold-dependent group aggregation bug

**Note date:** 2026-04-14 12:53:58

The cross-validation workflow rebuilt SNP groups inside each fold but passed only
the final fold's group names to the summary step. When groups differed between
folds, that final name list could mislabel or omit values accumulated from other folds.

The problematic aggregation pattern was:

```python
# group_names contains only the final fold's grouping.
summary = summarize_grouped_shapley(grouped_df.drop(columns=["fold"]), group_names)
```

The issue affected data-dependent LD/PLINK groupings. Fixed genomic windows with
unchanged membership across folds did not have this particular mismatch.

The proposed repair was to summarize each fold separately and aggregate across
folds only when group labels and SNP membership coincide. The affected shared
summary function is `summarize_grouped_shapley` in `scripts/grouped_shap_utils.py`.

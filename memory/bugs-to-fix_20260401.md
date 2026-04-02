# Grouped Shapley sampler and model-wrapper defects

**Note date:** 2026-04-01

The following issues were identified in the grouped attribution helper during an April 1 code review. They describe that implementation state, not a current open-bug list.

## BUG-1 (CRITICAL) — `grouped_shapley_interaction_single_mc` estimator is biased after linter rewrite

**File**: `scripts/grouped_shap_utils.py`, `grouped_shapley_interaction_single_mc()`

**Problem**: The linter rewrote the sampling strategy from the provably-unbiased
"uniform size + prefix of rest-permutation" to a "max-position of full permutation" approach.

Old approach gave P(specific S of size s) = s!(m−s−2)!/(m−1)! = w(s,m) ✓ (proved analytically)

New approach gives P(specific S of size s) = 2(s+1)!(m−s−2)!/m!
The ratio 2(s+1)/m varies with s → biased for the standard Shapley Interaction Index.

The docstring says "S_t is the set of players appearing before j and k" — should be
min(pos_j, pos_k) for "before both", but even min doesn't give SII weights.
Only the old size-then-prefix-of-rest approach does.

**Fix**: Revert to the original unbiased estimator:
```python
rest = [g for g in range(m) if g not in (j, k)]
perm = np.random.permutation(rest)
s_size = np.random.randint(0, len(rest) + 1)  # uniform over {0,...,m-2}
S = frozenset(perm[:s_size])
```

**Proof of correctness**:
P(size=s) = 1/(m-1)  [m-1 choices: 0,...,m-2]
P(specific unordered S of size s) = s!(m-2-s)!/(m-2)!  [= 1/C(m-2,s)]
P(specific S of size s) = s!(m-2-s)!/((m-1)(m-2)!) = s!(m-s-2)!/(m-1)! = w(s,m) ✓

---

## BUG-2 (MAJOR) — `value_function_numpy` drops DataFrame conversion; breaks model wrappers

**File**: `scripts/grouped_shap_utils.py`, `value_function_numpy()`

**Problem**: Linter removed the `feature_names` parameter and removed the
`pd.DataFrame(X_syn, columns=feature_names)` conversion before calling
`model.predict()`. Now passes raw NumPy array directly.

This silently breaks `SklearnWrapperForGroupedShap` in
`scripts/transformer_mamba_gwas_template.py`, which expects a DataFrame input.
Sklearn models accept NumPy, but any wrapped model (transformer, pipeline with
column-name-aware preprocessing) will fail or silently produce wrong results.

**Fix**: Either restore the DataFrame conversion, or document the breaking change
and update all model wrappers to accept NumPy. The transformer wrapper needs
to be verified explicitly.

---

## BUG-3 (MINOR) — `value_function_numpy` shape assertion removed

**File**: `scripts/grouped_shap_utils.py`, `value_function_numpy()`

**Problem**: The `assert eval_x_arr.ndim == 1` guard added during the quality
review was dropped by the linter rewrite.

**Fix**: Re-add:
```python
assert eval_x_arr.ndim == 1, f"eval_x_arr must be 1-D, got shape {eval_x_arr.shape}"
```

---

## BUG-4 (MINOR) — `load_group_manifest` error message truncated

**File**: `scripts/grouped_shap_utils.py`, `load_group_manifest()`

**Problem**: The linter truncated the error message — it no longer shows which
columns were actually found:
```python
# Old (informative):
raise ValueError(f"...must have 'feature' and 'group' columns. Found: {list(manifest_df.columns)}")
# New (missing context):
raise ValueError(f"...must have 'feature' and 'group' columns.")
```

**Fix**: Restore `Found: {list(manifest_df.columns)}` in the error string.

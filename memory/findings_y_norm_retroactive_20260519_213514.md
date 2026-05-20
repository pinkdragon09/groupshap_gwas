# Normalization follow-ups not executed

**Note date:** 2026-05-19 21:35:14

The normalization plan required a Lactate pilot to improve its selected recovery
metric before proceeding with two further tasks. That stop condition was reached,
so the following tasks were not executed:

- The yeast KAN/Shapley task: `scripts/history/yeast_bloom_kan_train.py` and
  `scripts/history/yeast_bloom_kan_recall.py` were not patched, rerun or compared.
- The AraPheno Study 43 task: `scripts/history/run_joint_kan_study43.py` was not
  inspected for this normalization change, patched or rerun.

This record defines the work left unperformed as of May 19. It does not describe
later changes to those workflows or supply evidence for a normalization effect.

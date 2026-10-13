# pykan KANLayer seed-argument fix

**Note date:** 2026-04-26 19:36:39

`scripts/history/yeast_bloom_kan_affine_ablation.py` attempted to pass `seed=seed`
to `KANLayer`. pykan 0.2.8 did not accept that keyword. The repair removed the
constructor keyword while retaining the preceding `torch.manual_seed(seed)` call
to initialize the random-number generator used for the layer's parameters.

```python
torch.manual_seed(seed)
self.inner = KANLayer(in_dim=n_features, out_dim=hidden, num=grid, k=k)
```

The ablation replaced the outer mapping with `torch.nn.Linear(8, 1, bias=True)`.
The checked weight and bias shapes were `(1, 8)` and `(1,)`.
Training used Adam, learning rate 0.001, batch size 64, 200 epochs and patience 25.

Pairwise Shapley interaction evaluation ran sequentially on 12 reference pairs
per phenotype, using 30 coalition samples, 15 background rows and 15 evaluation
rows. The seed for row index `ridx` in the pair table was `42 + ridx * 1009`.
The two phenotypes were Neomycin and Lactate. These settings describe the
implemented compatibility fix and control, not a biological interaction conclusion.

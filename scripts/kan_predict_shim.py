"""Wrap a fitted pykan KAN with a numpy `.predict(X)` interface.

The Shapley fast-pairwise estimator in
``yeast_bloom_kan_interaction.py`` calls ``model.predict(X)`` repeatedly
with large batched numpy arrays. pykan's KAN is a ``torch.nn.Module``;
this shim handles the conversion + ``torch.no_grad()`` context.
"""
from __future__ import annotations

import numpy as np
import torch


class KanPredictShim:
    """Numpy `.predict(X)` shim around a torch KAN.

    Stays on CPU; the model is assumed to have been moved to CPU and put
    in ``.eval()`` mode by the caller.
    """

    def __init__(self, kan_model, dtype=torch.float32):
        self.model = kan_model
        self.dtype = dtype

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.as_tensor(X, dtype=self.dtype)
        with torch.no_grad():
            out = self.model(X_t)
        out = out.cpu().numpy()
        return out.ravel() if out.ndim > 1 else out

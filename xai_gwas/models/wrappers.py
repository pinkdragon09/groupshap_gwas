"""Standardized sklearn-compatible model wrappers.

All wrappers expose a ``predict(X: DataFrame | ndarray) -> ndarray`` method
so they can be passed directly to the attribution functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    n_estimators: int = 300,
    max_features: float = 0.33,
    min_samples_leaf: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
):
    """Fit a Random Forest regressor with sensible genomics defaults.

    Parameters
    ----------
    X_train        : Training genotype DataFrame.
    y_train        : Phenotype values.
    n_estimators   : Number of trees.
    max_features   : Fraction of features per split (0.33 ≈ sqrt for wide data).
    min_samples_leaf: Minimum samples per leaf (5 gives conservative bias-variance).
    n_jobs         : Joblib parallelism.
    random_state   : RNG seed.

    Returns
    -------
    Fitted sklearn RandomForestRegressor.
    """
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    return rf


# ---------------------------------------------------------------------------
# KAN shim (requires pykan or efficient-kan)
# ---------------------------------------------------------------------------

class KanPredictShim:
    """Numpy ``predict(X)`` shim around a PyTorch KAN model.

    The Shapley estimator calls ``model.predict(X)`` with large batched
    numpy arrays.  pykan's KAN is a ``torch.nn.Module``; this shim handles
    the conversion and ``torch.no_grad()`` context.

    Parameters
    ----------
    kan_model : Fitted pykan or efficient-kan model on CPU in eval() mode.
    dtype     : Torch dtype for the input tensor (default float32).
    """

    def __init__(self, kan_model, dtype=None):
        self.model = kan_model
        if dtype is None:
            import torch
            self.dtype = torch.float32
        else:
            self.dtype = dtype

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        import torch

        if isinstance(X, pd.DataFrame):
            X = X.values
        X_t = torch.as_tensor(X, dtype=self.dtype)
        with torch.no_grad():
            out = self.model(X_t)
        out = out.cpu().numpy()
        return out.ravel() if out.ndim > 1 else out

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_class,
        model_kwargs: dict,
        dtype=None,
    ) -> "KanPredictShim":
        """Load a KAN from a torch checkpoint file.

        Parameters
        ----------
        checkpoint_path : Path to the ``.pt`` or ``.pth`` file.
        model_class     : The KAN class (e.g. ``KAN`` from pykan).
        model_kwargs    : Constructor kwargs for model_class.
        dtype           : Torch dtype override.
        """
        import torch

        model = model_class(**model_kwargs)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return cls(model, dtype=dtype)


# ---------------------------------------------------------------------------
# MLP wrapper (vanilla PyTorch)
# ---------------------------------------------------------------------------

class MLPWrapper:
    """Simple fully-connected MLP with ``predict()`` interface.

    Use this when a lightweight nonlinear baseline is needed without the
    KAN dependency.

    Parameters
    ----------
    hidden_sizes : Sequence of hidden layer widths.
    activation   : Activation function name ('relu', 'tanh', 'gelu').
    random_state : RNG seed for weight initialisation.
    max_iter     : Number of training epochs.
    lr           : Adam learning rate.
    batch_size   : Minibatch size.
    device       : 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        hidden_sizes: list[int] = (64, 32),
        activation: str = "relu",
        random_state: int = 42,
        max_iter: int = 200,
        lr: float = 1e-3,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.hidden_sizes = list(hidden_sizes)
        self.activation = activation
        self.random_state = random_state
        self.max_iter = max_iter
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self._net = None

    def _build_net(self, n_features: int):
        import torch
        import torch.nn as nn

        act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        ActCls = act_map.get(self.activation, nn.ReLU)

        layers: list[nn.Module] = []
        in_dim = n_features
        for h in self.hidden_sizes:
            layers += [nn.Linear(in_dim, h), ActCls()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers).to(self.device)

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "MLPWrapper":
        import torch
        import torch.optim as optim

        torch.manual_seed(self.random_state)
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.as_tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)

        self._net = self._build_net(X.shape[1])
        optimizer = optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        n = len(X_t)
        for _ in range(self.max_iter):
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start: start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self._net(X_t[idx]), y_t[idx])
                loss.backward()
                optimizer.step()
        self._net.eval()
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        import torch

        if self._net is None:
            raise RuntimeError("Call fit() before predict().")
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self._net(X_t).cpu().numpy().ravel()
        return out

"""
pipeline.models.lstm
====================
PyTorch LSTM architectures for sequence-based forecasting.

Two variants
------------
**LSTMModel**  — Plain LSTM → dense head.
  Input: ``(batch, seq_len=P, n_features)``
  Output: ``(batch, 1)``

**LSTMPreModel** — Dense layers per time-step → LSTM → dense head.
  Input:  same as above
  The dense "pre-processor" learns a richer per-step representation
  before the recurrent block.

Both expose a sklearn-compatible ``.fit(X, y)`` / ``.predict(X)`` API so
they integrate seamlessly with the experiment runner.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ======================================================================== #
#  PyTorch modules                                                          #
# ======================================================================== #

if TORCH_AVAILABLE:

    class _LSTMNet(nn.Module):
        """Plain LSTM → linear head."""

        def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            # x: (B, T, F)
            out, _ = self.lstm(x)
            last = out[:, -1, :]  # last time-step
            return self.head(last).squeeze(-1)

    class _LSTMPreNet(nn.Module):
        """Dense pre-processor per time-step → LSTM → linear head."""

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            pre_dense_sizes: List[int],
        ):
            super().__init__()
            layers = []
            in_dim = input_size
            for out_dim in pre_dense_sizes:
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = out_dim
            self.pre = nn.Sequential(*layers)

            self.lstm = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            # x: (B, T, F)
            B, T, F = x.shape
            x_flat = x.reshape(B * T, F)
            x_pre = self.pre(x_flat).reshape(B, T, -1)
            out, _ = self.lstm(x_pre)
            last = out[:, -1, :]
            return self.head(last).squeeze(-1)


# ======================================================================== #
#  Sklearn-compatible wrappers                                              #
# ======================================================================== #

class _BaseLSTMWrapper:
    """
    Base class with sklearn-style fit / predict that accepts:
      - 2-D tabular input  ``(N, P * n_features)`` — auto-reshaped to 3-D
      - 3-D sequence input ``(N, P, n_features)``
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 256,
        patience: int = 10,
        device: Optional[str] = None,
        P: Optional[int] = None,
        n_features: Optional[int] = None,
        **kwargs,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM models.")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.P = P
        self.n_features = n_features
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = None
        self.extra_kwargs = kwargs

    def _build_net(self, input_size: int) -> nn.Module:
        raise NotImplementedError

    def _to_tensor(self, X):
        if isinstance(X, np.ndarray):
            if X.ndim == 2:
                # Reshape flat tabular → (N, P, F)
                if self.P is None or self.n_features is None:
                    raise ValueError(
                        "For 2-D input, set P and n_features on the model "
                        "so it can reshape to (N, P, n_features)."
                    )
                X = X.reshape(-1, self.P, self.n_features)
            return torch.tensor(X, dtype=torch.float32)
        return X

    def fit(self, X, y, X_val=None, y_val=None):
        X_t = self._to_tensor(X).to(self.device)
        y_t = torch.tensor(np.asarray(y, dtype=np.float32)).to(self.device)

        input_size = X_t.shape[-1]
        self.model_ = self._build_net(input_size).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        patience_ctr = 0

        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # Validation
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v = self._to_tensor(X_val).to(self.device)
            y_v = torch.tensor(np.asarray(y_val, dtype=np.float32)).to(self.device)

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                pred = self.model_(xb)
                loss = criterion(pred, yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(ds)

            # Early stopping on validation
            if has_val:
                self.model_.eval()
                with torch.no_grad():
                    val_pred = self.model_(X_v)
                    val_loss = criterion(val_pred, y_v).item()
            else:
                val_loss = epoch_loss

            if val_loss < best_loss:
                best_loss = val_loss
                patience_ctr = 0
                self._best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    break

        # Restore best
        if hasattr(self, "_best_state"):
            self.model_.load_state_dict(self._best_state)
        self.model_.eval()
        return self

    def predict(self, X):
        X_t = self._to_tensor(X).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(X_t).cpu().numpy()
        return np.clip(preds, 0.0, 1.0)


class LSTMModel(_BaseLSTMWrapper):
    """Plain LSTM wrapper (sklearn-compatible)."""

    def _build_net(self, input_size):
        return _LSTMNet(input_size, self.hidden_size, self.num_layers, self.dropout)


class LSTMPreModel(_BaseLSTMWrapper):
    """Dense-pre → LSTM wrapper (sklearn-compatible)."""

    def __init__(self, pre_dense_sizes: Optional[List[int]] = None, **kwargs):
        super().__init__(**kwargs)
        self.pre_dense_sizes = pre_dense_sizes or [128, 64]

    def _build_net(self, input_size):
        return _LSTMPreNet(
            input_size,
            self.hidden_size,
            self.num_layers,
            self.dropout,
            self.pre_dense_sizes,
        )

"""
PyTorch LSTM model for sequential trading signal prediction.

Two public classes are provided:

* ``LSTMModel``    — the nn.Module (architecture only).
* ``LSTMTrainer`` — stateful wrapper that owns the training loop, evaluation,
                    serialization, and inference, mirroring the interface of
                    XGBoostModel / RandomForestModel.

Input sequences are expected to have shape ``(n_samples, lookback, n_features)``
where ``lookback`` defaults to 60 bars.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detect device once at import time
# ---------------------------------------------------------------------------
_DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
logger.info("LSTMModel using device: %s", _DEVICE)


# ===========================================================================
# Architecture
# ===========================================================================


class LSTMModel(nn.Module):
    """
    Stacked LSTM with a fully-connected head for binary classification.

    Parameters
    ----------
    input_size : int
        Number of features per time step.
    hidden_size : int
        Number of LSTM hidden units.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability applied between LSTM layers (and after the final
        hidden state before the FC head).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_size)

        Returns
        -------
        torch.Tensor, shape (batch,)
            Sigmoid-squashed output — probability of class 1 (bullish).
        """
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        # Take the final time step
        last_hidden = lstm_out[:, -1, :]          # (batch, hidden_size)
        dropped = self.dropout(last_hidden)
        logit = self.fc(dropped)                  # (batch, 1)
        prob = self.sigmoid(logit)                # (batch, 1)
        return prob.squeeze(dim=-1)               # (batch,)


# ===========================================================================
# Training wrapper
# ===========================================================================


class LSTMTrainer:
    """
    Stateful trainer / inference wrapper for :class:`LSTMModel`.

    Attributes
    ----------
    net : LSTMModel
        The underlying network.  None until :meth:`train` or :meth:`load`.
    metrics : dict
        Populated after :meth:`train` completes.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.net: Optional[LSTMModel] = None
        self.metrics: Dict[str, float] = {}
        self._input_size: Optional[int] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_sequences: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 0.001,
        patience: int = 10,
        val_split: float = 0.20,
        pos_weight_factor: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Train the LSTM on sequential data.

        Parameters
        ----------
        X_sequences : np.ndarray, shape (n_samples, lookback, n_features)
        y : np.ndarray, shape (n_samples,)  — binary labels {0, 1}
        epochs : int
            Maximum training epochs.
        batch_size : int
        lr : float
            Initial Adam learning rate.
        patience : int
            Early stopping patience (epochs without val loss improvement).
        val_split : float
            Fraction of data to hold out for validation (chronological split).
        pos_weight_factor : float, optional
            Override for BCEWithLogitsLoss pos_weight.  If None, computed
            automatically from class distribution.

        Returns
        -------
        dict
            Best-epoch metrics on the validation set.
        """
        n_samples, lookback, n_features = X_sequences.shape
        self._input_size = n_features
        logger.info(
            "LSTM training — samples=%d  lookback=%d  features=%d",
            n_samples,
            lookback,
            n_features,
        )

        # ----------------------------------------------------------------
        # Chronological 80/20 split (no shuffle — time series)
        # ----------------------------------------------------------------
        split_idx = int(n_samples * (1.0 - val_split))
        X_tr, X_val = X_sequences[:split_idx], X_sequences[split_idx:]
        y_tr, y_val = y[:split_idx], y[split_idx:]

        # ----------------------------------------------------------------
        # Build model
        # ----------------------------------------------------------------
        self.net = LSTMModel(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(_DEVICE)

        # ----------------------------------------------------------------
        # Class-imbalance: compute pos_weight for BCEWithLogitsLoss
        # ----------------------------------------------------------------
        n_neg_tr = float(np.sum(y_tr == 0))
        n_pos_tr = float(np.sum(y_tr == 1))
        if n_pos_tr == 0:
            raise ValueError("Training fold contains no positive samples.")
        auto_pw = n_neg_tr / n_pos_tr
        pw_value = pos_weight_factor if pos_weight_factor is not None else auto_pw
        pos_weight_tensor = torch.tensor([pw_value], dtype=torch.float32).to(_DEVICE)
        logger.info("BCEWithLogitsLoss pos_weight=%.4f", pw_value)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=False
        )

        # ----------------------------------------------------------------
        # DataLoaders
        # ----------------------------------------------------------------
        train_loader = self._make_loader(X_tr, y_tr, batch_size, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, batch_size, shuffle=False)

        # ----------------------------------------------------------------
        # Training loop with early stopping
        # ----------------------------------------------------------------
        best_val_loss = float("inf")
        best_state: Dict = {}
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(
                train_loader, criterion, optimizer, training=True
            )
            val_loss = self._run_epoch(
                val_loader, criterion, optimizer=None, training=False
            )
            scheduler.step(val_loss)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d  train_loss=%.5f  val_loss=%.5f",
                    epoch,
                    epochs,
                    train_loss,
                    val_loss,
                )

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone() for k, v in self.net.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        "Early stopping triggered at epoch %d "
                        "(patience=%d, best_val_loss=%.5f)",
                        epoch,
                        patience,
                        best_val_loss,
                    )
                    break

        # Restore best weights
        if best_state:
            self.net.load_state_dict(
                {k: v.to(_DEVICE) for k, v in best_state.items()}
            )

        # ----------------------------------------------------------------
        # Final evaluation on validation set
        # ----------------------------------------------------------------
        self.metrics = self._evaluate(X_val, y_val, dataset_label="validation")
        self.metrics["best_val_loss"] = float(best_val_loss)
        return self.metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X_sequences: np.ndarray) -> np.ndarray:
        """
        Return the probability of class 1 (bullish) for each sequence.

        Parameters
        ----------
        X_sequences : np.ndarray, shape (n_samples, lookback, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Probabilities in [0, 1].
        """
        self._assert_fitted()
        self.net.eval()
        tensor = torch.tensor(X_sequences, dtype=torch.float32).to(_DEVICE)
        with torch.no_grad():
            proba = self.net(tensor).cpu().numpy()
        return proba.astype(np.float64)

    def predict(self, X_sequences: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X_sequences) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save model weights and architecture hyperparameters to *path*.

        Parameters
        ----------
        path : str
            File path, e.g. ``"models/lstm_BTCUSDT_1h.pt"``
        """
        self._assert_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "state_dict": self.net.state_dict(),
            "input_size": self._input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "metrics": self.metrics,
        }
        torch.save(checkpoint, path)
        logger.info("LSTMTrainer saved to %s", path)

    def load(self, path: str, input_size: Optional[int] = None) -> None:
        """
        Load a previously saved LSTMTrainer checkpoint from *path*.

        Parameters
        ----------
        path : str
        input_size : int, optional
            Number of features.  Read from the checkpoint if not provided.
        """
        checkpoint = torch.load(path, map_location=_DEVICE)
        resolved_input_size = input_size or checkpoint.get("input_size")
        if resolved_input_size is None:
            raise ValueError(
                "input_size could not be determined.  "
                "Pass it explicitly to load()."
            )
        self._input_size = resolved_input_size
        self.hidden_size = checkpoint.get("hidden_size", self.hidden_size)
        self.num_layers = checkpoint.get("num_layers", self.num_layers)
        self.dropout = checkpoint.get("dropout", self.dropout)
        self.metrics = checkpoint.get("metrics", {})

        self.net = LSTMModel(
            input_size=self._input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(_DEVICE)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()
        logger.info("LSTMTrainer loaded from %s", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        training: bool,
    ) -> float:
        """Run one full pass over *loader*.  Returns mean loss."""
        self.net.train(training)
        total_loss = 0.0
        n_batches = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(_DEVICE)
                y_batch = y_batch.to(_DEVICE)

                # Forward
                # Use raw logits for BCEWithLogitsLoss (numerically stable).
                # We bypass the sigmoid in LSTMModel by accessing the FC layer
                # directly during training, but since BCEWithLogitsLoss expects
                # logits, we need to avoid double-sigmoiding.
                # Solution: swap the loss criterion to BCELoss and keep sigmoid
                # output, which is simpler and explicit.
                proba = self.net(X_batch)   # sigmoid output from LSTMModel
                # BCEWithLogitsLoss expects logits, but we have sigmoid output.
                # Clamp probabilities to avoid log(0) in manual BCE:
                proba_clamped = proba.clamp(min=1e-7, max=1 - 1e-7)
                loss = criterion(torch.logit(proba_clamped), y_batch)

                if training and optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping to mitigate exploding gradients
                    nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_label: str = "test",
    ) -> Dict[str, float]:
        y_proba = self.predict_proba(X)
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, y_proba)),
        }

        logger.info(
            "[LSTM | %s] accuracy=%.4f  precision=%.4f  recall=%.4f  "
            "f1=%.4f  roc_auc=%.4f",
            dataset_label,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"],
        )
        return metrics

    def _assert_fitted(self) -> None:
        if self.net is None:
            raise RuntimeError(
                "LSTMTrainer has not been trained yet.  "
                "Call train() or load() first."
            )

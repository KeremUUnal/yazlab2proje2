import numpy as np
import torch
import torch.nn as nn

from src.config import Config
from src.models._train_utils import fit_model
from src.models.base import BaseModel


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, units: list, dropout: float):
        super().__init__()
        self.lstms = nn.ModuleList()
        self.drops = nn.ModuleList()
        prev = input_size
        for u in units:
            self.lstms.append(nn.LSTM(prev, u, batch_first=True))
            self.drops.append(nn.Dropout(dropout))
            prev = u
        self.fc = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lstm, drop in zip(self.lstms, self.drops):
            x, _ = lstm(x)
            x = drop(x)
        return torch.sigmoid(self.fc(x[:, -1, :]))


class LSTMModel(BaseModel):
    def __init__(self, config: Config):
        self.config = config
        self.net: _LSTMNet = None

    def build(self, input_shape: tuple) -> None:
        _, n_features = input_shape
        cfg = self.config.model.lstm
        self.net = _LSTMNet(n_features, cfg.units, cfg.dropout)

    def fit(self, X_train, y_train, X_val, y_val) -> dict:
        cfg = self.config.model
        return fit_model(
            self.net, X_train, y_train, X_val, y_val,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            patience=cfg.early_stopping.patience,
            model_label="LSTM",
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(X, dtype=torch.float32))
        return out.numpy().flatten()

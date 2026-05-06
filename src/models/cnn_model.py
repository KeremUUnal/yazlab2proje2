import numpy as np
import torch
import torch.nn as nn

from src.config import Config
from src.models._train_utils import fit_model
from src.models.base import BaseModel


class _CNNNet(nn.Module):
    def __init__(self, input_size: int, filters: list, kernel_size: int, dropout: float):
        super().__init__()
        layers = []
        in_ch = input_size
        for f in filters:
            layers += [
                nn.Conv1d(in_ch, f, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.MaxPool1d(2, padding=0),
            ]
            in_ch = f
        self.conv = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)   # (batch, seq, feat) → (batch, feat, seq)
        x = self.conv(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))


class CNNModel(BaseModel):
    def __init__(self, config: Config):
        self.config = config
        self.net: _CNNNet = None

    def build(self, input_shape: tuple) -> None:
        _, n_features = input_shape
        cfg = self.config.model.cnn
        self.net = _CNNNet(n_features, cfg.filters, cfg.kernel_size, cfg.dropout)

    def fit(self, X_train, y_train, X_val, y_val) -> dict:
        cfg = self.config.model
        return fit_model(
            self.net, X_train, y_train, X_val, y_val,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            patience=cfg.early_stopping.patience,
            model_label="1D-CNN",
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(X, dtype=torch.float32))
        return out.numpy().flatten()

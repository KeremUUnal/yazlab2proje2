"""Ortak PyTorch eğitim döngüsü — LSTM, GRU ve CNN paylaşır."""
import copy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def fit_model(
    net: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    patience: int,
    model_label: str = "",
) -> Dict:
    optimizer = torch.optim.Adam(net.parameters())

    # Sınıf dengesizliğini gider — anomali az olduğunda daha fazla ağırlık ver
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_w = float(n_neg / max(n_pos, 1))

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    best_weights = None
    patience_cnt = 0
    history: Dict = {"train_loss": [], "val_loss": [], "val_f1": []}

    pbar = tqdm(range(epochs), desc=model_label, unit="epoch", leave=True)
    for epoch in pbar:
        net.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = net(xb)
            # Her örneğe sınıf ağırlığı uygula
            w = torch.where(yb == 1, torch.tensor(pos_w), torch.ones_like(yb))
            loss = F.binary_cross_entropy(pred, yb, weight=w)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_tr)

        net.eval()
        with torch.no_grad():
            val_out = net(X_v)
            w_val = torch.where(y_v == 1, torch.tensor(pos_w), torch.ones_like(y_v))
            val_loss = F.binary_cross_entropy(val_out, y_v, weight=w_val).item()
            val_pred = (val_out.numpy().flatten() > 0.5).astype(int)
            val_f1 = f1_score(y_val, val_pred, zero_division=0)

        history["train_loss"].append(round(epoch_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_f1"].append(round(val_f1, 4))

        pbar.set_postfix(
            train_loss=f"{epoch_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_f1=f"{val_f1:.4f}",
            patience=f"{patience_cnt}/{patience}",
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(net.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                pbar.set_description(f"{model_label} [erken durdurma e={epoch+1}]")
                break

    net.load_state_dict(best_weights)
    return history

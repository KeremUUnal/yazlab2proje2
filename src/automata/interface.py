"""
Kişi 2 için otomata modeli arayüzü.

Beklenen giriş formatı (src/data/preprocessor.py DataSplit'ten):
    X : np.ndarray, shape (n_samples, 1)  — PCA sonrası PC1
    y : np.ndarray, shape (n_samples,)    — binary (0: normal, 1: anomali)

Bu arayüzü implement eden sınıf Trainer ile aynı şekilde çağrılacak.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class BaseAutomataModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """SAX sözlüğünü ve geçiş olasılıklarını sadece train verisiyle oluştur."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Tahmin döndür: 0 = normal, 1 = anomali."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Her örnek için anomali olasılığı döndür (path probability)."""

    @abstractmethod
    def explain(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Her zaman adımı için JSON formatında açıklama döndür.
        Zorunlu alanlar:
          time_step, state, pattern, status (seen/unseen),
          mapped_to (unseen ise), transitions, probability, decision
        """

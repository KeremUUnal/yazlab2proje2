import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from dataclasses import dataclass
from typing import Optional, List, Tuple
from src.config import Config


@dataclass
class DataSplit:
    """
    Kişi 1 ve Kişi 2 arasındaki ortak veri arayüzü.

    DL modelleri için: X shape (n, n_features) — PCA uygulanmaz
    Otomata modeli için: X shape (n, 1) — PCA ile PC1
    y alanları: (n_samples,) — binary (0: normal, 1: anomali)
    groups: SKAB GroupKFold için source_file değerleri
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    groups: Optional[np.ndarray] = None


class SKABPreprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=config.preprocessing.pca.n_components)

    def get_features_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        cfg = self.config.data.skab
        target = df[cfg.target_col].values
        groups = df["source_file"].values
        X = df.drop(columns=cfg.exclude_cols + [cfg.target_col], errors="ignore")
        return X, target, groups

    def fit_transform(self, X: pd.DataFrame, use_pca: bool = True) -> np.ndarray:
        arr = X.values.astype(float)
        if self.config.preprocessing.handle_missing:
            arr = _fill_missing(arr)
        if self.config.preprocessing.normalize:
            arr = self.scaler.fit_transform(arr)
        if use_pca and self.config.preprocessing.pca.enabled:
            arr = self.pca.fit_transform(arr)
        return arr

    def transform(self, X: pd.DataFrame, use_pca: bool = True) -> np.ndarray:
        arr = X.values.astype(float)
        if self.config.preprocessing.handle_missing:
            arr = _fill_missing(arr)
        if self.config.preprocessing.normalize:
            arr = self.scaler.transform(arr)
        if use_pca and self.config.preprocessing.pca.enabled:
            arr = self.pca.transform(arr)
        return arr

    def get_cv_splits(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        cfg = self.config.evaluation.skab
        if cfg.method == "StratifiedGroupKFold":
            cv = StratifiedGroupKFold(n_splits=cfg.n_splits)
        else:
            cv = GroupKFold(n_splits=cfg.n_splits)
        return list(cv.split(X, y, groups))


class BATADALPreprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=config.preprocessing.pca.n_components)

    def get_features_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        cfg = self.config.data.batadal
        exclude = cfg.time_cols + [cfg.label_col]
        X = df.drop(columns=exclude, errors="ignore")
        y = df[cfg.label_col].values
        return X, y

    def split(self, df: pd.DataFrame, use_pca: bool = True) -> DataSplit:
        """
        use_pca=False → DL modelleri için tüm özellikler (43 boyut)
        use_pca=True  → Otomata modeli için PC1 (1 boyut)
        Scaler her iki durumda da sadece train verisiyle fit edilir.
        """
        cfg = self.config.data.batadal
        X, y = self.get_features_target(df)
        n = len(df)
        train_end = int(n * cfg.train_ratio)
        val_end = train_end + int(n * cfg.val_ratio)

        X_train_raw = X.iloc[:train_end]
        X_val_raw = X.iloc[train_end:val_end]
        X_test_raw = X.iloc[val_end:]

        X_train = self._fit_transform(X_train_raw, use_pca=use_pca)
        X_val = self._transform(X_val_raw, use_pca=use_pca)
        X_test = self._transform(X_test_raw, use_pca=use_pca)

        return DataSplit(
            X_train=X_train,
            y_train=y[:train_end],
            X_val=X_val,
            y_val=y[train_end:val_end],
            X_test=X_test,
            y_test=y[val_end:],
        )

    def _fit_transform(self, X: pd.DataFrame, use_pca: bool = True) -> np.ndarray:
        arr = X.values.astype(float)
        if self.config.preprocessing.handle_missing:
            arr = _fill_missing(arr)
        if self.config.preprocessing.normalize:
            arr = self.scaler.fit_transform(arr)
        if use_pca and self.config.preprocessing.pca.enabled:
            arr = self.pca.fit_transform(arr)
        return arr

    def _transform(self, X: pd.DataFrame, use_pca: bool = True) -> np.ndarray:
        arr = X.values.astype(float)
        if self.config.preprocessing.handle_missing:
            arr = _fill_missing(arr)
        if self.config.preprocessing.normalize:
            arr = self.scaler.transform(arr)
        if use_pca and self.config.preprocessing.pca.enabled:
            arr = self.pca.transform(arr)
        return arr


def _fill_missing(arr: np.ndarray) -> np.ndarray:
    col_means = np.nanmean(arr, axis=0)
    nan_idx = np.where(np.isnan(arr))
    arr[nan_idx] = np.take(col_means, nan_idx[1])
    return arr

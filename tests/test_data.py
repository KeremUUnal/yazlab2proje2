import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.data.preprocessor import BATADALPreprocessor, SKABPreprocessor


@pytest.fixture
def config():
    return Config.from_yaml("config/config.yaml")


def _make_batadal_df(config: Config, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            config.data.batadal.time_cols[0]: range(n),
            "F1": rng.standard_normal(n),
            "F2": rng.standard_normal(n),
            config.data.batadal.label_col: rng.integers(0, 2, n),
        }
    )


def test_batadal_split_shapes(config):
    df = _make_batadal_df(config, n=1000)
    prep = BATADALPreprocessor(config)
    split = prep.split(df)

    total = len(split.X_train) + len(split.X_val) + len(split.X_test)
    assert total == 1000
    # PCA bileşen sayısı özellik sayısıyla sınırlandırılır
    n_features = 2  # mock veri: F1, F2
    expected_components = min(config.preprocessing.pca.n_components, n_features)
    assert split.X_train.shape[1] == expected_components
    assert split.X_val.shape[1] == expected_components
    assert split.X_test.shape[1] == expected_components


def test_batadal_split_ratios(config):
    n = 1000
    df = _make_batadal_df(config, n=n)
    prep = BATADALPreprocessor(config)
    split = prep.split(df)

    assert len(split.X_train) == int(n * config.data.batadal.train_ratio)
    assert len(split.X_val) == int(n * config.data.batadal.val_ratio)
    # test kümesi kalan kayıtlar
    expected_test = n - int(n * 0.6) - int(n * 0.2)
    assert len(split.X_test) == expected_test


def test_no_data_leakage_scaler(config):
    """Scaler sadece train üzerinde fit edilmeli."""
    n = 1000
    df = _make_batadal_df(config, n=n)
    prep = BATADALPreprocessor(config)
    prep.split(df)
    expected_train_n = int(n * config.data.batadal.train_ratio)
    assert prep.scaler.n_samples_seen_ == expected_train_n


def test_no_data_leakage_pca(config):
    """PCA sadece train üzerinde fit edilmeli."""
    n = 1000
    df = _make_batadal_df(config, n=n)
    prep = BATADALPreprocessor(config)
    prep.split(df)
    expected_train_n = int(n * config.data.batadal.train_ratio)
    assert prep.pca.n_samples_ == expected_train_n


def test_missing_value_handling(config):
    n = 200
    df = _make_batadal_df(config, n=n)
    # %5 eksik veri ekle
    df.iloc[::20, 1] = np.nan
    prep = BATADALPreprocessor(config)
    split = prep.split(df)
    assert not np.any(np.isnan(split.X_train))
    assert not np.any(np.isnan(split.X_test))

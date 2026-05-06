import numpy as np
from src.config import Config


def add_gaussian_noise(X: np.ndarray, config: Config, seed: int = 42) -> np.ndarray:
    """Gaussian gürültü ekler. std config'den okunur, veri sızıntısı olmaz."""
    rng = np.random.default_rng(seed)
    std = config.experiment.noise.gaussian_std
    return X + rng.normal(0.0, std, X.shape)

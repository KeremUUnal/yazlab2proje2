"""
PAA (Piecewise Aggregate Approximation)

Zaman serisini window_size uzunlugundaki pencerelere bolup
her pencerenin ortalamasini alir. Veriyi sadeletirir.

Ornek:
    [0.12, 0.15, 0.11, 0.14, 0.87, 0.91, 0.85, 0.89] -> window_size=4
    -> [0.13, 0.88]
"""
import numpy as np


def apply_paa(series: np.ndarray, window_size: int) -> np.ndarray:
    """
    1-boyutlu zaman serisine PAA uygular.

    Args:
        series: 1D numpy array (n_samples,)
        window_size: Her pencerenin uzunlugu

    Returns:
        PAA sonrasi kisaltilmis seri (n_windows,)
    """
    if window_size < 1:
        raise ValueError(f"window_size en az 1 olmali, verilen: {window_size}")

    n = len(series)
    if n == 0:
        return np.array([])

    # Tam pencerelerin sayisi
    n_windows = n // window_size

    if n_windows == 0:
        # Veri pencereden kisa ise tek bir ortalama dondur
        return np.array([np.mean(series)])

    # Tam pencerelere denk gelen kismi kes
    trimmed = series[: n_windows * window_size]
    # (n_windows, window_size) seklinde yeniden boyutlandir ve satirlarin ortalamasini al
    reshaped = trimmed.reshape(n_windows, window_size)
    paa_values = reshaped.mean(axis=1)

    return paa_values

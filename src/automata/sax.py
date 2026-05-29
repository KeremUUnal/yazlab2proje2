"""
SAX (Symbolic Aggregate approXimation)

PAA ciktisini sembolik harflere donusturur.
Normal dagilim breakpoint'lerini kullanarak veriyi
alphabet_size sayida bolgede (a, b, c, ...) kategorize eder.

Ornek (alphabet_size=3):
    Dusuk degerler  -> 'a'
    Orta degerler   -> 'b'
    Yuksek degerler -> 'c'
"""
import numpy as np
from scipy import stats
from typing import List


def compute_breakpoints(alphabet_size: int) -> np.ndarray:
    """
    Normal dagilim breakpoint'lerini hesaplar.
    alphabet_size bolge icin (alphabet_size - 1) breakpoint gerekir.

    Ornek: alphabet_size=3 -> breakpoints = [-0.43, 0.43] (yaklasik)
    """
    if alphabet_size < 2:
        raise ValueError(f"alphabet_size en az 2 olmali, verilen: {alphabet_size}")

    breakpoints = np.array([
        stats.norm.ppf(i / alphabet_size)
        for i in range(1, alphabet_size)
    ])
    return breakpoints


def values_to_symbols(z_values: np.ndarray, breakpoints: np.ndarray) -> List[str]:
    """
    Z-normalize edilmis degerleri sembol harflere donusturur.

    Args:
        z_values: Z-normalize edilmis PAA degerleri
        breakpoints: Normal dagilim breakpoint'leri

    Returns:
        Her deger icin bir harf listesi ['a', 'b', 'c', ...]
    """
    symbols = []
    alphabet = [chr(ord('a') + i) for i in range(len(breakpoints) + 1)]

    for val in z_values:
        idx = np.searchsorted(breakpoints, val)
        symbols.append(alphabet[idx])

    return symbols


class SAXTransformer:
    """
    SAX donusturucusu. fit() sadece train verisiyle cagrilmali (data leakage onleme).

    Kullanim:
        sax = SAXTransformer(alphabet_size=3)
        sax.fit(train_paa_values)
        symbols_train = sax.transform(train_paa_values)
        symbols_test  = sax.transform(test_paa_values)
    """

    def __init__(self, alphabet_size: int):
        self.alphabet_size = alphabet_size
        self.breakpoints = compute_breakpoints(alphabet_size)
        self.mean_ = None
        self.std_ = None
        self._is_fitted = False

    def fit(self, paa_values: np.ndarray) -> "SAXTransformer":
        """
        Train verisinden ortalama ve std ogrenilir (Z-normalizasyon icin).
        """
        self.mean_ = np.mean(paa_values)
        self.std_ = np.std(paa_values)
        if self.std_ == 0:
            self.std_ = 1.0  # Sabit seri icin bolme hatasini onle
        self._is_fitted = True
        return self

    def transform(self, paa_values: np.ndarray) -> List[str]:
        """
        PAA degerlerini Z-normalize edip sembol harflere donusturur.
        """
        if not self._is_fitted:
            raise RuntimeError("Once fit() cagrilmali")

        z_values = (paa_values - self.mean_) / self.std_
        return values_to_symbols(z_values, self.breakpoints)

    def fit_transform(self, paa_values: np.ndarray) -> List[str]:
        """fit + transform tek adimda."""
        self.fit(paa_values)
        return self.transform(paa_values)

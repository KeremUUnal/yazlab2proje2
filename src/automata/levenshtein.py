"""
Levenshtein (Edit Distance) ve Unseen Pattern Yonetimi

Test sirasinda SAX sozlugunde bulunmayan pattern'lar icin
en yakin bilinen pattern'i bulur.

Ornek:
    bilinen = {"abc", "aab", "bcc"}
    unseen  = "adc"
    -> en yakin: "abc" (mesafe=1)
"""
from typing import Dict, List, Optional, Tuple


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Iki string arasindaki Levenshtein (edit) mesafesini hesaplar.
    Dinamik programlama (Wagner-Fischer) algoritmasi kullanir.

    Args:
        s1: Birinci string
        s2: Ikinci string

    Returns:
        Minimum ekleme/silme/degistirme sayisi
    """
    m, n = len(s1), len(s2)

    # DP tablosu
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Bos string'e donusum maliyetleri
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Tabloyu doldur
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # silme
                    dp[i][j - 1],      # ekleme
                    dp[i - 1][j - 1],  # degistirme
                )

    return dp[m][n]


def find_nearest_pattern(
    unseen_pattern: str,
    known_patterns: List[str],
) -> Tuple[str, int]:
    """
    Unseen bir pattern icin bilinen pattern'lar arasindaki en yakin olani bulur.

    Args:
        unseen_pattern: SAX sozlugunde bulunmayan pattern
        known_patterns: Bilinen (train'den ogrenmis) pattern listesi

    Returns:
        (en_yakin_pattern, mesafe) tuple'i

    Raises:
        ValueError: Bilinen pattern listesi bos ise
    """
    if not known_patterns:
        raise ValueError("Bilinen pattern listesi bos olamaz")

    best_pattern = known_patterns[0]
    best_distance = levenshtein_distance(unseen_pattern, best_pattern)

    for pattern in known_patterns[1:]:
        dist = levenshtein_distance(unseen_pattern, pattern)
        if dist < best_distance:
            best_distance = dist
            best_pattern = pattern
        if best_distance == 0:
            break  # Tam eslesme bulundu

    return best_pattern, best_distance


class UnseenPatternHandler:
    """
    Unseen pattern yonetim sinifi.
    Train'den ogrenmis SAX sozlugunu tutar ve
    test sirasinda bilinmeyen pattern'lari esler.
    """

    def __init__(self):
        self.known_patterns: List[str] = []
        self._mapping_cache: Dict[str, Tuple[str, int]] = {}

    def fit(self, patterns: List[str]) -> "UnseenPatternHandler":
        """Bilinen pattern sozlugunu olusturur (sadece train'den)."""
        self.known_patterns = list(set(patterns))
        self._mapping_cache = {}
        return self

    def is_unseen(self, pattern: str) -> bool:
        """Pattern'in sozlukte olup olmadigini kontrol eder."""
        return pattern not in self.known_patterns

    def resolve(self, pattern: str) -> Tuple[str, int, bool]:
        """
        Pattern'i cozumler.

        Returns:
            (cozumlenmis_pattern, mesafe, unseen_mi) tuple'i
            - Bilinen pattern ise: (pattern, 0, False)
            - Unseen ise: (en_yakin_pattern, mesafe, True)
        """
        if not self.is_unseen(pattern):
            return pattern, 0, False

        # Cache'e bak
        if pattern in self._mapping_cache:
            nearest, dist = self._mapping_cache[pattern]
            return nearest, dist, True

        # En yakin pattern'i bul ve cache'le
        nearest, dist = find_nearest_pattern(pattern, self.known_patterns)
        self._mapping_cache[pattern] = (nearest, dist)
        return nearest, dist, True

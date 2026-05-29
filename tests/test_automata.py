"""
Otomata modulu birim testleri.

Zorunlu: Levenshtein + Unseen mekanizmasi testleri (proje sarti)
Ek: PAA, SAX ve tam pipeline testleri.
"""
import numpy as np
import pytest

from src.automata.paa import apply_paa
from src.automata.sax import SAXTransformer, compute_breakpoints, values_to_symbols
from src.automata.levenshtein import (
    levenshtein_distance,
    find_nearest_pattern,
    UnseenPatternHandler,
)


# ======================================================================
# PAA Testleri
# ======================================================================
class TestPAA:
    def test_basic_averaging(self):
        """4 elemanli seri, window=2 -> 2 ortalama."""
        series = np.array([1.0, 3.0, 5.0, 7.0])
        result = apply_paa(series, window_size=2)
        np.testing.assert_array_almost_equal(result, [2.0, 6.0])

    def test_window_size_equals_length(self):
        """Seri uzunlugu = window -> tek bir ortalama."""
        series = np.array([2.0, 4.0, 6.0, 8.0])
        result = apply_paa(series, window_size=4)
        np.testing.assert_array_almost_equal(result, [5.0])

    def test_remainder_trimmed(self):
        """5 eleman, window=2 -> 2 pencere, son eleman kesilir."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_paa(series, window_size=2)
        np.testing.assert_array_almost_equal(result, [1.5, 3.5])

    def test_empty_series(self):
        """Bos seri -> bos sonuc."""
        result = apply_paa(np.array([]), window_size=4)
        assert len(result) == 0

    def test_series_shorter_than_window(self):
        """Seri pencereden kisa -> tek ortalama."""
        series = np.array([3.0, 5.0])
        result = apply_paa(series, window_size=4)
        np.testing.assert_array_almost_equal(result, [4.0])

    def test_invalid_window_size(self):
        """window_size < 1 -> hata."""
        with pytest.raises(ValueError):
            apply_paa(np.array([1.0, 2.0]), window_size=0)


# ======================================================================
# SAX Testleri
# ======================================================================
class TestSAX:
    def test_breakpoints_3(self):
        """alphabet=3 -> 2 breakpoint, simetrik."""
        bp = compute_breakpoints(3)
        assert len(bp) == 2
        assert bp[0] < 0 < bp[1]
        np.testing.assert_almost_equal(bp[0], -bp[1], decimal=5)

    def test_breakpoints_2(self):
        """alphabet=2 -> 1 breakpoint (0'da)."""
        bp = compute_breakpoints(2)
        assert len(bp) == 1
        np.testing.assert_almost_equal(bp[0], 0.0, decimal=5)

    def test_invalid_alphabet(self):
        """alphabet < 2 -> hata."""
        with pytest.raises(ValueError):
            compute_breakpoints(1)

    def test_transform_basic(self):
        """Dusuk/orta/yuksek degerler -> a/b/c."""
        sax = SAXTransformer(alphabet_size=3)
        # Z-normalize edildiginde: -1.22, 0, 1.22 yaklasik
        data = np.array([1.0, 5.0, 9.0])
        symbols = sax.fit_transform(data)
        assert symbols[0] == 'a'  # en dusuk
        assert symbols[2] == 'c'  # en yuksek

    def test_fit_then_transform(self):
        """fit ve transform ayri cagrildiginda ayni sonuc."""
        sax = SAXTransformer(alphabet_size=3)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s1 = sax.fit_transform(data)
        s2 = sax.transform(data)
        assert s1 == s2

    def test_transform_without_fit(self):
        """fit() olmadan transform() -> hata."""
        sax = SAXTransformer(alphabet_size=3)
        with pytest.raises(RuntimeError):
            sax.transform(np.array([1.0, 2.0]))

    def test_constant_series(self):
        """Sabit seri -> tum semboller ayni (std=0 durumu)."""
        sax = SAXTransformer(alphabet_size=3)
        data = np.array([5.0, 5.0, 5.0, 5.0])
        symbols = sax.fit_transform(data)
        # std=0 durumunda z-score=0 -> hepsi ortanca sembol
        assert len(set(symbols)) == 1  # hepsi ayni


# ======================================================================
# Levenshtein Testleri (ZORUNLU - proje sarti)
# ======================================================================
class TestLevenshtein:
    def test_identical_strings(self):
        """Ayni string -> mesafe 0."""
        assert levenshtein_distance("abc", "abc") == 0

    def test_single_substitution(self):
        """Tek harf degisiklik -> mesafe 1."""
        assert levenshtein_distance("abc", "adc") == 1

    def test_single_insertion(self):
        """Tek ekleme -> mesafe 1."""
        assert levenshtein_distance("abc", "abcd") == 1

    def test_single_deletion(self):
        """Tek silme -> mesafe 1."""
        assert levenshtein_distance("abcd", "abc") == 1

    def test_completely_different(self):
        """Tamamen farkli -> mesafe = uzun stringin uzunlugu."""
        assert levenshtein_distance("abc", "xyz") == 3

    def test_empty_strings(self):
        """Bos string'ler."""
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "abc") == 3

    def test_symmetric(self):
        """Mesafe simetrik: d(a,b) == d(b,a)."""
        assert levenshtein_distance("abc", "adc") == levenshtein_distance("adc", "abc")
        assert levenshtein_distance("kitten", "sitting") == levenshtein_distance("sitting", "kitten")

    def test_known_example(self):
        """Bilinen ornek: kitten -> sitting = 3."""
        assert levenshtein_distance("kitten", "sitting") == 3


class TestFindNearestPattern:
    def test_exact_match(self):
        """Bilinen pattern -> mesafe 0."""
        nearest, dist = find_nearest_pattern("abc", ["abc", "bcd", "cde"])
        assert nearest == "abc"
        assert dist == 0

    def test_closest_pattern(self):
        """En yakin pattern bulunur."""
        nearest, dist = find_nearest_pattern("adc", ["abc", "xyz", "bbb"])
        assert nearest == "abc"
        assert dist == 1

    def test_empty_known_list(self):
        """Bos liste -> hata."""
        with pytest.raises(ValueError):
            find_nearest_pattern("abc", [])

    def test_multiple_equidistant(self):
        """Esit mesafeli pattern'lardan biri secilir (deterministik)."""
        nearest, dist = find_nearest_pattern("aaa", ["baa", "aba"])
        assert dist == 1
        assert nearest in ["baa", "aba"]


class TestUnseenPatternHandler:
    def test_seen_pattern(self):
        """Bilinen pattern -> unseen degil."""
        handler = UnseenPatternHandler()
        handler.fit(["abc", "bcd", "cde"])

        assert not handler.is_unseen("abc")
        resolved, dist, is_unseen = handler.resolve("abc")
        assert resolved == "abc"
        assert dist == 0
        assert is_unseen is False

    def test_unseen_pattern(self):
        """Bilinmeyen pattern -> en yakin eslestirme."""
        handler = UnseenPatternHandler()
        handler.fit(["abc", "bcd", "cde"])

        assert handler.is_unseen("adc")
        resolved, dist, is_unseen = handler.resolve("adc")
        assert resolved == "abc"
        assert dist == 1
        assert is_unseen is True

    def test_cache_works(self):
        """Ayni unseen pattern ikinci kez sorulunca cache'ten gelir."""
        handler = UnseenPatternHandler()
        handler.fit(["abc", "bcd"])

        r1 = handler.resolve("adc")
        r2 = handler.resolve("adc")
        assert r1 == r2

    def test_deduplicated_patterns(self):
        """Tekrarli pattern'lar deduplicate edilir."""
        handler = UnseenPatternHandler()
        handler.fit(["abc", "abc", "abc", "bcd"])
        assert len(handler.known_patterns) == 2


# ======================================================================
# Entegrasyon Testi — Tam Pipeline
# ======================================================================
class TestFullPipeline:
    def test_paa_then_sax(self):
        """PAA -> SAX pipeline'i calisir."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        paa = apply_paa(series, window_size=2)
        assert len(paa) == 4

        sax = SAXTransformer(alphabet_size=3)
        symbols = sax.fit_transform(paa)
        assert len(symbols) == 4
        assert all(s in "abc" for s in symbols)

    def test_unseen_in_pipeline(self):
        """
        Train'de gorulmeyen pattern test'te dogru eslenir.
        Tam unseen akisi: SAX -> pattern -> Levenshtein -> resolve.
        """
        handler = UnseenPatternHandler()
        train_patterns = ["aab", "abc", "bcc", "cab"]
        handler.fit(train_patterns)

        # Test'te yeni pattern
        test_pattern = "adc"
        resolved, dist, is_unseen = handler.resolve(test_pattern)

        assert is_unseen is True
        assert dist > 0
        assert resolved in train_patterns

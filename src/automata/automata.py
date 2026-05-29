"""
Probabilistic Automata — Ana Model

Akis:
    Ham seri -> PAA -> SAX -> Sliding Window -> Pattern (State)
    State gecisleri sayilir -> Gecis olasiliklari hesaplanir
    Dusuk olasiklikli path -> Anomali

Kullanim:
    model = ProbabilisticAutomata(cfg)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    explanations = model.explain(X_test)
"""
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from src.automata.paa import apply_paa
from src.automata.sax import SAXTransformer
from src.automata.levenshtein import UnseenPatternHandler
from src.automata.interface import BaseAutomataModel
from src.config import AutomataConfig


class ProbabilisticAutomata(BaseAutomataModel):
    """
    Olasiliksal Otomata Modeli.

    Tum parametreler config'den okunur — hard-coded deger yok.
    """

    def __init__(self, config: AutomataConfig):
        self.config = config
        self.window_size = config.window_size
        self.alphabet_size = config.alphabet_size

        # Alt bilesenler
        self.sax = SAXTransformer(self.alphabet_size)
        self.unseen_handler = UnseenPatternHandler()

        # Ogrenilen yapilar
        self.transition_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.transition_probs: Dict[str, Dict[str, float]] = {}
        self.state_labels: Dict[str, Dict[str, int]] = {}  # state -> {0: count, 1: count}
        self.anomaly_threshold: float = 0.5

        # Laplace smoothing icin
        self._all_states: List[str] = []
        self._smoothing_alpha: float = 1e-6

        self._is_fitted = False

    # ------------------------------------------------------------------
    # Temel donusum pipeline'i
    # ------------------------------------------------------------------
    def _series_to_paa(self, X: np.ndarray) -> np.ndarray:
        """(n_samples, 1) -> flatten -> PAA."""
        series = X.flatten()
        from src.automata.paa import apply_paa
        return apply_paa(series, self.window_size)

    def _paa_to_symbols(self, paa_values: np.ndarray, fit: bool = False) -> List[str]:
        """PAA -> SAX sembol dizisi."""
        if fit:
            return self.sax.fit_transform(paa_values)
        return self.sax.transform(paa_values)

    def _symbols_to_patterns(self, symbols: List[str]) -> List[str]:
        """Sembol dizisi -> sliding window pattern'lari."""
        patterns = []
        for i in range(len(symbols) - self.window_size + 1):
            pattern = "".join(symbols[i: i + self.window_size])
            patterns.append(pattern)
        return patterns

    def _full_pipeline(self, X: np.ndarray, fit: bool = False) -> List[str]:
        """X -> PAA -> SAX -> Pattern listesi (tam pipeline)."""
        paa_values = self._series_to_paa(X)
        symbols = self._paa_to_symbols(paa_values, fit=fit)
        patterns = self._symbols_to_patterns(symbols)
        return patterns

    # ------------------------------------------------------------------
    # Gecis olasiliklari
    # ------------------------------------------------------------------
    def _pattern_to_original_indices(self, pattern_idx: int, n_samples: int) -> tuple:
        """
        Pattern indeksini orijinal seri indekslerine esler.

        Pattern[i] = symbols[i : i+ws] kullanir.
        Symbol[j] = PAA[j] = mean(original[j*ws : (j+1)*ws])
        Dolayisiyla pattern[i] orijinal seride [i*ws, (i+ws)*ws) araligina denk gelir.
        Ama stride olarak her pattern bir PAA adimi (ws sample) kayar.
        """
        ws = self.window_size
        start = pattern_idx * ws
        end = min(start + ws * ws, n_samples)
        return start, end

    def _build_transitions(self, patterns: List[str], labels: np.ndarray) -> None:
        """
        Ardasik pattern'lardan gecis sayilarini ve state etiketlerini olusturur.
        """
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.state_labels = defaultdict(lambda: defaultdict(int))

        n_samples = len(labels)

        # Her pattern'a denk gelen etiketleri topla
        for i, pattern in enumerate(patterns):
            start, end = self._pattern_to_original_indices(i, n_samples)
            if start < end:
                segment = labels[start:end]
                anomaly_count = int(np.sum(segment == 1))
                normal_count = len(segment) - anomaly_count
                self.state_labels[pattern][0] = self.state_labels[pattern].get(0, 0) + normal_count
                self.state_labels[pattern][1] = self.state_labels[pattern].get(1, 0) + anomaly_count

        # Gecis sayilarini topla
        for i in range(len(patterns) - 1):
            src = patterns[i]
            dst = patterns[i + 1]
            self.transition_counts[src][dst] += 1

        # Gecis olasiliklarini hesapla (Laplace smoothing ile)
        self._all_states = list(set(patterns))
        self.transition_probs = {}

        for src, destinations in self.transition_counts.items():
            total = sum(destinations.values())
            n_destinations = len(destinations)
            self.transition_probs[src] = {}
            for dst, count in destinations.items():
                self.transition_probs[src][dst] = (
                    (count + self._smoothing_alpha)
                    / (total + self._smoothing_alpha * n_destinations)
                )

        # State anomali oranlari (state-level anomaly scoring)
        self.state_anomaly_rate: Dict[str, float] = {}
        for state, counts in self.state_labels.items():
            total = counts.get(0, 0) + counts.get(1, 0)
            self.state_anomaly_rate[state] = counts.get(1, 0) / total if total > 0 else 0.0

    def _get_transition_prob(self, src: str, dst: str) -> float:
        """Iki state arasi gecis olasiliginii dondurur (unseen cozumleme dahil)."""
        src_r, _, _ = self.unseen_handler.resolve(src)
        dst_r, _, _ = self.unseen_handler.resolve(dst)
        if src_r in self.transition_probs:
            return self.transition_probs[src_r].get(dst_r, self._smoothing_alpha)
        return self._smoothing_alpha

    def _compute_anomaly_scores(self, patterns: List[str]) -> List[float]:
        """
        Her pattern icin anomali skoru hesaplar.

        Iki sinyal birlestirilir:
        1. State anomali orani: egitimde bu state ne kadar anomali ile iliskiliydi
        2. Path beklenmediklik: gecis olasiligi ne kadar dusuk

        Skor [0, 1] araliginda: 0 = kesinlikle normal, 1 = kesinlikle anomali.
        """
        if len(patterns) < 2:
            return [0.5] * len(patterns)

        scores = []
        for i in range(len(patterns)):
            resolved, _, _ = self.unseen_handler.resolve(patterns[i])

            # Sinyal 1: State anomali orani (egitimden)
            state_score = self.state_anomaly_rate.get(resolved, 0.5)

            # Sinyal 2: Gecis beklenmedikligi
            if i < len(patterns) - 1:
                trans_prob = self._get_transition_prob(patterns[i], patterns[i + 1])
                # Log-olasiligin negatifini normalize et
                # trans_prob [0,1] -> unexpectedness [0,1]
                unexpectedness = 1.0 - trans_prob
            else:
                unexpectedness = 0.5

            # Birlesik skor
            combined = 0.6 * state_score + 0.4 * unexpectedness
            scores.append(combined)

        return scores

    def _compute_anomaly_threshold(self, patterns: List[str], labels: np.ndarray) -> None:
        """
        Train verisinden F1'i maksimize eden anomali threshold'unu hesaplar.
        """
        scores = self._compute_anomaly_scores(patterns)

        if len(scores) == 0:
            self.anomaly_threshold = 0.5
            return

        # Her pattern icin cogunluk etiketi
        n_samples = len(labels)
        ws = self.window_size
        pattern_labels = []
        for i in range(len(patterns)):
            start = i * ws
            end = min(start + ws, n_samples)
            if start < end:
                pattern_labels.append(1 if np.mean(labels[start:end]) > 0.5 else 0)
            else:
                pattern_labels.append(0)

        pattern_labels = np.array(pattern_labels)
        scores_arr = np.array(scores)

        # F1'i maksimize eden threshold'u bul
        # Ek kisit: tahmin dagilimi makul olmali (hep 0 veya hep 1 olmasin)
        best_f1 = -1
        best_threshold = 0.5
        n_total = len(pattern_labels)
        anomaly_rate = np.mean(pattern_labels)

        thresholds = np.percentile(scores_arr, np.arange(1, 100, 2))
        thresholds = np.unique(thresholds)

        for t in thresholds:
            preds = (scores_arr >= t).astype(int)
            pred_rate = np.mean(preds)

            # Tahmin orani cok asiri olmamali (en az %5 her siniftan)
            if pred_rate < 0.05 or pred_rate > 0.95:
                continue

            tp = np.sum((preds == 1) & (pattern_labels == 1))
            fp = np.sum((preds == 1) & (pattern_labels == 0))
            fn = np.sum((preds == 0) & (pattern_labels == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        # Hicbir makul threshold bulunamazsa, anomali oranina gore yuzdelik sec
        if best_f1 <= 0:
            target_percentile = max(10, min(90, (1 - anomaly_rate) * 100))
            best_threshold = np.percentile(scores_arr, target_percentile)

        self.anomaly_threshold = best_threshold
        self._train_best_f1 = best_f1

    def _compute_path_probabilities(self, patterns: List[str]) -> List[float]:
        """Her ardasik pattern cifti icin gecis olasiliklarini hesaplar."""
        probs = []
        for i in range(len(patterns) - 1):
            src = patterns[i]
            dst = patterns[i + 1]

            # Unseen pattern'i cozumle
            src_resolved, _, _ = self.unseen_handler.resolve(src)
            dst_resolved, _, _ = self.unseen_handler.resolve(dst)

            if src_resolved in self.transition_probs:
                prob = self.transition_probs[src_resolved].get(
                    dst_resolved, self._smoothing_alpha
                )
            else:
                prob = self._smoothing_alpha

            probs.append(prob)
        return probs

    # ------------------------------------------------------------------
    # Public API (BaseAutomataModel interface)
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train verisiyle modeli olusturur:
        1. PAA + SAX + Pattern cikartma
        2. Gecis olasiliklarini hesaplama
        3. Unseen handler'a bilinen pattern'lari ogretme
        4. Anomali threshold'unu belirleme
        """
        patterns = self._full_pipeline(X, fit=True)

        if len(patterns) < 2:
            raise ValueError(
                f"Yeterli pattern uretilemedi (n={len(patterns)}). "
                f"Veri boyutu veya window_size'i kontrol edin."
            )

        # Bilinen pattern'lari kaydet
        self.unseen_handler.fit(patterns)

        # Gecis olasiliklarini olustur
        self._build_transitions(patterns, y)

        # Anomali threshold'unu belirle
        self._compute_anomaly_threshold(patterns, y)

        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Test verisi icin anomali tahminleri uretir.
        Returns: 0=normal, 1=anomali (n_samples,) boyutunda
        """
        if not self._is_fitted:
            raise RuntimeError("Once fit() cagrilmali")

        patterns = self._full_pipeline(X, fit=False)
        scores = self._compute_anomaly_scores(patterns)

        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)

        if len(scores) == 0:
            return predictions

        ws = self.window_size
        for i, score in enumerate(scores):
            start_idx = i * ws
            end_idx = min(start_idx + ws, n_samples)
            predictions[start_idx:end_idx] = 1 if score >= self.anomaly_threshold else 0

        # Kalan sample'lar icin son skor
        last_covered = len(scores) * ws
        if last_covered < n_samples and len(scores) > 0:
            predictions[last_covered:] = 1 if scores[-1] >= self.anomaly_threshold else 0

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Her sample icin anomali olasiligi dondurur.
        Yuksek skor = yuksek anomali olasiligi.
        """
        if not self._is_fitted:
            raise RuntimeError("Once fit() cagrilmali")

        patterns = self._full_pipeline(X, fit=False)
        scores = self._compute_anomaly_scores(patterns)

        n_samples = X.shape[0]
        anomaly_scores = np.full(n_samples, 0.5)

        if len(scores) == 0:
            return anomaly_scores

        ws = self.window_size
        for i, score in enumerate(scores):
            start_idx = i * ws
            end_idx = min(start_idx + ws, n_samples)
            anomaly_scores[start_idx:end_idx] = score

        last_covered = len(scores) * ws
        if last_covered < n_samples and len(scores) > 0:
            anomaly_scores[last_covered:] = scores[-1]

        return anomaly_scores

    def explain(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Her zaman adimi icin detayli aciklama uretir.
        Proje gereksinimi: JSON formatinda, deterministik ve tekrar uretebilir.
        """
        if not self._is_fitted:
            raise RuntimeError("Once fit() cagrilmali")

        patterns = self._full_pipeline(X, fit=False)
        scores = self._compute_anomaly_scores(patterns)
        explanations = []

        for i in range(len(patterns)):
            pattern = patterns[i]
            resolved, distance, is_unseen = self.unseen_handler.resolve(pattern)

            # State anomali orani
            state_anomaly = self.state_anomaly_rate.get(resolved, 0.5)

            # Gecis bilgisi
            transitions = {}
            trans_prob = 1.0

            if i < len(patterns) - 1:
                next_pattern = patterns[i + 1]
                next_resolved, _, _ = self.unseen_handler.resolve(next_pattern)
                trans_prob = self._get_transition_prob(patterns[i], next_pattern)
                transitions[f"{resolved}->{next_resolved}"] = round(trans_prob, 6)

            anomaly_score = scores[i]
            decision = "anomaly" if anomaly_score >= self.anomaly_threshold else "normal"

            explanation = {
                "time_step": i,
                "state": resolved,
                "pattern": pattern,
                "status": "unseen" if is_unseen else "seen",
                "mapped_to": resolved if is_unseen else None,
                "distance": distance if is_unseen else 0,
                "transitions": transitions,
                "transition_probability": round(trans_prob, 6),
                "state_anomaly_rate": round(state_anomaly, 6),
                "anomaly_score": round(anomaly_score, 6),
                "probability": round(trans_prob, 6),
                "decision": decision,
                "confidence": round(1.0 - abs(anomaly_score - self.anomaly_threshold), 6),
            }
            explanations.append(explanation)

        return explanations

    # ------------------------------------------------------------------
    # Yardimci metotlar (parametre analizi icin)
    # ------------------------------------------------------------------
    def get_state_count(self) -> int:
        """Toplam benzersiz state sayisi."""
        return len(self._all_states)

    def get_transition_density(self) -> float:
        """
        Gecis yogunlugu: gerceklesen gecis / mumkun gecis orani.
        1.0 = her state'ten her state'e gecis var (tam bagli).
        """
        n_states = len(self._all_states)
        if n_states <= 1:
            return 0.0

        actual_transitions = sum(
            len(dests) for dests in self.transition_counts.values()
        )
        possible_transitions = n_states * n_states
        return actual_transitions / possible_transitions

    def get_transition_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Gecis olasilik matrisini dondurur (gorsellestirme icin).

        Returns:
            (matrix, state_labels) tuple'i
            matrix: (n_states, n_states) numpy array
            state_labels: state isimleri listesi
        """
        states = sorted(self._all_states)
        n = len(states)
        state_idx = {s: i for i, s in enumerate(states)}
        matrix = np.zeros((n, n))

        for src, dests in self.transition_probs.items():
            if src in state_idx:
                for dst, prob in dests.items():
                    if dst in state_idx:
                        matrix[state_idx[src]][state_idx[dst]] = prob

        return matrix, states

"""
Olasiliksal Aciklanabilirlik Modulu (20 puan)

Her karar icin:
- Mevcut durum (state)
- Gozlemlenen oruntu (pattern)
- Oruntunun egitim verisinde bulunup bulunmadigi
- Unseen durumunda uygulanan esleme mekanizmasi
- Gerceklesen durum gecisleri (state transitions)
- Her gecisin olasiligi
- Gozlemlenen oruntu dizisinin toplam olasiligi (path probability)
- Nihai karar ve bu kararin olasiliksal gerekcesi
"""
import json
from typing import Any, Dict, List, Optional

import numpy as np

from src.automata.automata import ProbabilisticAutomata


class ExplainabilityModule:
    """
    Otomata modelinin kararlarini aciklayan modul.
    Ciktilar deterministik, yeniden uretilebilir ve
    modelin ic hesaplamalari ile tutarlidir.
    """

    def __init__(self, model: ProbabilisticAutomata):
        self.model = model

    def explain_single(self, X: np.ndarray, sample_index: int = 0) -> Dict[str, Any]:
        """
        Tek bir tahmin icin detayli aciklama uretir.

        Args:
            X: Giris verisi (n_samples, 1)
            sample_index: Aciklanacak sample'in indeksi

        Returns:
            Detayli aciklama dict'i
        """
        explanations = self.model.explain(X)
        if sample_index < len(explanations):
            return explanations[sample_index]
        return explanations[-1] if explanations else {}

    def explain_sequence(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Tum seri icin aciklama dizisi uretir.
        Path probability'yi kumulatif olarak hesaplar.
        """
        raw_explanations = self.model.explain(X)

        # Kumulatif path probability ekle
        cumulative_prob = 1.0
        enriched = []

        for i, exp in enumerate(raw_explanations):
            if exp["transitions"]:
                step_prob = list(exp["transitions"].values())[0]
                cumulative_prob *= step_prob

            enriched_exp = {
                **exp,
                "cumulative_path_probability": round(cumulative_prob, 8),
                "steps_so_far": i + 1,
            }
            enriched.append(enriched_exp)

        return enriched

    def generate_report(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Tum tahmin sureci icin ozet rapor uretir.

        Returns:
            Anomali istatistikleri, unseen oranlari, ortalama guven skoru vb.
        """
        explanations = self.model.explain(X)
        predictions = self.model.predict(X)

        if len(explanations) == 0:
            return {"error": "Aciklama uretilemedi"}

        # Istatistikler
        n_total = len(explanations)
        n_unseen = sum(1 for e in explanations if e["status"] == "unseen")
        n_anomaly = sum(1 for e in explanations if e["decision"] == "anomaly")
        confidences = [e["confidence"] for e in explanations]

        report = {
            "summary": {
                "total_patterns": n_total,
                "unseen_patterns": n_unseen,
                "unseen_ratio": round(n_unseen / n_total, 4) if n_total > 0 else 0,
                "anomaly_decisions": n_anomaly,
                "anomaly_ratio": round(n_anomaly / n_total, 4) if n_total > 0 else 0,
                "mean_confidence": round(float(np.mean(confidences)), 4),
                "min_confidence": round(float(np.min(confidences)), 6),
                "max_confidence": round(float(np.max(confidences)), 6),
            },
            "model_info": {
                "window_size": self.model.window_size,
                "alphabet_size": self.model.alphabet_size,
                "total_states": self.model.get_state_count(),
                "transition_density": round(self.model.get_transition_density(), 4),
                "anomaly_threshold": round(self.model.anomaly_threshold, 6),
            },
            "sample_explanations": explanations[:5],  # Ilk 5 ornek
        }

        # Gercek etiket varsa dogrluk bilgisi ekle
        if y_true is not None:
            pred_labels = predictions
            n_correct = int(np.sum(pred_labels == y_true))
            report["summary"]["accuracy_on_predictions"] = round(
                n_correct / len(y_true), 4
            )

        return report

    def to_json(self, explanation: Any, indent: int = 2) -> str:
        """Aciklamayi JSON string'e cevirir."""
        return json.dumps(explanation, ensure_ascii=False, indent=indent, default=str)

    def format_decision(self, explanation: Dict[str, Any]) -> str:
        """
        Tek bir karari okunabilir metin formatinda dondurur.
        PDF'deki ornek formata uygun.
        """
        lines = [
            "[SYSTEM DECISION]",
            f"Time Step: t = {explanation.get('time_step', '?')}",
            f"Previous State: \"{explanation.get('state', '?')}\"",
            f"Incoming Pattern: \"{explanation.get('pattern', '?')}\"",
            f"Status: {explanation.get('status', '?').capitalize()}",
        ]

        if explanation.get("status") == "unseen":
            lines.append(
                f"Nearest Pattern: \"{explanation.get('mapped_to', '?')}\" "
                f"(distance = {explanation.get('distance', '?')})"
            )

        lines.append("")
        lines.append("Transitions:")
        for trans, prob in explanation.get("transitions", {}).items():
            lines.append(f"  {trans} : {prob}")

        lines.append("")
        lines.append(f"Path Probability: {explanation.get('probability', '?')}")
        lines.append("")
        lines.append("Decision:")

        prob = explanation.get("probability", 0)
        if explanation.get("decision") == "anomaly":
            lines.append("  Low probability path detected")
            lines.append("  Result: ANOMALY")
        else:
            lines.append("  Normal probability path")
            lines.append("  Result: NORMAL")

        lines.append(f"  Confidence Score: {explanation.get('confidence', '?')}")

        return "\n".join(lines)

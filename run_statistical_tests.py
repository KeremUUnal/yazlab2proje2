"""
Modeller arası istatistiksel anlamlılık testleri (McNemar testi).

McNemar testi, aynı test seti üzerinde iki sınıflandırıcının tahminlerini
örnek bazında karşılaştırır. n=5 seed ile Wilcoxon'un minimum p=0.0625
sınırlaması yoktur; binlerce test örneği üzerinde çalışır.

Kullanım:
    python run_statistical_tests.py
    python run_statistical_tests.py --batadal-dir results2 --skab-dir results_skab
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from src.evaluation.statistical import mcnemar_test

MODELS = ["LSTM", "GRU", "CNN"]
SCENARIOS = ["original", "noisy"]


def load_predictions(json_path: Path):
    """JSON dosyasından ilk seed'in y_true ve y_pred'ini döndürür."""
    if not json_path.exists():
        return None, None
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    seed0 = data.get("seed_results", [{}])[0]
    y_true = seed0.get("y_true")
    y_pred = seed0.get("y_pred")
    if y_true is None or y_pred is None:
        return None, None
    return np.array(y_true), np.array(y_pred)


def run_mcnemar_for_dataset(results_dir: Path, dataset_prefix: str, use_skab_fold: bool = False):
    print(f"\n{'='*60}")
    print(f"Veri Seti: {dataset_prefix}  (McNemar Testi, seed=42)")
    print(f"{'='*60}")

    all_results = {}

    for scenario in SCENARIOS:
        print(f"\n--- Senaryo: {scenario} ---")

        preds = {}
        y_true_ref = None

        for model in MODELS:
            if use_skab_fold:
                path = results_dir / f"SKAB_fold0_{model}_{scenario}.json"
            else:
                path = results_dir / f"{dataset_prefix}_{model}_{scenario}.json"

            y_true, y_pred = load_predictions(path)
            if y_true is None:
                print(f"  {model}: tahmin verisi yok, atlandı")
                continue

            preds[model] = y_pred
            if y_true_ref is None:
                y_true_ref = y_true

            acc = float(np.mean(y_pred == y_true))
            print(f"  {model}: n={len(y_pred)}  acc={acc:.4f}")

        if len(preds) < 2:
            print("  Karsilastirma icin yeterli model yok.")
            continue

        print(f"\n  McNemar Testleri ({scenario}):")
        scenario_results = []

        for m_a, m_b in combinations(preds.keys(), 2):
            try:
                result = mcnemar_test(y_true_ref, preds[m_a], preds[m_b])
                sig = "ANLAMLI (p<0.05)" if result["significant"] else "anlamsiz"
                ct = result["contingency_table"]
                print(
                    f"    {m_a} vs {m_b}: "
                    f"p={result['p_value']:.4f}  "
                    f"contingency=[{ct[0][0]},{ct[0][1]};{ct[1][0]},{ct[1][1]}]  "
                    f"-> {sig}"
                )
                scenario_results.append({
                    "model_a": m_a,
                    "model_b": m_b,
                    **result,
                })
            except Exception as e:
                print(f"    {m_a} vs {m_b}: test yapilamadi ({e})")

        all_results[scenario] = scenario_results

    return all_results


def main():
    parser = argparse.ArgumentParser(description="McNemar istatistiksel test")
    parser.add_argument("--batadal-dir", default="results2")
    parser.add_argument("--skab-dir", default="results_skab")
    parser.add_argument("--output", default="results2/statistical_tests.json")
    args = parser.parse_args()

    batadal_dir = Path(args.batadal_dir)
    skab_dir = Path(args.skab_dir)
    all_results = {}

    if batadal_dir.exists():
        all_results["BATADAL"] = run_mcnemar_for_dataset(
            batadal_dir, "BATADAL", use_skab_fold=False
        )

    if skab_dir.exists():
        all_results["SKAB"] = run_mcnemar_for_dataset(
            skab_dir, "SKAB", use_skab_fold=True
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSonuclar kaydedildi: {out_path}")


if __name__ == "__main__":
    main()

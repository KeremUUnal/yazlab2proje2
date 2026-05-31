import argparse
import json
from pathlib import Path

from src.config import Config
from src.experiments.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description="YazLab2 — Anomali Tespiti: DL vs Probabilistic Automata"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Konfigurasyon dosyasi yolu",
    )
    parser.add_argument(
        "--dataset",
        choices=["skab", "batadal", "both"],
        default="both",
        help="Calistirilacak veri seti",
    )
    parser.add_argument(
        "--model",
        choices=["dl", "automata", "all"],
        default="all",
        help="Calistirilacak model tipi (dl/automata/all)",
    )
    parser.add_argument(
        "--param-analysis",
        action="store_true",
        help="Otomata parametre analizi calistir (window_size x alphabet_size)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Sonuclarin kaydedilecegi klasor",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.output_dir:
        config.results.output_dir = args.output_dir

    Path(config.results.output_dir).mkdir(parents=True, exist_ok=True)

    runner = ExperimentRunner(config)
    all_results = {}

    # --- Parametre analizi ---
    if args.param_analysis:
        if args.dataset in ("batadal", "both"):
            runner.run_param_analysis("batadal")
        if args.dataset in ("skab", "both"):
            runner.run_param_analysis("skab")
        return

    # --- Ana deneyler ---
    if args.dataset in ("batadal", "both"):
        print("\n=== BATADAL Deneyleri ===")
        batadal_results = runner.run_batadal()
        all_results.update(batadal_results)
        print(f"BATADAL tamamlandi: {len(batadal_results)} deney")

    if args.dataset in ("skab", "both"):
        print("\n=== SKAB Deneyleri ===")
        skab_results = runner.run_skab()
        all_results.update(skab_results)
        print(f"SKAB tamamlandi: {len(skab_results)} deney")

    # --- Sonuclari kaydet ---
    summary_path = Path(config.results.output_dir) / "all_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nTum sonuclar kaydedildi: {summary_path}")

    # --- Ozet tablo ---
    print("\n" + "=" * 70)
    print(f"{'Deney':<40} {'F1':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8}")
    print("-" * 70)
    for key, val in all_results.items():
        f1 = val.get("f1_mean", val.get("f1_fold_mean", 0))
        acc = val.get("accuracy_mean", val.get("accuracy_fold_mean", 0))
        prec = val.get("precision_mean", val.get("precision_fold_mean", 0))
        rec = val.get("recall_mean", val.get("recall_fold_mean", 0))
        print(f"{key:<40} {f1:>8.4f} {acc:>8.4f} {prec:>8.4f} {rec:>8.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

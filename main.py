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
        help="Konfigürasyon dosyası yolu",
    )
    parser.add_argument(
        "--dataset",
        choices=["skab", "batadal", "both"],
        default="both",
        help="Çalıştırılacak veri seti",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Sonuçların kaydedileceği klasör (varsayılan: config'deki değer)",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.output_dir:
        config.results.output_dir = args.output_dir
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    runner = ExperimentRunner(config)
    all_results = {}

    if args.dataset in ("batadal", "both"):
        print("\n=== BATADAL Deneyleri ===")
        batadal_results = runner.run_batadal()
        all_results.update(batadal_results)
        print(f"BATADAL tamamlandı: {len(batadal_results)} senaryo")

    if args.dataset in ("skab", "both"):
        print("\n=== SKAB Deneyleri ===")
        skab_results = runner.run_skab()
        all_results.update(skab_results)
        print(f"SKAB tamamlandı: {len(skab_results)} senaryo")

    summary_path = Path(config.results.output_dir) / "all_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nTüm sonuçlar kaydedildi: {summary_path}")


if __name__ == "__main__":
    main()

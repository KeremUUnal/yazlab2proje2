"""
Automata görsellerini üretir:
- Transition probability heatmap
- State diagram
- DL vs Automata karşılaştırma

Kullanım:
    python generate_automata_plots.py
    python generate_automata_plots.py --dataset batadal
    python generate_automata_plots.py --dataset skab
"""
import argparse
import json
from copy import deepcopy
from pathlib import Path

from src.config import Config
from src.data.loader import BATADALLoader, SKABLoader
from src.data.preprocessor import BATADALPreprocessor, SKABPreprocessor
from src.automata.automata import ProbabilisticAutomata
from src.visualization.automata_plots import (
    plot_transition_heatmap,
    plot_state_diagram,
    plot_dl_vs_automata,
    plot_param_sensitivity_grid,
)


def get_automata_config(config: Config) -> Config:
    cfg = deepcopy(config)
    cfg.preprocessing.pca.n_components = 1
    return cfg


def run_batadal(config: Config, out_dir: str):
    print("\n=== BATADAL Automata Görselleri ===")
    cfg_auto = get_automata_config(config)
    cfg_auto.results.output_dir = out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df = BATADALLoader(cfg_auto).load()
    split = BATADALPreprocessor(cfg_auto).split(df, use_pca=True)

    model = ProbabilisticAutomata(cfg_auto.automata)
    model.fit(split.X_train, split.y_train)

    print(f"  Model eğitildi: {model.get_state_count()} state")

    p1 = plot_transition_heatmap(model, "BATADAL", cfg_auto)
    print(f"  Kaydedildi: {p1}")

    p2 = plot_state_diagram(model, "BATADAL", cfg_auto)
    print(f"  Kaydedildi: {p2}")

    # DL vs Automata karşılaştırma
    all_results_path = Path(out_dir) / "all_results.json"
    if all_results_path.exists():
        with open(all_results_path, encoding="utf-8") as f:
            all_results = json.load(f)
        p3 = plot_dl_vs_automata(all_results, "BATADAL", cfg_auto)
        print(f"  Kaydedildi: {p3}")

    # Parametre analizi görselleri (varsa)
    param_path = Path(out_dir) / "batadal_param_analysis.json"
    if param_path.exists():
        with open(param_path, encoding="utf-8") as f:
            param_results = json.load(f)
        paths = plot_param_sensitivity_grid(param_results, "BATADAL", cfg_auto)
        for p in paths:
            print(f"  Kaydedildi: {p}")
    else:
        print("  Parametre analizi sonucu yok — önce --param-analysis çalıştırın")


def run_skab(config: Config, out_dir: str):
    print("\n=== SKAB Automata Görselleri ===")
    cfg_auto = get_automata_config(config)
    cfg_auto.results.output_dir = out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df = SKABLoader(cfg_auto).load()
    prep = SKABPreprocessor(cfg_auto)
    X_raw, y, groups = prep.get_features_target(df)
    splits = prep.get_cv_splits(X_raw.values, y, groups)

    # İlk fold üzerinde model eğit (temsili görsel için)
    train_idx, test_idx = splits[0]
    fold_prep = SKABPreprocessor(cfg_auto)
    X_tr = fold_prep.fit_transform(X_raw.iloc[train_idx], use_pca=True)
    y_tr = y[train_idx]

    model = ProbabilisticAutomata(cfg_auto.automata)
    model.fit(X_tr, y_tr)

    print(f"  Model eğitildi (fold0): {model.get_state_count()} state")

    p1 = plot_transition_heatmap(model, "SKAB", cfg_auto)
    print(f"  Kaydedildi: {p1}")

    p2 = plot_state_diagram(model, "SKAB", cfg_auto)
    print(f"  Kaydedildi: {p2}")

    # DL vs Automata karşılaştırma
    all_results_path = Path(out_dir) / "all_results.json"
    if all_results_path.exists():
        with open(all_results_path, encoding="utf-8") as f:
            all_results = json.load(f)
        p3 = plot_dl_vs_automata(all_results, "SKAB", cfg_auto)
        print(f"  Kaydedildi: {p3}")

    # Parametre analizi görselleri (varsa)
    param_path = Path(out_dir) / "skab_param_analysis.json"
    if param_path.exists():
        with open(param_path, encoding="utf-8") as f:
            param_results = json.load(f)
        paths = plot_param_sensitivity_grid(param_results, "SKAB", cfg_auto)
        for p in paths:
            print(f"  Kaydedildi: {p}")
    else:
        print("  Parametre analizi sonucu yok — önce --param-analysis çalıştırın")


def main():
    parser = argparse.ArgumentParser(description="Automata görsel üretici")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", choices=["batadal", "skab", "both"], default="both")
    parser.add_argument("--batadal-dir", default="results2")
    parser.add_argument("--skab-dir", default="results_skab")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    if args.dataset in ("batadal", "both"):
        run_batadal(config, args.batadal_dir)

    if args.dataset in ("skab", "both"):
        run_skab(config, args.skab_dir)

    print("\nTamamlandı.")


if __name__ == "__main__":
    main()

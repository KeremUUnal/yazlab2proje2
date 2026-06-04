"""
Kaydedilmiş JSON sonuç dosyalarından görselleştirmeler üretir.

Kullanım:
    python generate_plots.py --results-dir results2 --output-dir plots/batadal --dataset batadal
    python generate_plots.py --results-dir results_skab --output-dir plots/skab --dataset skab
    python generate_plots.py  # varsayılan: her iki veri seti
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

MODELS = ["LSTM", "GRU", "CNN"]
ALL_MODELS = ["LSTM", "GRU", "CNN", "Automata"]
SCENARIOS = ["original", "noisy"]


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")


# ---------------------------------------------------------------------------
# Grafik fonksiyonları
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, title: str, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Normal", "Anomali"],
        yticklabels=["Normal", "Anomali"],
    )
    ax.set_title(title)
    ax.set_ylabel("Gercek Etiket")
    ax.set_xlabel("Tahmin Edilen Etiket")
    save_fig(fig, out_path)


def plot_roc(y_true, y_proba, title: str, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    save_fig(fig, out_path)


def plot_pr(y_true, y_proba, title: str, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, lw=2, label=f"AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    save_fig(fig, out_path)


def plot_model_comparison(model_names, f1_means, f1_stds, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(model_names))
    bars = ax.bar(x, f1_means, yerr=f1_stds, capsize=6, color="steelblue", alpha=0.85)
    for bar, mean in zip(bars, f1_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("F1 Score (mean +/- std)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    save_fig(fig, out_path)


def plot_noise_comparison(model_names, orig_means, noisy_means, title: str, out_path: Path) -> None:
    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, orig_means, width, label="Original", color="steelblue", alpha=0.85)
    ax.bar(x + width / 2, noisy_means, width, label="Noisy", color="coral", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("F1 Score (mean)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    save_fig(fig, out_path)


def plot_all_metrics_comparison(model_names, results_by_model, title: str, out_path: Path) -> None:
    metrics = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(model_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [results_by_model[m].get(metric, 0) for m in model_names]
        ax.bar(x + i * width, vals, width, label=label, alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# BATADAL görselleştirmeleri
# ---------------------------------------------------------------------------

def generate_batadal_plots(results_dir: Path, out_dir: Path) -> None:
    print("\n=== BATADAL Gorsellestirmeleri ===")

    orig_means, noisy_means, orig_stds = [], [], []
    all_orig = {}

    for model in MODELS:
        orig_path = results_dir / f"BATADAL_{model}_original.json"
        noisy_path = results_dir / f"BATADAL_{model}_noisy.json"

        if not orig_path.exists():
            print(f"  Atlandi (dosya yok): {orig_path}")
            continue

        orig = load_json(orig_path)
        all_orig[model] = orig

        f1_m = orig.get("f1_mean", 0)
        f1_s = orig.get("f1_std", 0)
        orig_means.append(f1_m)
        orig_stds.append(f1_s)

        noisy_f1 = 0
        if noisy_path.exists():
            noisy = load_json(noisy_path)
            noisy_f1 = noisy.get("f1_mean", 0)
        noisy_means.append(noisy_f1)

        # Confusion matrix + ROC + PR — sadece predictions varsa
        for scenario in SCENARIOS:
            p = results_dir / f"BATADAL_{model}_{scenario}.json"
            if not p.exists():
                continue
            data = load_json(p)
            seed_results = data.get("seed_results", [])
            seed0 = seed_results[0] if seed_results else {}
            y_true = seed0.get("y_true")
            y_pred = seed0.get("y_pred")
            y_proba = seed0.get("y_proba")

            if y_true is None:
                print(f"  Tahminler yok ({model}/{scenario}) — CM/ROC atlandı")
                continue

            y_true_arr = np.array(y_true)
            y_pred_arr = np.array(y_pred)

            plot_confusion_matrix(
                y_true_arr, y_pred_arr,
                title=f"Confusion Matrix — {model} BATADAL ({scenario})",
                out_path=out_dir / f"cm_BATADAL_{model}_{scenario}.png",
            )

            if y_proba is not None:
                y_proba_arr = np.array(y_proba)
                plot_roc(
                    y_true_arr, y_proba_arr,
                    title=f"ROC Egrisi — {model} BATADAL ({scenario})",
                    out_path=out_dir / f"roc_BATADAL_{model}_{scenario}.png",
                )
                plot_pr(
                    y_true_arr, y_proba_arr,
                    title=f"Precision-Recall — {model} BATADAL ({scenario})",
                    out_path=out_dir / f"pr_BATADAL_{model}_{scenario}.png",
                )

    if orig_means:
        plot_model_comparison(
            MODELS[:len(orig_means)], orig_means, orig_stds,
            title="Model Karsilastirmasi — BATADAL (Original)",
            out_path=out_dir / "model_comparison_BATADAL.png",
        )
        plot_noise_comparison(
            MODELS[:len(orig_means)], orig_means, noisy_means,
            title="Gurultu Etkisi — BATADAL",
            out_path=out_dir / "noise_comparison_BATADAL.png",
        )
        plot_all_metrics_comparison(
            MODELS[:len(all_orig)], all_orig,
            title="Tum Metrikler — BATADAL (Original)",
            out_path=out_dir / "all_metrics_BATADAL.png",
        )


# ---------------------------------------------------------------------------
# SKAB görselleştirmeleri
# ---------------------------------------------------------------------------

def generate_skab_plots(results_dir: Path, out_dir: Path) -> None:
    print("\n=== SKAB Gorsellestirmeleri ===")

    # Fold-aggregate sonuçlarını oku (all_results.json veya fold JSON'ları)
    all_path = results_dir / "all_results.json"
    if all_path.exists():
        all_data = load_json(all_path)
    else:
        all_data = {}

    orig_means, orig_stds, noisy_means = [], [], []
    all_orig = {}

    for model in MODELS:
        orig_key = f"SKAB_{model}_original"
        noisy_key = f"SKAB_{model}_noisy"

        orig = all_data.get(orig_key)
        if orig is None:
            # Fallback: fold0 JSON
            fold0_path = results_dir / f"SKAB_fold0_{model}_original.json"
            if not fold0_path.exists():
                print(f"  Atlandi (veri yok): {model}")
                continue
            orig = load_json(fold0_path)

        f1_key = "f1_fold_mean" if "f1_fold_mean" in orig else "f1_mean"
        std_key = "f1_fold_std" if "f1_fold_std" in orig else "f1_std"
        f1_m = orig.get(f1_key, 0)
        f1_s = orig.get(std_key, 0)
        orig_means.append(f1_m)
        orig_stds.append(f1_s)
        all_orig[model] = orig

        noisy = all_data.get(noisy_key)
        noisy_f1 = 0
        if noisy:
            nf_key = "f1_fold_mean" if "f1_fold_mean" in noisy else "f1_mean"
            noisy_f1 = noisy.get(nf_key, 0)
        noisy_means.append(noisy_f1)

        # Fold 0 / seed 42 tahminleri varsa confusion matrix
        fold0_orig = results_dir / f"SKAB_fold0_{model}_original.json"
        if fold0_orig.exists():
            data = load_json(fold0_orig)
            seed_results = data.get("seed_results", [])
            seed0 = seed_results[0] if seed_results else {}
            y_true = seed0.get("y_true")
            y_pred = seed0.get("y_pred")
            y_proba = seed0.get("y_proba")

            if y_true is not None:
                y_true_arr = np.array(y_true)
                y_pred_arr = np.array(y_pred)
                plot_confusion_matrix(
                    y_true_arr, y_pred_arr,
                    title=f"Confusion Matrix — {model} SKAB (fold0, original)",
                    out_path=out_dir / f"cm_SKAB_{model}_original.png",
                )
                if y_proba is not None:
                    y_proba_arr = np.array(y_proba)
                    plot_roc(
                        y_true_arr, y_proba_arr,
                        title=f"ROC Egrisi — {model} SKAB (fold0, original)",
                        out_path=out_dir / f"roc_SKAB_{model}_original.png",
                    )
                    plot_pr(
                        y_true_arr, y_proba_arr,
                        title=f"Precision-Recall — {model} SKAB (fold0, original)",
                        out_path=out_dir / f"pr_SKAB_{model}_original.png",
                    )
            else:
                print(f"  Tahminler yok (SKAB/{model}/fold0) — CM/ROC atlandı")

    if orig_means:
        valid_models = MODELS[:len(orig_means)]
        plot_model_comparison(
            valid_models, orig_means, orig_stds,
            title="Model Karsilastirmasi — SKAB (Original, 5-fold ortalama)",
            out_path=out_dir / "model_comparison_SKAB.png",
        )
        plot_noise_comparison(
            valid_models, orig_means, noisy_means,
            title="Gurultu Etkisi — SKAB",
            out_path=out_dir / "noise_comparison_SKAB.png",
        )

        # all_orig'u uyumlu hale getir
        compat_orig = {}
        for m, r in all_orig.items():
            compat_orig[m] = {
                "accuracy_mean": r.get("accuracy_fold_mean", r.get("accuracy_mean", 0)),
                "precision_mean": r.get("precision_fold_mean", r.get("precision_mean", 0)),
                "recall_mean": r.get("recall_fold_mean", r.get("recall_mean", 0)),
                "f1_mean": r.get("f1_fold_mean", r.get("f1_mean", 0)),
            }
        plot_all_metrics_comparison(
            valid_models, compat_orig,
            title="Tum Metrikler — SKAB (Original)",
            out_path=out_dir / "all_metrics_SKAB.png",
        )


# ---------------------------------------------------------------------------
# Cross-dataset karşılaştırma
# ---------------------------------------------------------------------------

def generate_cross_dataset_plot(
    batadal_dir: Path, skab_dir: Path, out_dir: Path
) -> None:
    print("\n=== Cross-Dataset Karsilastirma ===")
    datasets = []
    model_f1s = {m: [] for m in MODELS}

    for ds, res_dir in [("BATADAL", batadal_dir), ("SKAB", skab_dir)]:
        all_path = res_dir / "all_results.json"
        if not all_path.exists():
            continue
        all_data = load_json(all_path)
        datasets.append(ds)
        for model in MODELS:
            key = f"{ds}_{model}_original"
            r = all_data.get(key, {})
            f1 = r.get("f1_fold_mean", r.get("f1_mean", 0))
            model_f1s[model].append(f1)

    if len(datasets) < 2:
        return

    x = np.arange(len(datasets))
    width = 0.25
    colors = ["steelblue", "coral", "seagreen"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (model, color) in enumerate(zip(MODELS, colors)):
        vals = model_f1s[model]
        if len(vals) == len(datasets):
            ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("F1 Score (mean)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-Dataset F1 Karsilastirmasi")
    ax.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_dir / "cross_dataset_comparison.png")


# ---------------------------------------------------------------------------
# Ana fonksiyon
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Otomata gorsellestirmeleri
# ---------------------------------------------------------------------------

def generate_automata_plots(results_dir: Path, out_dir: Path, dataset_upper: str) -> None:
    """Otomata modeline ozel gorseller: heatmap, state diagram, parametre analizi."""
    print(f"\n=== {dataset_upper} Otomata Gorselleri ===")

    from copy import deepcopy
    from src.config import Config
    from src.data.loader import BATADALLoader, SKABLoader
    from src.data.preprocessor import BATADALPreprocessor, SKABPreprocessor
    from src.automata.automata import ProbabilisticAutomata
    from src.visualization.automata_plots import (
        plot_transition_heatmap,
        plot_state_diagram,
        plot_param_sensitivity_grid,
        plot_dl_vs_automata,
    )

    cfg = Config.from_yaml("config/config.yaml")
    cfg.results.output_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Otomata icin PCA=1
    cfg_auto = deepcopy(cfg)
    cfg_auto.preprocessing.pca.n_components = 1

    ds = dataset_upper.lower()
    if ds == "batadal":
        df = BATADALLoader(cfg_auto).load()
        split = BATADALPreprocessor(cfg_auto).split(df, use_pca=True)
        X_train, y_train = split.X_train, split.y_train
        X_test, y_test = split.X_test, split.y_test
    else:
        df = SKABLoader(cfg_auto).load()
        prep = SKABPreprocessor(cfg_auto)
        X_raw, y, groups = prep.get_features_target(df)
        splits = prep.get_cv_splits(X_raw.values, y, groups)
        train_idx, test_idx = splits[0]
        fold_prep = SKABPreprocessor(cfg_auto)
        X_train = fold_prep.fit_transform(X_raw.iloc[train_idx], use_pca=True)
        X_test = fold_prep.transform(X_raw.iloc[test_idx], use_pca=True)
        y_train, y_test = y[train_idx], y[test_idx]

    # Model egit
    model = ProbabilisticAutomata(cfg_auto.automata)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Transition Heatmap
    plot_transition_heatmap(model, dataset_upper, cfg)
    print(f"  Transition heatmap olusturuldu")

    # State Diagram
    try:
        import networkx
        plot_state_diagram(model, dataset_upper, cfg)
        print(f"  State diagram olusturuldu")
    except ImportError:
        print(f"  networkx yuklenmemis — state diagram atlandi (pip install networkx)")

    # Confusion Matrix
    plot_confusion_matrix(
        y_test, y_pred,
        title=f"Confusion Matrix — Automata {dataset_upper}",
        out_path=out_dir / f"cm_{dataset_upper}_Automata_original.png",
    )

    # ROC + PR
    plot_roc(
        y_test, y_proba,
        title=f"ROC Egrisi — Automata {dataset_upper}",
        out_path=out_dir / f"roc_{dataset_upper}_Automata_original.png",
    )
    plot_pr(
        y_test, y_proba,
        title=f"Precision-Recall — Automata {dataset_upper}",
        out_path=out_dir / f"pr_{dataset_upper}_Automata_original.png",
    )

    # Parametre analizi gorselleri
    param_path = results_dir / f"{ds}_param_analysis.json"
    if param_path.exists():
        param_results = load_json(param_path)
        plot_param_sensitivity_grid(param_results, dataset_upper, cfg)
        print(f"  Parametre grafikleri olusturuldu")
    else:
        print(f"  Parametre analizi yok — once: python main.py --param-analysis --dataset {ds}")

    # DL vs Automata karsilastirma
    all_path = results_dir / "all_results.json"
    if all_path.exists():
        all_results = load_json(all_path)
        plot_dl_vs_automata(all_results, dataset_upper, cfg)
        print(f"  DL vs Automata grafigi olusturuldu")


def main():
    parser = argparse.ArgumentParser(description="Gorselleştirme uretici")
    parser.add_argument("--batadal-dir", default="results2", help="BATADAL JSON dizini")
    parser.add_argument("--skab-dir", default="results_skab", help="SKAB JSON dizini")
    parser.add_argument("--output-dir", default="plots", help="Gorsel cıktı dizini")
    parser.add_argument("--results-dir", default="results", help="Ortak results dizini (otomata icin)")
    parser.add_argument(
        "--dataset",
        choices=["batadal", "skab", "both"],
        default="both",
    )
    args = parser.parse_args()

    batadal_dir = Path(args.batadal_dir)
    skab_dir = Path(args.skab_dir)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)

    if args.dataset in ("batadal", "both"):
        generate_batadal_plots(batadal_dir, out_dir / "batadal")
        generate_automata_plots(results_dir, out_dir / "batadal", "BATADAL")

    if args.dataset in ("skab", "both"):
        generate_skab_plots(skab_dir, out_dir / "skab")
        generate_automata_plots(results_dir, out_dir / "skab", "SKAB")

    if args.dataset == "both":
        generate_cross_dataset_plot(batadal_dir, skab_dir, out_dir)

    print("\nTamamlandi. Gorseller kaydedildi:", out_dir)


if __name__ == "__main__":
    main()

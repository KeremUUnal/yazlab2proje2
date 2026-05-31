"""
Gercek SKAB + BATADAL verisi ile otomata modelinin uctan uca testi.
Calistirma: python -m tests.test_real_data
"""
import numpy as np
import sys
sys.path.insert(0, ".")

from src.config import Config
from src.data.loader import SKABLoader, BATADALLoader
from src.data.preprocessor import SKABPreprocessor, BATADALPreprocessor
from src.automata.automata import ProbabilisticAutomata
from src.automata.explainability import ExplainabilityModule
from src.evaluation.metrics import compute_metrics


def test_skab(cfg: Config):
    print("=" * 60)
    print("SKAB Verisi ile Otomata Modeli Testi")
    print("=" * 60)
    print(f"Window size: {cfg.automata.window_size}")
    print(f"Alphabet size: {cfg.automata.alphabet_size}")

    # Veri yukle
    df = SKABLoader(cfg).load()
    print(f"\nToplam veri: {len(df)} satir")
    print(f"Anomali dagilimi:\n{df['anomaly'].value_counts().to_string()}")

    # On isleme
    prep = SKABPreprocessor(cfg)
    X_raw, y, groups = prep.get_features_target(df)
    print(f"Ozellik sayisi (ham): {X_raw.shape[1]}")

    # GroupKFold — ilk fold
    splits = prep.get_cv_splits(X_raw.values, y, groups)
    train_idx, test_idx = splits[0]

    # Otomata icin ozel preprocessor: PCA n_components=1 (sadece PC1)
    from copy import deepcopy
    cfg_auto = deepcopy(cfg)
    cfg_auto.preprocessing.pca.n_components = 1

    fold_prep = SKABPreprocessor(cfg_auto)
    X_train = fold_prep.fit_transform(X_raw.iloc[train_idx], use_pca=True)
    X_test = fold_prep.transform(X_raw.iloc[test_idx], use_pca=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"\n--- Fold 0 ---")
    print(f"Train: {X_train.shape[0]} sample (anomali: {np.sum(y_train == 1)})")
    print(f"Test:  {X_test.shape[0]} sample (anomali: {np.sum(y_test == 1)})")
    print(f"Otomata girdi boyutu: {X_train.shape[1]} (PC1)")

    # Model egit
    model = ProbabilisticAutomata(cfg.automata)
    model.fit(X_train, y_train)

    print(f"\nState sayisi: {model.get_state_count()}")
    print(f"Gecis yogunlugu: {model.get_transition_density():.4f}")
    print(f"Anomali threshold: {model.anomaly_threshold:.6f}")

    # Tahmin
    predictions = model.predict(X_test)
    print(f"Tahmin dagilimi: normal={np.sum(predictions == 0)}, anomali={np.sum(predictions == 1)}")

    # Metrikler
    metrics = compute_metrics(y_test, predictions)
    print(f"\n--- Metrikler ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Aciklanabilirlik (ilk 3)
    explainer = ExplainabilityModule(model)
    explanations = model.explain(X_test)
    print(f"\n--- Aciklanabilirlik (ilk 3 adim) ---")
    for exp in explanations[:3]:
        print(explainer.format_decision(exp))
        print("-" * 40)

    # Ozet rapor
    report = explainer.generate_report(X_test, y_test)
    print(f"\n--- Ozet ---")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")
    print()


def test_batadal(cfg: Config):
    print("=" * 60)
    print("BATADAL Verisi ile Otomata Modeli Testi")
    print("=" * 60)

    # Veri yukle
    df = BATADALLoader(cfg).load()
    print(f"Toplam veri: {len(df)} satir")
    print(f"Anomali dagilimi:\n{df['ATT_FLAG'].value_counts().to_string()}")

    # Otomata icin ozel preprocessor: PCA n_components=1 (sadece PC1)
    from copy import deepcopy
    cfg_auto = deepcopy(cfg)
    cfg_auto.preprocessing.pca.n_components = 1

    prep = BATADALPreprocessor(cfg_auto)
    split = prep.split(df, use_pca=True)

    X_train = split.X_train
    X_val = split.X_val
    X_test = split.X_test

    print(f"\nTrain: {X_train.shape[0]} sample (anomali: {np.sum(split.y_train == 1)})")
    print(f"Val:   {X_val.shape[0]} sample (anomali: {np.sum(split.y_val == 1)})")
    print(f"Test:  {X_test.shape[0]} sample (anomali: {np.sum(split.y_test == 1)})")
    print(f"Otomata girdi boyutu: {X_train.shape[1]} (PC1)")

    # Model egit
    model = ProbabilisticAutomata(cfg.automata)
    model.fit(X_train, split.y_train)

    print(f"\nState sayisi: {model.get_state_count()}")
    print(f"Gecis yogunlugu: {model.get_transition_density():.4f}")
    print(f"Anomali threshold: {model.anomaly_threshold:.6f}")

    # Tahmin (test seti uzerinde)
    predictions = model.predict(X_test)
    print(f"Tahmin dagilimi: normal={np.sum(predictions == 0)}, anomali={np.sum(predictions == 1)}")

    # Metrikler
    metrics = compute_metrics(split.y_test, predictions)
    print(f"\n--- Metrikler ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Aciklanabilirlik
    explainer = ExplainabilityModule(model)
    explanations = model.explain(X_test)
    print(f"\n--- Aciklanabilirlik (ilk 3 adim) ---")
    for exp in explanations[:3]:
        print(explainer.format_decision(exp))
        print("-" * 40)

    report = explainer.generate_report(X_test, split.y_test)
    print(f"\n--- Ozet ---")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")
    print()


def main():
    cfg = Config.from_yaml("config/config.yaml")

    test_skab(cfg)
    test_batadal(cfg)

    print("=" * 60)
    print("TAMAMLANDI — Her iki veri seti de basariyla test edildi")
    print("=" * 60)


if __name__ == "__main__":
    main()

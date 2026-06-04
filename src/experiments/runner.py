"""
Deney kosuturucusu — DL ve Otomata modellerini ayni pipeline'da calistirir.

3 senaryo: original, noisy, unseen
2 veri seti: SKAB (GroupKFold), BATADAL (zaman sirali 60/20/20)
5 seed: [42, 123, 2026, 7, 999]
"""
import json
import time
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.config import Config
from src.data.loader import BATADALLoader, SKABLoader
from src.data.preprocessor import BATADALPreprocessor, SKABPreprocessor
from src.evaluation.metrics import compute_metrics
from src.experiments.noise import add_gaussian_noise
from src.models.cnn_model import CNNModel
from src.models.gru_model import GRUModel
from src.models.lstm_model import LSTMModel
from src.models.trainer import Trainer
from src.automata.automata import ProbabilisticAutomata
from src.automata.explainability import ExplainabilityModule

DL_MODELS = {
    "LSTM": LSTMModel,
    "GRU": GRUModel,
    "CNN": CNNModel,
}


def _setup_logger(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("experiment_runner")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(
            Path(log_dir) / "experiments.log", encoding="utf-8"
        )
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(fh)
    return logger


class ExperimentRunner:
    def __init__(self, config: Config):
        self.config = config
        self.trainer = Trainer(config)
        Path(config.results.output_dir).mkdir(parents=True, exist_ok=True)
        self.logger = _setup_logger(config.results.log_dir)

    # ==================================================================
    # BATADAL — zaman sirali 60/20/20
    # ==================================================================
    def run_batadal(self) -> Dict:
        loader = BATADALLoader(self.config)
        df = loader.load()

        results = {}

        # --- DL Modelleri ---
        results.update(self._run_batadal_dl(df))

        # --- Otomata Modeli ---
        results.update(self._run_batadal_automata(df))

        return results

    def _run_batadal_dl(self, df) -> Dict:
        """BATADAL uzerinde DL modellerini calistirir."""
        preprocessor = BATADALPreprocessor(self.config)
        split = preprocessor.split(df, use_pca=True)

        results = {}
        for model_name, model_cls in DL_MODELS.items():
            for scenario in self.config.experiment.scenarios:
                if scenario == "unseen":
                    continue

                X_train = split.X_train.copy()
                X_val = split.X_val.copy()

                if scenario == "noisy":
                    X_train = add_gaussian_noise(X_train, self.config)
                    X_val = add_gaussian_noise(X_val, self.config)

                key = f"BATADAL_{model_name}_{scenario}"
                results[key] = self.trainer.train_evaluate(
                    model_cls, model_name, "BATADAL",
                    X_train, split.y_train,
                    X_val, split.y_val,
                    split.X_test, split.y_test,
                    scenario=scenario,
                )
        return results

    def _run_batadal_automata(self, df) -> Dict:
        """BATADAL uzerinde otomata modelini 3 senaryo ile calistirir."""
        # Otomata icin PCA n_components=1
        cfg_auto = deepcopy(self.config)
        cfg_auto.preprocessing.pca.n_components = 1

        preprocessor = BATADALPreprocessor(cfg_auto)
        split = preprocessor.split(df, use_pca=True)

        results = {}
        for scenario in self.config.experiment.scenarios:
            X_train = split.X_train.copy()
            y_train = split.y_train.copy()
            X_test = split.X_test.copy()
            y_test = split.y_test.copy()

            if scenario == "noisy":
                X_train = add_gaussian_noise(X_train, self.config)
                X_test = add_gaussian_noise(X_test, self.config)

            key = f"BATADAL_Automata_{scenario}"
            results[key] = self._run_automata_single(
                X_train, y_train, X_test, y_test,
                "BATADAL", scenario,
            )

        return results

    # ==================================================================
    # SKAB — GroupKFold / StratifiedGroupKFold
    # ==================================================================
    def run_skab(self) -> Dict:
        loader = SKABLoader(self.config)
        df = loader.load()

        results = {}

        # --- DL Modelleri ---
        results.update(self._run_skab_dl(df))

        # --- Otomata Modeli ---
        results.update(self._run_skab_automata(df))

        return results

    def _run_skab_dl(self, df) -> Dict:
        """SKAB uzerinde DL modellerini GroupKFold ile calistirir."""
        preprocessor = SKABPreprocessor(self.config)
        X_raw, y, groups = preprocessor.get_features_target(df)
        splits = preprocessor.get_cv_splits(X_raw.values, y, groups)

        results = {}
        for model_name, model_cls in DL_MODELS.items():
            for scenario in self.config.experiment.scenarios:
                if scenario == "unseen":
                    continue

                fold_metrics = []
                for fold_idx, (train_idx, test_idx) in enumerate(splits):
                    fold_preprocessor = SKABPreprocessor(self.config)
                    X_tr = fold_preprocessor.fit_transform(
                        X_raw.iloc[train_idx], use_pca=True
                    )
                    X_te = fold_preprocessor.transform(
                        X_raw.iloc[test_idx], use_pca=True
                    )
                    y_tr = y[train_idx]
                    y_te = y[test_idx]

                    if scenario == "noisy":
                        X_tr = add_gaussian_noise(X_tr, self.config)

                    val_size = max(1, int(len(X_te) * 0.25))
                    X_val = X_te[:val_size]
                    y_val = y_te[:val_size]
                    X_te_final = X_te[val_size:]
                    y_te_final = y_te[val_size:]

                    summary = self.trainer.train_evaluate(
                        model_cls, model_name, f"SKAB_fold{fold_idx}",
                        X_tr, y_tr, X_val, y_val,
                        X_te_final, y_te_final,
                        scenario=scenario,
                    )
                    fold_metrics.append(summary)

                key = f"SKAB_{model_name}_{scenario}"
                results[key] = _aggregate_folds(
                    fold_metrics, model_name, "SKAB", scenario
                )
        return results

    def _run_skab_automata(self, df) -> Dict:
        """SKAB uzerinde otomata modelini GroupKFold ile 3 senaryo calistirir."""
        # Otomata icin PCA n_components=1
        cfg_auto = deepcopy(self.config)
        cfg_auto.preprocessing.pca.n_components = 1

        preprocessor = SKABPreprocessor(cfg_auto)
        X_raw, y, groups = preprocessor.get_features_target(df)
        splits = preprocessor.get_cv_splits(X_raw.values, y, groups)

        results = {}
        for scenario in self.config.experiment.scenarios:
            fold_metrics = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                fold_preprocessor = SKABPreprocessor(cfg_auto)
                X_tr = fold_preprocessor.fit_transform(
                    X_raw.iloc[train_idx], use_pca=True
                )
                X_te = fold_preprocessor.transform(
                    X_raw.iloc[test_idx], use_pca=True
                )
                y_tr = y[train_idx]
                y_te = y[test_idx]

                if scenario == "noisy":
                    X_tr = add_gaussian_noise(X_tr, self.config)
                    X_te = add_gaussian_noise(X_te, self.config)

                fold_result = self._run_automata_single(
                    X_tr, y_tr, X_te, y_te,
                    f"SKAB_fold{fold_idx}", scenario,
                )
                fold_metrics.append(fold_result)

            key = f"SKAB_Automata_{scenario}"
            results[key] = _aggregate_automata_folds(
                fold_metrics, "SKAB", scenario
            )

        return results

    # ==================================================================
    # Otomata tek calistirma
    # ==================================================================
    def _run_automata_single(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dataset_name: str,
        scenario: str,
    ) -> Dict:
        """Otomata modelini tek bir train/test bolmesi uzerinde calistirir."""

        self.logger.info(
            f"AUTOMATA BASLADI | dataset={dataset_name} scenario={scenario} "
            f"window={self.config.automata.window_size} "
            f"alphabet={self.config.automata.alphabet_size}"
        )

        model = ProbabilisticAutomata(self.config.automata)

        # Egitim
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        # Tahmin
        t0 = time.time()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        infer_time = time.time() - t0

        # Metrikler
        metrics = compute_metrics(y_test, y_pred)

        # Aciklanabilirlik raporu
        explainer = ExplainabilityModule(model)
        report = explainer.generate_report(X_test, y_test)

        result = {
            "model": "Automata",
            "dataset": dataset_name,
            "scenario": scenario,
            "train_time_sec": round(train_time, 3),
            "inference_time_sec": round(infer_time, 4),
            "accuracy_mean": metrics["accuracy"],
            "precision_mean": metrics["precision"],
            "recall_mean": metrics["recall"],
            "f1_mean": metrics["f1"],
            "accuracy_std": 0.0,
            "precision_std": 0.0,
            "recall_std": 0.0,
            "f1_std": 0.0,
            "automata_info": {
                "state_count": model.get_state_count(),
                "transition_density": round(model.get_transition_density(), 4),
                "anomaly_threshold": round(model.anomaly_threshold, 6),
                "unseen_ratio": report["summary"]["unseen_ratio"],
            },
            "explainability_summary": report["summary"],
        }

        # Sonucu kaydet
        self._save_automata_result(result, dataset_name, scenario)

        self.logger.info(
            f"AUTOMATA BITTI  | dataset={dataset_name} scenario={scenario} "
            f"f1={metrics['f1']:.4f} acc={metrics['accuracy']:.4f} "
            f"states={model.get_state_count()} "
            f"unseen={report['summary']['unseen_ratio']:.4f} "
            f"train={train_time:.3f}s"
        )

        print(
            f"  [Automata | {dataset_name} | {scenario}] "
            f"F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}  "
            f"States={model.get_state_count()}"
        )

        return result

    def _save_automata_result(
        self, result: Dict, dataset_name: str, scenario: str
    ) -> None:
        fname = f"{dataset_name}_Automata_{scenario}.json"
        path = Path(self.config.results.output_dir) / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # ==================================================================
    # Parametre analizi
    # ==================================================================
    def run_param_analysis(self, dataset: str = "batadal") -> Dict:
        """
        window_size x alphabet_size kombinasyonlari icin otomata modelini calistirir.
        Her kombinasyon icin: F1, state sayisi, gecis yogunlugu raporlanir.
        """
        window_sizes = self.config.automata.param_search.window_sizes
        alphabet_sizes = self.config.automata.param_search.alphabet_sizes

        print(f"\n=== Parametre Analizi ({dataset.upper()}) ===")
        print(f"Window sizes: {window_sizes}")
        print(f"Alphabet sizes: {alphabet_sizes}")
        print(f"Toplam kombinasyon: {len(window_sizes) * len(alphabet_sizes)}")

        # Veri hazirla
        if dataset == "batadal":
            X_train, y_train, X_test, y_test = self._prepare_batadal_for_automata()
        else:
            X_train, y_train, X_test, y_test = self._prepare_skab_for_automata()

        results = {}
        for ws in window_sizes:
            for als in alphabet_sizes:
                cfg_temp = deepcopy(self.config.automata)
                cfg_temp.window_size = ws
                cfg_temp.alphabet_size = als

                model = ProbabilisticAutomata(cfg_temp)

                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics = compute_metrics(y_test, y_pred)

                    key = f"ws{ws}_as{als}"
                    results[key] = {
                        "window_size": ws,
                        "alphabet_size": als,
                        "f1": metrics["f1"],
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "state_count": model.get_state_count(),
                        "transition_density": round(
                            model.get_transition_density(), 4
                        ),
                    }

                    print(
                        f"  ws={ws} as={als} | "
                        f"F1={metrics['f1']:.4f} "
                        f"Acc={metrics['accuracy']:.4f} "
                        f"States={model.get_state_count()} "
                        f"Density={model.get_transition_density():.4f}"
                    )
                except Exception as e:
                    print(f"  ws={ws} as={als} | HATA: {e}")
                    results[f"ws{ws}_as{als}"] = {"error": str(e)}

        # Sonuclari kaydet
        path = Path(self.config.results.output_dir) / f"{dataset}_param_analysis.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Parametre analizi kaydedildi: {path}")

        return results

    def _prepare_batadal_for_automata(self):
        """BATADAL verisini otomata icin hazirlar (PCA=1)."""
        cfg_auto = deepcopy(self.config)
        cfg_auto.preprocessing.pca.n_components = 1
        df = BATADALLoader(cfg_auto).load()
        split = BATADALPreprocessor(cfg_auto).split(df, use_pca=True)
        return split.X_train, split.y_train, split.X_test, split.y_test

    def _prepare_skab_for_automata(self):
        """SKAB verisini otomata icin hazirlar (PCA=1, ilk fold)."""
        cfg_auto = deepcopy(self.config)
        cfg_auto.preprocessing.pca.n_components = 1
        df = SKABLoader(cfg_auto).load()
        prep = SKABPreprocessor(cfg_auto)
        X_raw, y, groups = prep.get_features_target(df)
        splits = prep.get_cv_splits(X_raw.values, y, groups)
        train_idx, test_idx = splits[0]
        fold_prep = SKABPreprocessor(cfg_auto)
        X_tr = fold_prep.fit_transform(X_raw.iloc[train_idx], use_pca=True)
        X_te = fold_prep.transform(X_raw.iloc[test_idx], use_pca=True)
        return X_tr, y[train_idx], X_te, y[test_idx]


# ==================================================================
# Yardimci fonksiyonlar
# ==================================================================
def _aggregate_folds(fold_results, model_name, dataset, scenario) -> Dict:
    """DL fold sonuclarini birlestirir."""
    keys = ["f1_mean", "accuracy_mean", "precision_mean", "recall_mean"]
    aggregated: Dict = {
        "model": model_name,
        "dataset": dataset,
        "scenario": scenario,
        "n_folds": len(fold_results),
        "fold_results": fold_results,
    }
    for k in keys:
        vals = [r[k] for r in fold_results]
        metric = k.replace("_mean", "")
        aggregated[f"{metric}_fold_mean"] = round(float(np.mean(vals)), 4)
        aggregated[f"{metric}_fold_std"] = round(float(np.std(vals)), 4)
    return aggregated


def _aggregate_automata_folds(fold_results, dataset, scenario) -> Dict:
    """Otomata fold sonuclarini birlestirir."""
    aggregated: Dict = {
        "model": "Automata",
        "dataset": dataset,
        "scenario": scenario,
        "n_folds": len(fold_results),
        "fold_results": fold_results,
    }
    for metric in ["accuracy", "precision", "recall", "f1"]:
        vals = [r[f"{metric}_mean"] for r in fold_results]
        aggregated[f"{metric}_fold_mean"] = round(float(np.mean(vals)), 4)
        aggregated[f"{metric}_fold_std"] = round(float(np.std(vals)), 4)

    # Otomata-spesifik istatistikler
    state_counts = [r["automata_info"]["state_count"] for r in fold_results]
    aggregated["avg_state_count"] = round(float(np.mean(state_counts)), 1)
    aggregated["avg_unseen_ratio"] = round(
        float(np.mean([r["automata_info"]["unseen_ratio"] for r in fold_results])), 4
    )

    return aggregated

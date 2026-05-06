from typing import Dict

import numpy as np

from src.config import Config
from src.data.loader import BATADALLoader, SKABLoader
from src.data.preprocessor import BATADALPreprocessor, DataSplit, SKABPreprocessor
from src.evaluation.metrics import compute_metrics
from src.experiments.noise import add_gaussian_noise
from src.models.cnn_model import CNNModel
from src.models.gru_model import GRUModel
from src.models.lstm_model import LSTMModel
from src.models.trainer import Trainer, create_sequences

DL_MODELS = {
    "LSTM": LSTMModel,
    "GRU": GRUModel,
    "CNN": CNNModel,
}


class ExperimentRunner:
    def __init__(self, config: Config):
        self.config = config
        self.trainer = Trainer(config)

    # ------------------------------------------------------------------
    # BATADAL — zaman sıralı 60/20/20
    # ------------------------------------------------------------------
    def run_batadal(self) -> Dict:
        loader = BATADALLoader(self.config)
        df = loader.load()
        preprocessor = BATADALPreprocessor(self.config)
        split = preprocessor.split(df)

        results = {}
        for model_name, model_cls in DL_MODELS.items():
            for scenario in self.config.experiment.scenarios:
                if scenario == "unseen":
                    # Unseen senaryosu otomata modeline aittir
                    continue

                X_train = split.X_train.copy()
                X_val = split.X_val.copy()

                if scenario == "noisy":
                    X_train = add_gaussian_noise(X_train, self.config)
                    X_val = add_gaussian_noise(X_val, self.config)

                key = f"BATADAL_{model_name}_{scenario}"
                results[key] = self.trainer.train_evaluate(
                    model_cls,
                    model_name,
                    "BATADAL",
                    X_train,
                    split.y_train,
                    X_val,
                    split.y_val,
                    split.X_test,
                    split.y_test,
                    scenario=scenario,
                )
        return results

    # ------------------------------------------------------------------
    # SKAB — GroupKFold / StratifiedGroupKFold
    # ------------------------------------------------------------------
    def run_skab(self) -> Dict:
        loader = SKABLoader(self.config)
        df = loader.load()
        preprocessor = SKABPreprocessor(self.config)
        X_raw, y, groups = preprocessor.get_features_target(df)

        results: Dict = {}
        splits = preprocessor.get_cv_splits(
            X_raw.values, y, groups
        )

        for model_name, model_cls in DL_MODELS.items():
            for scenario in self.config.experiment.scenarios:
                if scenario == "unseen":
                    continue

                fold_metrics = []
                for fold_idx, (train_idx, test_idx) in enumerate(splits):
                    # Her fold'da scaler/PCA sadece train verisiyle fit edilir
                    fold_preprocessor = SKABPreprocessor(self.config)

                    X_tr_raw = X_raw.iloc[train_idx]
                    X_te_raw = X_raw.iloc[test_idx]
                    y_tr = y[train_idx]
                    y_te = y[test_idx]

                    X_tr = fold_preprocessor.fit_transform(X_tr_raw)
                    X_te = fold_preprocessor.transform(X_te_raw)

                    if scenario == "noisy":
                        X_tr = add_gaussian_noise(X_tr, self.config)

                    # Validation olarak test setinin %25'ini ayır
                    val_size = max(1, int(len(X_te) * 0.25))
                    X_val = X_te[:val_size]
                    y_val = y_te[:val_size]
                    X_te_final = X_te[val_size:]
                    y_te_final = y_te[val_size:]

                    summary = self.trainer.train_evaluate(
                        model_cls,
                        model_name,
                        f"SKAB_fold{fold_idx}",
                        X_tr, y_tr,
                        X_val, y_val,
                        X_te_final, y_te_final,
                        scenario=scenario,
                    )
                    fold_metrics.append(summary)

                key = f"SKAB_{model_name}_{scenario}"
                results[key] = _aggregate_folds(fold_metrics, model_name, "SKAB", scenario)
        return results


def _aggregate_folds(fold_results, model_name, dataset, scenario) -> Dict:
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

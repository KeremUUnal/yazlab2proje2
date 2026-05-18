import json
import time
from pathlib import Path
from typing import Dict, List, Type

import numpy as np
import torch

from src.config import Config
from src.evaluation.metrics import compute_metrics
from src.models.base import BaseModel


def create_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """1D zaman serisinden kayan pencere dizileri oluşturur."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        Path(config.results.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.results.log_dir).mkdir(parents=True, exist_ok=True)

    def train_evaluate(
        self,
        model_cls: Type[BaseModel],
        model_name: str,
        dataset_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scenario: str = "original",
    ) -> Dict:
        seq_len = self.config.model.sequence_length
        all_results = []

        for seed in self.config.model.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, seq_len)
            X_v_seq, y_v_seq = create_sequences(X_val, y_val, seq_len)
            X_te_seq, y_te_seq = create_sequences(X_test, y_test, seq_len)

            model = model_cls(self.config)
            model.build(input_shape=(seq_len, X_train.shape[1]))

            t0 = time.time()
            model.fit(X_tr_seq, y_tr_seq, X_v_seq, y_v_seq)
            train_time = time.time() - t0

            t0 = time.time()
            y_pred = model.predict(X_te_seq)
            y_proba = model.predict_proba(X_te_seq)
            infer_time = time.time() - t0

            metrics = compute_metrics(y_te_seq, y_pred)
            metrics["seed"] = seed
            metrics["train_time_sec"] = round(train_time, 3)
            metrics["inference_time_sec"] = round(infer_time, 4)
            # Sadece ilk seed için tahminleri sakla (görselleştirme için)
            if seed == self.config.model.seeds[0]:
                metrics["y_true"] = y_te_seq.tolist()
                metrics["y_pred"] = y_pred.tolist()
                metrics["y_proba"] = y_proba.tolist()
            all_results.append(metrics)

            print(
                f"  [{model_name} | {dataset_name} | {scenario} | seed={seed}] "
                f"F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}"
            )

        summary = self._summarize(all_results, model_name, dataset_name, scenario)
        self._save(summary, model_name, dataset_name, scenario)
        return summary

    def _summarize(
        self,
        results: List[Dict],
        model_name: str,
        dataset_name: str,
        scenario: str,
    ) -> Dict:
        keys = ["accuracy", "precision", "recall", "f1"]
        summary: Dict = {
            "model": model_name,
            "dataset": dataset_name,
            "scenario": scenario,
            "seed_results": results,
        }
        for k in keys:
            vals = [r[k] for r in results]
            summary[f"{k}_mean"] = round(float(np.mean(vals)), 4)
            summary[f"{k}_std"] = round(float(np.std(vals)), 4)
        return summary

    def _save(
        self, summary: Dict, model_name: str, dataset_name: str, scenario: str
    ) -> None:
        fname = f"{dataset_name}_{model_name}_{scenario}.json"
        path = Path(self.config.results.output_dir) / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

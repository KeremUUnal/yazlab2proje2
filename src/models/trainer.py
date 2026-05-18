import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Type

import numpy as np
import torch

from src.config import Config
from src.evaluation.metrics import compute_metrics
from src.models.base import BaseModel


def _setup_logger(log_dir: str) -> logging.Logger:
    log_path = Path(log_dir) / "training.log"
    logger = logging.getLogger("trainer")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
    return logger


def create_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
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
        self.logger = _setup_logger(config.results.log_dir)

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
        cfg = self.config.model

        self.logger.info(
            f"DENEY BASLADI | model={model_name} dataset={dataset_name} scenario={scenario} "
            f"seq_len={seq_len} epochs={cfg.epochs} batch={cfg.batch_size} "
            f"patience={cfg.early_stopping.patience} "
            f"pca_components={self.config.preprocessing.pca.n_components} "
            f"seeds={cfg.seeds}"
        )

        all_results = []

        for seed in cfg.seeds:
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

            if seed == cfg.seeds[0]:
                metrics["y_true"] = y_te_seq.tolist()
                metrics["y_pred"] = y_pred.tolist()
                metrics["y_proba"] = y_proba.tolist()

            all_results.append(metrics)

            self.logger.info(
                f"  seed={seed} | f1={metrics['f1']:.4f} acc={metrics['accuracy']:.4f} "
                f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} "
                f"train={train_time:.1f}s infer={infer_time:.3f}s"
            )

            print(
                f"  [{model_name} | {dataset_name} | {scenario} | seed={seed}] "
                f"F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}"
            )

        summary = self._summarize(all_results, model_name, dataset_name, scenario)
        self._save(summary, model_name, dataset_name, scenario)

        self.logger.info(
            f"DENEY BITTI  | model={model_name} dataset={dataset_name} scenario={scenario} "
            f"f1_mean={summary['f1_mean']:.4f} f1_std={summary['f1_std']:.4f} "
            f"acc_mean={summary['accuracy_mean']:.4f}"
        )

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

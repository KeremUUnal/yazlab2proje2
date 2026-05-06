from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.config import Config


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    dataset: str,
    config: Config,
) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name} ({dataset})")
    ax.set_ylabel("Gerçek Etiket")
    ax.set_xlabel("Tahmin Edilen Etiket")
    path = Path(config.results.output_dir) / f"cm_{dataset}_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    dataset: str,
    config: Config,
) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Eğrisi — {model_name} ({dataset})")
    ax.legend(loc="lower right")
    path = Path(config.results.output_dir) / f"roc_{dataset}_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_precision_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    dataset: str,
    config: Config,
) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {model_name} ({dataset})")
    path = Path(config.results.output_dir) / f"pr_{dataset}_{model_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_parameter_sensitivity(
    results: Dict[str, Dict],
    param_name: str,
    param_values: List,
    config: Config,
) -> Path:
    """
    results: {model_name: {param_value: f1_score}}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, scores in results.items():
        vals = [scores.get(v, float("nan")) for v in param_values]
        ax.plot(param_values, vals, marker="o", label=model_name)
    ax.set_xlabel(param_name)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Parametre Duyarlılığı: {param_name}")
    ax.legend()
    path = Path(config.results.output_dir) / f"sensitivity_{param_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(
    summary_results: List[Dict],
    dataset: str,
    config: Config,
) -> Path:
    """Tüm modellerin F1 ortalaması ve std'sini gösteren bar grafiği."""
    names = [r["model"] for r in summary_results]
    means = [r.get("f1_mean", r.get("f1_fold_mean", 0)) for r in summary_results]
    stds = [r.get("f1_std", r.get("f1_fold_std", 0)) for r in summary_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("F1 Score (mean ± std)")
    ax.set_title(f"Model Karşılaştırması — {dataset}")
    ax.set_ylim(0, 1)
    path = Path(config.results.output_dir) / f"model_comparison_{dataset}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

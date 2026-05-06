from typing import Dict, List

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test_fn


def wilcoxon_test(scores_a: List[float], scores_b: List[float]) -> Dict:
    """İki modelin F1 skorları üzerinde Wilcoxon işaretli sıra testi uygular."""
    stat, p = wilcoxon(scores_a, scores_b)
    return {
        "test": "wilcoxon",
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < 0.05),
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict:
    """İki modelin tahminleri arasında McNemar testi uygular."""
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true
    n11 = int(np.sum(correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    n00 = int(np.sum(~correct_a & ~correct_b))
    table = [[n11, n10], [n01, n00]]
    result = mcnemar_test_fn(table, exact=True)
    return {
        "test": "mcnemar",
        "contingency_table": table,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < 0.05),
    }

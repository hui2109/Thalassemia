# -*- coding: utf-8 -*-
"""
metrics.py
===========
三分类任务的核心评价指标计算，全部指标均可通过 Bootstrap 重采样给出 95% 置信区间。

包含的指标：
    - Accuracy（总体准确率）
    - Macro-average AUC（宏平均 AUC，one-vs-rest）
    - Macro F1 / Weighted F1
    - 每一类别的 灵敏度(Sensitivity) / 特异度(Specificity) / 阳性预测值(PPV) / 阴性预测值(NPV)
      （均按照 one-vs-rest 方式，把该类别视为"阳性"，其余类别合并视为"阴性"）

所有函数只依赖 numpy / pandas / sklearn，不涉及绘图或建模逻辑，
方便单独做单元测试。
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


def _confusion_counts_one_vs_rest(y_true: np.ndarray, y_pred: np.ndarray, positive_class: int) -> Tuple[int, int, int, int]:
    """计算某一类别在 one-vs-rest 视角下的混淆矩阵四格表。

    Parameters
    ----------
    y_true : np.ndarray
        真实标签。
    y_pred : np.ndarray
        预测标签。
    positive_class : int
        视为"阳性"的类别编码。

    Returns
    -------
    Tuple[int, int, int, int]
        (TP, FP, TN, FN)
    """
    y_true_bin = (y_true == positive_class).astype(int)
    y_pred_bin = (y_pred == positive_class).astype(int)

    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    return tp, fp, tn, fn


def per_class_sens_spec_ppv_npv(
    y_true: np.ndarray, y_pred: np.ndarray, classes: List[int]
) -> Dict[int, Dict[str, float]]:
    """计算每个类别（one-vs-rest）的灵敏度、特异度、PPV、NPV。

    Parameters
    ----------
    y_true : np.ndarray
        真实标签。
    y_pred : np.ndarray
        预测标签。
    classes : List[int]
        类别列表，如 [0, 1, 2]。

    Returns
    -------
    Dict[int, Dict[str, float]]
        {类别: {"sensitivity":.., "specificity":.., "ppv":.., "npv":..}}
        分母为 0 时对应指标记为 np.nan。
    """
    results = {}
    for c in classes:
        tp, fp, tn, fn = _confusion_counts_one_vs_rest(y_true, y_pred, c)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        results[c] = {"sensitivity": sensitivity, "specificity": specificity, "ppv": ppv, "npv": npv}
    return results


def compute_point_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, classes: List[int]
) -> Dict[str, object]:
    """计算一次预测结果的全部点估计指标（不含置信区间）。

    Parameters
    ----------
    y_true : np.ndarray
        真实标签，shape (n_samples,)。
    y_pred : np.ndarray
        预测标签，shape (n_samples,)。
    y_proba : np.ndarray
        预测概率矩阵，shape (n_samples, n_classes)，列顺序需与 `classes` 一致。
    classes : List[int]
        类别列表，如 [0, 1, 2]。

    Returns
    -------
    Dict[str, object]
        {
          "accuracy": float, "macro_auc": float, "macro_f1": float, "weighted_f1": float,
          "per_class": {类别: {sensitivity, specificity, ppv, npv}}
        }
    """
    accuracy = accuracy_score(y_true, y_pred)

    # 若测试集/重采样样本中缺失某个类别，roc_auc_score 会报错，需要兜底处理
    try:
        macro_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=classes)
    except ValueError:
        macro_auc = np.nan

    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=classes, zero_division=0)
    per_class = per_class_sens_spec_ppv_npv(y_true, y_pred, classes)

    return {
        "accuracy": accuracy,
        "macro_auc": macro_auc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


def bootstrap_ci_for_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: List[int],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Dict[str, object]:
    """通过对样本进行有放回重抽样（Bootstrap），计算全部指标的 95% 置信区间。

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        真实标签与预测标签。
    y_proba : np.ndarray
        预测概率矩阵。
    classes : List[int]
        类别列表。
    n_bootstrap : int, default 1000
        重采样次数。
    alpha : float, default 0.05
        显著性水平，0.05 对应 95% CI。
    random_state : int, default 42
        随机种子。

    Returns
    -------
    Dict[str, object]
        与 `compute_point_metrics` 结构一致，但每个数值替换为 (point, lower, upper) 三元组。
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)

    scalar_names = ["accuracy", "macro_auc", "macro_f1", "weighted_f1"]
    scalar_samples: Dict[str, List[float]] = {k: [] for k in scalar_names}
    per_class_samples: Dict[int, Dict[str, List[float]]] = {
        c: {"sensitivity": [], "specificity": [], "ppv": [], "npv": []} for c in classes
    }

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    successful_draws = 0
    max_attempts = n_bootstrap * 3
    attempts = 0
    while successful_draws < n_bootstrap and attempts < max_attempts:
        attempts += 1
        idx = rng.randint(0, n, size=n)
        yt, yp, ypb = y_true[idx], y_pred[idx], y_proba[idx]

        # 若该次重采样未覆盖到全部类别，跳过（否则 AUC / per-class 指标无法定义）
        if len(np.unique(yt)) < len(classes):
            continue

        point = compute_point_metrics(yt, yp, ypb, classes)
        for k in scalar_names:
            scalar_samples[k].append(point[k])
        for c in classes:
            for metric_name in ["sensitivity", "specificity", "ppv", "npv"]:
                per_class_samples[c][metric_name].append(point["per_class"][c][metric_name])
        successful_draws += 1

    point_estimate = compute_point_metrics(y_true, y_pred, y_proba, classes)

    def _ci(sample_list: List[float], point_value: float) -> Tuple[float, float, float]:
        arr = np.array([v for v in sample_list if not np.isnan(v)])
        if len(arr) == 0:
            return point_value, np.nan, np.nan
        lower = np.percentile(arr, 100 * alpha / 2)
        upper = np.percentile(arr, 100 * (1 - alpha / 2))
        return point_value, lower, upper

    result = {}
    for k in scalar_names:
        result[k] = _ci(scalar_samples[k], point_estimate[k])

    result["per_class"] = {}
    for c in classes:
        result["per_class"][c] = {}
        for metric_name in ["sensitivity", "specificity", "ppv", "npv"]:
            result["per_class"][c][metric_name] = _ci(
                per_class_samples[c][metric_name], point_estimate["per_class"][c][metric_name]
            )

    return result


def metrics_dict_to_dataframe(
    metrics_with_ci: Dict[str, object], class_labels: Dict[int, str], model_name: str
) -> pd.DataFrame:
    """将 `bootstrap_ci_for_metrics` 的输出整理为一张长表，便于导出 csv / 拼接多模型结果。

    Parameters
    ----------
    metrics_with_ci : Dict[str, object]
        `bootstrap_ci_for_metrics` 的返回值。
    class_labels : Dict[int, str]
        类别编码到中文名的映射。
    model_name : str
        模型名称，将作为一列写入结果表。

    Returns
    -------
    pd.DataFrame
        列: model, metric, class(仅per-class指标有值), estimate, ci_lower, ci_upper, formatted
    """
    rows = []
    for metric_name in ["accuracy", "macro_auc", "macro_f1", "weighted_f1"]:
        point, lower, upper = metrics_with_ci[metric_name]
        rows.append(
            {
                "model": model_name,
                "metric": metric_name,
                "class": "-",
                "estimate": point,
                "ci_lower": lower,
                "ci_upper": upper,
                "formatted": f"{point:.3f} ({lower:.3f}-{upper:.3f})" if not np.isnan(point) else "NA",
            }
        )

    for c, class_metrics in metrics_with_ci["per_class"].items():
        for metric_name, (point, lower, upper) in class_metrics.items():
            rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "class": class_labels.get(c, str(c)),
                    "estimate": point,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "formatted": f"{point:.3f} ({lower:.3f}-{upper:.3f})" if not np.isnan(point) else "NA",
                }
            )
    return pd.DataFrame(rows)

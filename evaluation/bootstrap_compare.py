# -*- coding: utf-8 -*-
"""
bootstrap_compare.py
======================
测试集 Bootstrap 重采样模型对比：对全部 7 个模型，使用完全相同的 1000 组重采样索引
（配对 Bootstrap），计算各模型在每组重采样下的指标，最终给出：
    - 每个模型的指标点估计 + 95% CI
    - 模型两两比较时，指标差值的 95% CI（用于判断差异是否有统计学意义）

使用相同重采样索引（而非各模型独立重采样）是"配对比较"的关键，
能够消除因重采样本身随机性带来的额外噪声，使模型间比较更加公平、更具功效。
"""

from itertools import combinations
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from evaluation.metrics import compute_point_metrics


def _generate_bootstrap_indices(n_samples: int, n_bootstrap: int, random_state: int) -> List[np.ndarray]:
    """生成 n_bootstrap 组有放回重抽样的索引，所有模型共用，保证配对比较。

    Parameters
    ----------
    n_samples : int
        测试集样本量。
    n_bootstrap : int
        重采样次数。
    random_state : int
        随机种子。

    Returns
    -------
    List[np.ndarray]
        长度为 n_bootstrap 的索引数组列表。
    """
    rng = np.random.RandomState(random_state)
    return [rng.randint(0, n_samples, size=n_samples) for _ in range(n_bootstrap)]


def bootstrap_model_metrics(
    fitted_models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    classes: List[int],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """对每个模型在相同的重采样测试集上计算核心指标，返回汇总表。

    Parameters
    ----------
    fitted_models : Dict[str, Pipeline]
        已训练好的模型字典。
    X_test, y_test : pd.DataFrame, pd.Series
        原始（未重采样）测试集。
    classes : List[int]
        类别列表。
    n_bootstrap : int, default 1000
        重采样次数。
    alpha : float, default 0.05
        显著性水平。
    random_state : int, default 42
        随机种子，保证全部模型使用相同的重采样索引。

    Returns
    -------
    pd.DataFrame
        列: model, accuracy(_lower/_upper), macro_auc(...), macro_f1(...), weighted_f1(...)
        每个模型一行，为测试集 Bootstrap 结果的汇总表（表格 5 的核心内容）。
    """
    y_test_arr = np.asarray(y_test)
    X_test_arr = X_test.reset_index(drop=True)
    n = len(y_test_arr)
    boot_indices = _generate_bootstrap_indices(n, n_bootstrap, random_state)

    # 先算出每个模型对全体测试集的预测，重采样时直接按索引取值，避免重复调用 predict
    base_preds = {}
    for name, model in fitted_models.items():
        y_pred_full = model.predict(X_test_arr)
        y_proba_full = model.predict_proba(X_test_arr)
        base_preds[name] = (np.asarray(y_pred_full), np.asarray(y_proba_full))

    metric_samples = {name: {"accuracy": [], "macro_auc": [], "macro_f1": [], "weighted_f1": []} for name in fitted_models}

    for idx in boot_indices:
        yt = y_test_arr[idx]
        if len(np.unique(yt)) < len(classes):
            continue
        for name in fitted_models:
            y_pred_full, y_proba_full = base_preds[name]
            yp, ypb = y_pred_full[idx], y_proba_full[idx]
            point = compute_point_metrics(yt, yp, ypb, classes)
            for m in ["accuracy", "macro_auc", "macro_f1", "weighted_f1"]:
                metric_samples[name][m].append(point[m])

    rows = []
    for name in fitted_models:
        y_pred_full, y_proba_full = base_preds[name]
        point_est = compute_point_metrics(y_test_arr, y_pred_full, y_proba_full, classes)
        row = {"model": name}
        for m in ["accuracy", "macro_auc", "macro_f1", "weighted_f1"]:
            arr = np.array([v for v in metric_samples[name][m] if not np.isnan(v)])
            lower, upper = (np.percentile(arr, 100 * alpha / 2), np.percentile(arr, 100 * (1 - alpha / 2))) if len(arr) else (np.nan, np.nan)
            row[f"{m}"] = point_est[m]
            row[f"{m}_lower"] = lower
            row[f"{m}_upper"] = upper
            row[f"{m}_formatted"] = f"{point_est[m]:.3f} ({lower:.3f}-{upper:.3f})" if not np.isnan(point_est[m]) else "NA"
        rows.append(row)

    return pd.DataFrame(rows)


def pairwise_metric_difference(
    fitted_models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    classes: List[int],
    metric: str = "macro_auc",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """对模型两两配对比较，计算指定指标差值的 Bootstrap 95% CI，判断差异是否具有统计学意义。

    Parameters
    ----------
    fitted_models : Dict[str, Pipeline]
        已训练好的模型字典。
    X_test, y_test : pd.DataFrame, pd.Series
        测试集。
    classes : List[int]
        类别列表。
    metric : str, default "macro_auc"
        用于比较的指标名，可选 "accuracy", "macro_auc", "macro_f1", "weighted_f1"。
    n_bootstrap : int, default 1000
        重采样次数。
    alpha : float, default 0.05
        显著性水平。
    random_state : int, default 42
        随机种子（与 `bootstrap_model_metrics` 保持一致的重采样索引）。

    Returns
    -------
    pd.DataFrame
        列: model_a, model_b, diff, ci_lower, ci_upper, significant
        若 95% CI 不包含 0，则判定两模型在该指标上存在统计学显著差异。
    """
    y_test_arr = np.asarray(y_test)
    X_test_arr = X_test.reset_index(drop=True)
    n = len(y_test_arr)
    boot_indices = _generate_bootstrap_indices(n, n_bootstrap, random_state)

    base_preds = {}
    for name, model in fitted_models.items():
        base_preds[name] = (np.asarray(model.predict(X_test_arr)), np.asarray(model.predict_proba(X_test_arr)))

    model_names = list(fitted_models.keys())
    diff_samples = {pair: [] for pair in combinations(model_names, 2)}

    for idx in boot_indices:
        yt = y_test_arr[idx]
        if len(np.unique(yt)) < len(classes):
            continue
        scores = {}
        for name in model_names:
            y_pred_full, y_proba_full = base_preds[name]
            yp, ypb = y_pred_full[idx], y_proba_full[idx]
            scores[name] = compute_point_metrics(yt, yp, ypb, classes)[metric]
        for a, b in diff_samples:
            if not (np.isnan(scores[a]) or np.isnan(scores[b])):
                diff_samples[(a, b)].append(scores[a] - scores[b])

    rows = []
    for (a, b), diffs in diff_samples.items():
        arr = np.array(diffs)
        if len(arr) == 0:
            continue
        lower, upper = np.percentile(arr, 100 * alpha / 2), np.percentile(arr, 100 * (1 - alpha / 2))
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "metric": metric,
                "diff_mean": arr.mean(),
                "ci_lower": lower,
                "ci_upper": upper,
                "significant": bool(lower > 0 or upper < 0),
            }
        )
    return pd.DataFrame(rows)

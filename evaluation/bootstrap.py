# -*- coding: utf-8 -*-
"""
bootstrap.py
通用 Bootstrap 重采样置信区间计算模块。

核心思路：模型已经训练/预测完毕后，我们不需要重新训练模型即可估计指标的抽样波动 ——
只需对"测试集样本（含真实标签、预测标签、预测概率）"做有放回重采样 N 次，
每次重新计算一遍指标，最终取经验分布的 2.5% / 97.5% 分位数作为 95% CI。
这是临床预测模型验证中最常用、也最不依赖分布假设的置信区间估计方法。
"""

import numpy as np
import pandas as pd

from config import settings
from evaluation.metrics import compute_all_metrics


def bootstrap_indices(n_samples, n_bootstrap, random_state=None):
    """
    生成 n_bootstrap 组有放回重采样的样本下标。

    Parameters
    ----------
    n_samples : int
        原始样本量。
    n_bootstrap : int
        重采样次数。
    random_state : int, optional

    Returns
    -------
    numpy.ndarray, shape (n_bootstrap, n_samples)
    """
    rng = np.random.default_rng(random_state)
    return rng.integers(0, n_samples, size=(n_bootstrap, n_samples))


def bootstrap_metric_distribution(y_true, y_pred, y_proba, n_bootstrap=None,
                                  classes=None, random_state=None):
    """
    对测试集预测结果做 Bootstrap 重采样，返回每次重采样得到的完整指标字典列表。

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
    y_pred : array-like, shape (n_samples,)
    y_proba : array-like, shape (n_samples, n_classes)
    n_bootstrap : int, optional
        默认使用 settings.N_BOOTSTRAP (1000)。
    classes : list, optional
    random_state : int, optional
        默认使用 settings.RANDOM_STATE。

    Returns
    -------
    pandas.DataFrame
        每行是一次重采样的指标结果，列为各项指标名称。
    """
    n_bootstrap = n_bootstrap or settings.N_BOOTSTRAP
    random_state = settings.RANDOM_STATE if random_state is None else random_state
    classes = classes or settings.CLASS_ORDER

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    n_samples = len(y_true)

    idx_matrix = bootstrap_indices(n_samples, n_bootstrap, random_state)

    records = []
    for idx in idx_matrix:
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
        # 极端情况下某次重采样可能缺失某个类别，AUC 等指标会计算失败，此时跳过该次重采样
        if len(np.unique(yt)) < len(classes):
            continue
        try:
            records.append(compute_all_metrics(yt, yp, ypr, classes))
        except ValueError:
            continue

    return pd.DataFrame(records)


def summarize_ci(distribution_df, alpha=None):
    """
    根据 Bootstrap 得到的指标分布，汇总为 "点估计 (95% CI)" 的展示表。

    Parameters
    ----------
    distribution_df : pandas.DataFrame
        bootstrap_metric_distribution() 的输出。
    alpha : float, optional
        显著性水平，默认使用 settings.CI_ALPHA (0.05 -> 95% CI)。

    Returns
    -------
    pandas.DataFrame
        列：metric, mean, ci_lower, ci_upper, display（形如 "0.912 (0.887-0.935)"）
    """
    alpha = alpha if alpha is not None else settings.CI_ALPHA
    lower_q, upper_q = alpha / 2 * 100, (1 - alpha / 2) * 100

    rows = []
    for metric in distribution_df.columns:
        values = distribution_df[metric].dropna().values
        mean_val = float(np.mean(values))
        lo, hi = np.percentile(values, [lower_q, upper_q])
        rows.append({
            "metric": metric,
            "mean": mean_val,
            "ci_lower": float(lo),
            "ci_upper": float(hi),
            "display": f"{mean_val:.3f} ({lo:.3f}-{hi:.3f})",
        })
    return pd.DataFrame(rows)


def point_estimate_with_bootstrap_ci(y_true, y_pred, y_proba, n_bootstrap=None,
                                     classes=None, random_state=None):
    """
    最常用的对外接口：直接给出"测试集单点估计 + Bootstrap 95% CI"汇总表。
    点估计使用完整测试集（未重采样）计算，CI 上下限来自 Bootstrap 分布，
    这是临床预测模型论文中最常见的报告方式。

    Parameters
    ----------
    y_true, y_pred, y_proba : array-like
    n_bootstrap : int, optional
    classes : list, optional
    random_state : int, optional

    Returns
    -------
    pandas.DataFrame
        列：metric, point_estimate, ci_lower, ci_upper, display
    """
    classes = classes or settings.CLASS_ORDER
    point_estimates = compute_all_metrics(y_true, y_pred, y_proba, classes)

    dist = bootstrap_metric_distribution(
        y_true, y_pred, y_proba, n_bootstrap=n_bootstrap,
        classes=classes, random_state=random_state,
    )
    ci_table = summarize_ci(dist)

    ci_table["point_estimate"] = ci_table["metric"].map(point_estimates)
    ci_table["display"] = ci_table.apply(
        lambda r: f"{r['point_estimate']:.3f} ({r['ci_lower']:.3f}-{r['ci_upper']:.3f})",
        axis=1,
    )
    return ci_table[["metric", "point_estimate", "ci_lower", "ci_upper", "display"]]

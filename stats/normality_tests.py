# -*- coding: utf-8 -*-
"""
normality_tests.py
====================
单因素分析的前置检验：正态性检验与方差齐性检验。
根据检验结果决定后续使用参数检验（单因素方差分析 ANOVA）
还是非参数检验（Kruskal-Wallis 检验）。
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def test_normality(sample: pd.Series, alpha: float = 0.05) -> Dict[str, float]:
    """对单个样本组进行正态性检验。

    样本量 <= 5000 时使用 Shapiro-Wilk 检验（小样本更稳健、更常用于医学统计）；
    样本量 > 5000 时改用 D'Agostino-Pearson 检验（Shapiro-Wilk 在大样本下计算不稳定）。

    Parameters
    ----------
    sample : pd.Series
        单组连续型变量取值（已去除缺失值）。
    alpha : float, default 0.05
        显著性水平。

    Returns
    -------
    Dict[str, float]
        {"method": 使用的检验方法, "statistic": 检验统计量,
         "p_value": P值, "is_normal": 是否服从正态分布(bool)}
    """
    sample = pd.Series(sample).dropna()
    n = len(sample)

    if n < 3:
        # 样本量过小时无法判断正态性，保守起见按非正态处理
        return {"method": "insufficient_n", "statistic": np.nan, "p_value": np.nan, "is_normal": False}

    if n <= 5000:
        stat, p = stats.shapiro(sample)
        method = "Shapiro-Wilk"
    else:
        stat, p = stats.normaltest(sample)
        method = "D'Agostino-Pearson"

    return {"method": method, "statistic": float(stat), "p_value": float(p), "is_normal": bool(p > alpha)}


def test_homogeneity_of_variance(groups: List[pd.Series], alpha: float = 0.05) -> Dict[str, float]:
    """对多组样本进行方差齐性检验（Levene 检验，对非正态分布也稳健）。

    Parameters
    ----------
    groups : List[pd.Series]
        多组样本值列表，每个元素为一组（已去除缺失值）。
    alpha : float, default 0.05
        显著性水平。

    Returns
    -------
    Dict[str, float]
        {"statistic": 检验统计量, "p_value": P值, "is_homogeneous": 方差是否齐(bool)}
    """
    groups = [pd.Series(g).dropna() for g in groups]
    groups = [g for g in groups if len(g) > 1]

    if len(groups) < 2:
        return {"statistic": np.nan, "p_value": np.nan, "is_homogeneous": False}

    stat, p = stats.levene(*groups, center="median")
    return {"statistic": float(stat), "p_value": float(p), "is_homogeneous": bool(p > alpha)}


def decide_test_method(groups: List[pd.Series], alpha: float = 0.05) -> Dict[str, object]:
    """综合正态性与方差齐性检验结果，决定该变量应使用的组间比较方法。

    判定规则：仅当“所有组均服从正态分布”且“方差齐”时，使用单因素方差分析（ANOVA）；
    否则使用 Kruskal-Wallis 秩和检验。

    Parameters
    ----------
    groups : List[pd.Series]
        多组样本值列表。
    alpha : float, default 0.05
        显著性水平。

    Returns
    -------
    Dict[str, object]
        {"normality_results": 各组正态性检验结果列表,
         "homogeneity_result": 方差齐性检验结果,
         "all_normal": 是否所有组均正态(bool),
         "recommended_test": "ANOVA" 或 "Kruskal-Wallis"}
    """
    normality_results = [test_normality(g, alpha=alpha) for g in groups]
    all_normal = all(r["is_normal"] for r in normality_results)
    homogeneity_result = test_homogeneity_of_variance(groups, alpha=alpha)

    if all_normal and homogeneity_result["is_homogeneous"]:
        recommended_test = "ANOVA"
    else:
        recommended_test = "Kruskal-Wallis"

    return {
        "normality_results": normality_results,
        "homogeneity_result": homogeneity_result,
        "all_normal": all_normal,
        "recommended_test": recommended_test,
    }

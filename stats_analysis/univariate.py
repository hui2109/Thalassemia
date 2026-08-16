# -*- coding: utf-8 -*-
"""
univariate.py
单因素分析模块。

流程（遵循临床统计学的标准套路）：
    对每一个连续变量，按 diagnosis 分成 4 组 -->
        1. 正态性检验：各组分别做 Shapiro-Wilk 检验；
        2. 方差齐性检验：Levene 检验；
        3. 若"全部组正态 且 方差齐" --> 单因素方差分析 (One-way ANOVA)；
           否则 --> Kruskal-Wallis H 检验（非参数）。
    最终输出一张包含检验方法、统计量、P 值、是否具有统计学意义的汇总表。
"""

import pandas as pd
from scipy import stats

from config import settings


def _test_normality(groups):
    """
    对每一组数据做 Shapiro-Wilk 正态性检验。

    Parameters
    ----------
    groups : list[numpy.ndarray]
        按类别切分后的若干组数值数据。

    Returns
    -------
    bool
        True 表示所有组都不拒绝正态性假设 (p > 0.05)，即可认为整体近似正态。
    """
    for g in groups:
        # 样本量过少时 Shapiro 检验不稳定，n>=3 才检验；n>5000 时 Shapiro 不适用（本数据集无需考虑）
        if len(g) < 3:
            continue
        _, p = stats.shapiro(g)
        if p <= 0.05:
            return False
    return True


def _test_homogeneity(groups):
    """
    Levene 检验方差齐性。

    Parameters
    ----------
    groups : list[numpy.ndarray]

    Returns
    -------
    bool
        True 表示方差齐（p > 0.05）。
    """
    _, p = stats.levene(*groups)
    return p > 0.05


def analyze_single_variable(df, var, group_col=None):
    """
    对单个连续变量在多个诊断组之间做完整的单因素分析。

    Parameters
    ----------
    df : pandas.DataFrame
        清洗后的数据。
    var : str
        待检验的变量名。
    group_col : str, optional
        分组变量名，默认使用 settings.TARGET_COL。

    Returns
    -------
    dict
        包含变量名、各组正态性结果、方差齐性结果、所用检验方法、
        检验统计量、P 值、是否显著（P<0.05）。
    """
    group_col = group_col or settings.TARGET_COL
    # 分组取值直接从数据中动态获取，而不是写死 settings.CLASS_ORDER —— 这样本函数
    # 既能用于按 diagnosis（0/1/2/3）分组，也能复用于按"训练集/测试集"等任意分组比较。
    group_values = sorted(df[group_col].dropna().unique(), key=str)
    groups = [
        df.loc[df[group_col] == g, var].dropna().values
        for g in group_values
    ]

    is_normal = _test_normality(groups)
    is_homogeneous = _test_homogeneity(groups)

    if is_normal and is_homogeneous:
        method = "One-way ANOVA"
        statistic, p_value = stats.f_oneway(*groups)
    else:
        method = "Kruskal-Wallis H"
        statistic, p_value = stats.kruskal(*groups)

    return {
        "variable": var,
        "normal": is_normal,
        "homogeneous_variance": is_homogeneous,
        "test_method": method,
        "statistic": round(float(statistic), 4),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def run_univariate_analysis(df, variables, group_col=None):
    """
    对一组变量批量执行单因素分析，返回汇总表。

    Parameters
    ----------
    df : pandas.DataFrame
        清洗后的数据。
    variables : list[str]
        待检验的变量名列表（通常是全部候选特征，包括原始指标 + 9 个衍生指标）。
    group_col : str, optional
        分组变量名。

    Returns
    -------
    pandas.DataFrame
        汇总表，按 p_value 升序排列，列包括：
        variable, normal, homogeneous_variance, test_method, statistic, p_value, significant
    """
    records = [analyze_single_variable(df, var, group_col) for var in variables]
    result = pd.DataFrame(records).sort_values("p_value").reset_index(drop=True)
    result["p_value_display"] = result["p_value"].apply(_format_p_value)
    return result


def _format_p_value(p):
    """
    统计学论文常见的 P 值展示格式（<0.001 时不显示具体数值）。

    Parameters
    ----------
    p : float

    Returns
    -------
    str
    """
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

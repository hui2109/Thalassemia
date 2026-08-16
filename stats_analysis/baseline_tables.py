# -*- coding: utf-8 -*-
"""
baseline_tables.py
生成论文/报告中常见的"基线特征对照表"（Table 1 风格）。

包含两类表：
    1. 多组基线对照表：例如 4 个诊断分组之间的基线特征比较（含单因素分析 P 值）；
    2. 两组基线对照表：例如训练集 vs 测试集，用于证明分层抽样后两组特征分布一致，
       不存在因划分导致的系统性偏倚。

描述性统计的展示形式遵循统计学惯例：
    - 服从正态分布 --> 均数 ± 标准差，mean (SD)
    - 不服从正态分布 --> 中位数 [四分位间距]，median [Q1, Q3]
是否正态直接复用 stats_analysis.univariate 中对每个变量做出的判断，保证全文口径一致。
"""

import numpy as np
import pandas as pd

from config import settings
from stats_analysis.univariate import run_univariate_analysis, _test_normality


def _format_descriptive(series):
    """
    根据正态性自动选择 mean(SD) 或 median[Q1,Q3] 的展示格式。

    Parameters
    ----------
    series : pandas.Series
        某一组内的数值。

    Returns
    -------
    tuple(str, bool)
        (格式化字符串, 是否按正态分布展示)
    """
    values = series.dropna().values
    is_normal = _test_normality([values]) if len(values) >= 3 else True
    if is_normal:
        return f"{np.mean(values):.2f} ({np.std(values, ddof=1):.2f})", True
    q1, q2, q3 = np.percentile(values, [25, 50, 75])
    return f"{q2:.2f} [{q1:.2f}, {q3:.2f}]", False


def build_group_baseline_table(df, variables, group_col=None, group_labels=None):
    """
    构建"多组基线特征对照表"（如 4 个诊断分组）。

    Parameters
    ----------
    df : pandas.DataFrame
        清洗后的数据。
    variables : list[str]
        需要纳入表格的变量名（一般为全部候选特征，含 9 个衍生指标供参考）。
    group_col : str, optional
        分组变量名，默认使用 settings.TARGET_COL。
    group_labels : dict, optional
        分组编码到展示名称的映射，默认使用 settings.CLASS_LABELS。

    Returns
    -------
    pandas.DataFrame
        行 = 变量，列 = 各组描述统计 + 检验方法 + P 值。
    """
    group_col = group_col or settings.TARGET_COL
    group_labels = group_labels or settings.CLASS_LABELS
    groups = sorted(df[group_col].unique())

    univariate_result = run_univariate_analysis(df, variables, group_col=group_col)
    univariate_result = univariate_result.set_index("variable")

    rows = []
    for var in variables:
        row = {"Variable": var}
        for g in groups:
            col_name = group_labels.get(g, str(g))
            desc, _ = _format_descriptive(df.loc[df[group_col] == g, var])
            row[f"{col_name} (n={int((df[group_col] == g).sum())})"] = desc
        row["Test Method"] = univariate_result.loc[var, "test_method"]
        row["P-value"] = univariate_result.loc[var, "p_value_display"]
        row["Significant"] = "Yes" if univariate_result.loc[var, "significant"] else "No"
        rows.append(row)

    return pd.DataFrame(rows)


def build_split_baseline_table(train_df, test_df, variables, target_col=None):
    """
    构建"训练集 vs 测试集"基线特征对比表，用于验证分层随机划分后两组特征分布均衡。

    Parameters
    ----------
    train_df : pandas.DataFrame
        训练集（含特征列与标签列）。
    test_df : pandas.DataFrame
        测试集（含特征列与标签列）。
    variables : list[str]
        需要比较的变量名列表。
    target_col : str, optional
        诊断标签列名，默认使用 settings.TARGET_COL，会一并纳入比较（以类别构成比展示）。

    Returns
    -------
    pandas.DataFrame
        行 = 变量，列 = 训练集描述 / 测试集描述 / 检验方法 / P 值。
    """
    target_col = target_col or settings.TARGET_COL

    combined = pd.concat(
        [train_df.assign(__split__="Train"), test_df.assign(__split__="Test")],
        ignore_index=True,
    )

    univariate_result = run_univariate_analysis(
        combined, variables, group_col="__split__"
    ).set_index("variable")

    rows = []
    for var in variables:
        desc_train, _ = _format_descriptive(train_df[var])
        desc_test, _ = _format_descriptive(test_df[var])
        rows.append({
            "Variable": var,
            f"Train (n={len(train_df)})": desc_train,
            f"Test (n={len(test_df)})": desc_test,
            "Test Method": univariate_result.loc[var, "test_method"],
            "P-value": univariate_result.loc[var, "p_value_display"],
        })

    # Diagnosis group composition comparison (categorical variable, Chi-square test)
    from scipy.stats import chi2_contingency

    class_table = pd.crosstab(combined["__split__"], combined[target_col])
    chi2, p_chi2, _, _ = chi2_contingency(class_table)
    for cls in sorted(combined[target_col].unique()):
        label = settings.CLASS_LABELS.get(cls, str(cls))
        n_train = (train_df[target_col] == cls).sum()
        n_test = (test_df[target_col] == cls).sum()
        rows.append({
            "Variable": f"Diagnosis Group - {label}",
            f"Train (n={len(train_df)})": f"{n_train} ({n_train / len(train_df):.1%})",
            f"Test (n={len(test_df)})": f"{n_test} ({n_test / len(test_df):.1%})",
            "Test Method": "Chi-square" if cls == sorted(combined[target_col].unique())[0] else "",
            "P-value": (f"{p_chi2:.3f}" if p_chi2 >= 0.001 else "<0.001") if cls == sorted(combined[target_col].unique())[0] else "",
        })

    return pd.DataFrame(rows)

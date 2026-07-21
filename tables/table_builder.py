# -*- coding: utf-8 -*-
"""
table_builder.py
==================
汇总生成报告所需的各类表格：
    表1：三分类基线特征对照表（含单因素分析 P 值）
    表2：训练集与测试集基线特征对比表
    表3：训练集（5折交叉验证）与测试集 7 模型性能对比表
    表4：测试集 Bootstrap 重采样 1000 次后的模型性能对比表（由 evaluation.bootstrap_compare 提供数据源）

本模块只负责"整理、拼表、格式化"，具体的统计检验逻辑复用 stats 模块，
指标计算复用 evaluation 模块，保持职责单一。
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from stats.normality_tests import decide_test_method
from stats.univariate_analysis import _describe_group


def build_baseline_table(univariate_result_df: pd.DataFrame, class_labels: Dict[int, str]) -> pd.DataFrame:
    """整理单因素分析结果为对客户友好的三组基线特征对照表。

    Parameters
    ----------
    univariate_result_df : pd.DataFrame
        `stats.univariate_analysis.UnivariateAnalyzer.run` 的返回结果。
    class_labels : Dict[int, str]
        类别编码到中文名映射（用于确定列顺序）。

    Returns
    -------
    pd.DataFrame
        列: 变量, <各组标签>, 检验方法, 统计量, P值, FDR校正P值
    """
    group_cols = [class_labels[c] for c in sorted(class_labels.keys())]
    ordered_cols = ["variable"] + group_cols + ["test_method", "statistic", "p_value", "p_value_fdr"]
    table = univariate_result_df[ordered_cols].copy()
    table.columns = ["变量"] + group_cols + ["检验方法", "统计量", "P值", "FDR校正P值"]
    table["P值"] = table["P值"].apply(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")
    table["FDR校正P值"] = table["FDR校正P值"].apply(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")
    table["统计量"] = table["统计量"].round(3)
    return table


def build_train_test_comparison_table(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_cols: List[str],
    target_col: str,
    class_labels: Dict[int, str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """生成训练集与测试集的基线特征对比表，验证分层随机划分后两组是否均衡可比。

    连续变量：按正态性/方差齐性自动选择 独立样本 t 检验 或 Mann-Whitney U 检验；
    结局分布（诊断类别构成比）：使用卡方检验比较训练集与测试集是否同分布。

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        训练集/测试集特征矩阵。
    y_train, y_test : pd.Series
        训练集/测试集标签。
    feature_cols : List[str]
        特征列名列表。
    target_col : str
        结局列名（仅用于表格展示行名）。
    class_labels : Dict[int, str]
        类别编码到中文名映射。
    alpha : float, default 0.05
        显著性水平。

    Returns
    -------
    pd.DataFrame
        列: 变量, 训练集, 测试集, 检验方法, 统计量, P值
    """
    rows = []
    for var in feature_cols:
        s_train, s_test = X_train[var].dropna(), X_test[var].dropna()
        decision = decide_test_method([s_train, s_test], alpha=alpha)

        if decision["recommended_test"] == "ANOVA":
            stat, p = stats.ttest_ind(s_train, s_test, equal_var=True)
            method = "独立样本 t 检验"
            display_normal = True
        else:
            stat, p = stats.mannwhitneyu(s_train, s_test, alternative="two-sided")
            method = "Mann-Whitney U 检验"
            display_normal = False

        rows.append(
            {
                "变量": var,
                "训练集": _describe_group(s_train, is_normal=display_normal),
                "测试集": _describe_group(s_test, is_normal=display_normal),
                "检验方法": method,
                "统计量": round(float(stat), 3),
                "P值": "<0.001" if p < 0.001 else f"{p:.3f}",
            }
        )

    # 结局类别构成比对比（卡方检验）
    train_counts = y_train.value_counts().reindex(sorted(class_labels.keys()), fill_value=0)
    test_counts = y_test.value_counts().reindex(sorted(class_labels.keys()), fill_value=0)
    contingency = np.array([train_counts.values, test_counts.values])
    chi2, p_chi2, _, _ = stats.chi2_contingency(contingency)

    train_dist = ", ".join(f"{class_labels[c]} {train_counts[c]}({train_counts[c]/train_counts.sum()*100:.1f}%)" for c in class_labels)
    test_dist = ", ".join(f"{class_labels[c]} {test_counts[c]}({test_counts[c]/test_counts.sum()*100:.1f}%)" for c in class_labels)

    rows.append(
        {
            "变量": f"{target_col} (类别构成)",
            "训练集": train_dist,
            "测试集": test_dist,
            "检验方法": "卡方检验",
            "统计量": round(float(chi2), 3),
            "P值": "<0.001" if p_chi2 < 0.001 else f"{p_chi2:.3f}",
        }
    )

    return pd.DataFrame(rows)


def build_train_vs_test_model_table(
    train_cv_metrics: Dict[str, Dict[str, float]],
    test_metrics: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    """生成训练集（交叉验证均值）与测试集的 7 模型性能对比表。

    Parameters
    ----------
    train_cv_metrics : Dict[str, Dict[str, float]]
        {模型名: {"accuracy":.., "macro_auc":.., "macro_f1":.., "weighted_f1":..}}，
        取自训练集 5 折交叉验证的均值（见 main.py 中 cross_val_score 汇总）。
    test_metrics : Dict[str, Dict[str, object]]
        {模型名: {"accuracy": (point, lower, upper), "macro_auc": (...), ...}}，
        取自 `evaluation.metrics.bootstrap_ci_for_metrics` 的输出。

    Returns
    -------
    pd.DataFrame
        列: 模型, 训练集准确率(5折CV均值), 测试集准确率(95%CI), ... 依次类推四个核心指标。
    """
    rows = []
    for name in train_cv_metrics:
        row = {"模型": name}
        for metric_key, metric_cn in [
            ("accuracy", "准确率"),
            ("macro_auc", "宏平均AUC"),
            ("macro_f1", "宏平均F1"),
            ("weighted_f1", "加权F1"),
        ]:
            row[f"训练集{metric_cn}(5折CV均值)"] = f"{train_cv_metrics[name][metric_key]:.3f}"
            point, lower, upper = test_metrics[name][metric_key]
            row[f"测试集{metric_cn}(95%CI)"] = f"{point:.3f} ({lower:.3f}-{upper:.3f})"
        rows.append(row)
    return pd.DataFrame(rows)


def build_bootstrap_summary_table(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    """将 `evaluation.bootstrap_compare.bootstrap_model_metrics` 的原始输出整理为报告展示格式。

    Parameters
    ----------
    bootstrap_df : pd.DataFrame
        `bootstrap_model_metrics` 的返回值。

    Returns
    -------
    pd.DataFrame
        列: 模型, 准确率(95%CI), 宏平均AUC(95%CI), 宏平均F1(95%CI), 加权F1(95%CI)
    """
    display_cols = {
        "model": "模型",
        "accuracy_formatted": "准确率(95%CI)",
        "macro_auc_formatted": "宏平均AUC(95%CI)",
        "macro_f1_formatted": "宏平均F1(95%CI)",
        "weighted_f1_formatted": "加权F1(95%CI)",
    }
    return bootstrap_df[list(display_cols.keys())].rename(columns=display_cols)

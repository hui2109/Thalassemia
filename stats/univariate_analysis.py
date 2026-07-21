# -*- coding: utf-8 -*-
"""
univariate_analysis.py
========================
单因素分析模块：对每个候选特征，在三组（对照组/静止型/非静止型）之间做组间比较，
自动根据正态性 + 方差齐性检验结果选择 ANOVA 或 Kruskal-Wallis 检验，
并输出每组的描述性统计（正态用 均数±标准差，非正态用 中位数[四分位间距]）。

不依赖 statsmodels，多重比较校正（Benjamini-Hochberg FDR）在本模块内手工实现，
减少客户环境的第三方依赖。
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from stats.normality_tests import decide_test_method


def _describe_group(sample: pd.Series, is_normal: bool) -> str:
    """生成单组的描述性统计字符串。

    Parameters
    ----------
    sample : pd.Series
        该组该变量的取值。
    is_normal : bool
        该组是否服从正态分布，决定展示 均数±SD 还是 中位数[IQR]。

    Returns
    -------
    str
        格式化后的描述统计字符串。
    """
    sample = pd.Series(sample).dropna()
    if is_normal:
        return f"{sample.mean():.2f}±{sample.std():.2f}"
    q1, q3 = sample.quantile([0.25, 0.75])
    return f"{sample.median():.2f}[{q1:.2f}, {q3:.2f}]"


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg 多重比较校正（控制假发现率 FDR），手工实现避免额外依赖。

    Parameters
    ----------
    p_values : List[float]
        原始 P 值列表。

    Returns
    -------
    List[float]
        校正后的 P 值列表（与输入顺序一一对应），值域截断在 [0, 1]。
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    valid_mask = ~np.isnan(p)
    adjusted = np.full(n, np.nan)

    valid_p = p[valid_mask]
    if len(valid_p) == 0:
        return adjusted.tolist()

    order = np.argsort(valid_p)
    ranked = valid_p[order]
    m = len(ranked)
    bh_values = ranked * m / (np.arange(1, m + 1))
    # 保证单调不减（从后往前取累计最小值）
    bh_values = np.minimum.accumulate(bh_values[::-1])[::-1]
    bh_values = np.clip(bh_values, 0, 1)

    result_valid = np.empty(m)
    result_valid[order] = bh_values
    adjusted[valid_mask] = result_valid
    return adjusted.tolist()


class UnivariateAnalyzer:
    """三组间单因素分析器。

    典型用法
    --------
    >>> analyzer = UnivariateAnalyzer(group_col="diagnosis", class_labels={0: "对照组", 1: "静止型", 2: "非静止型"})
    >>> result_df = analyzer.run(df, feature_cols=["age", "RBC", "HB"])
    """

    def __init__(self, group_col: str, class_labels: Dict[int, str], alpha: float = 0.05):
        """
        Parameters
        ----------
        group_col : str
            分组列名（即结局列，如 'diagnosis'）。
        class_labels : Dict[int, str]
            类别编码到中文标签的映射，用于表格展示。
        alpha : float, default 0.05
            显著性水平。
        """
        self.group_col = group_col
        self.class_labels = class_labels
        self.alpha = alpha

    def analyze_one_variable(self, df: pd.DataFrame, var: str) -> Dict[str, object]:
        """对单个变量进行组间比较分析。

        Parameters
        ----------
        df : pd.DataFrame
            包含该变量与分组列的数据框。
        var : str
            待分析的变量名。

        Returns
        -------
        Dict[str, object]
            分析结果字典，包含各组描述统计、所用检验方法、统计量与 P 值。
        """
        groups_order = sorted(self.class_labels.keys())
        group_samples = [df.loc[df[self.group_col] == g, var].dropna() for g in groups_order]

        decision = decide_test_method(group_samples, alpha=self.alpha)
        is_normal_per_group = [r["is_normal"] for r in decision["normality_results"]]

        if decision["recommended_test"] == "ANOVA":
            stat, p_value = stats.f_oneway(*group_samples)
        else:
            stat, p_value = stats.kruskal(*group_samples)

        # 描述统计：只要该检验采用非参数方法，三组统一展示中位数[IQR]，保证同一变量在表格里格式一致
        use_normal_display = decision["recommended_test"] == "ANOVA"
        group_desc = {
            self.class_labels[g]: _describe_group(s, is_normal=use_normal_display)
            for g, s in zip(groups_order, group_samples)
        }

        return {
            "variable": var,
            **group_desc,
            "test_method": decision["recommended_test"],
            "statistic": float(stat),
            "p_value": float(p_value),
            "homogeneity_p": decision["homogeneity_result"]["p_value"],
            "all_groups_normal": decision["all_normal"],
        }

    def run(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """对多个变量批量执行单因素分析，并附加 FDR 校正后的 P 值。

        Parameters
        ----------
        df : pd.DataFrame
            完整数据框。
        feature_cols : List[str]
            待分析的特征列名列表。

        Returns
        -------
        pd.DataFrame
            汇总结果表，每行一个变量，按原始 P 值升序排列。
        """
        records = [self.analyze_one_variable(df, var) for var in feature_cols]
        result_df = pd.DataFrame(records)
        result_df["p_value_fdr"] = benjamini_hochberg(result_df["p_value"].tolist())
        result_df["significant_raw"] = result_df["p_value"] < self.alpha
        result_df["significant_fdr"] = result_df["p_value_fdr"] < self.alpha
        result_df = result_df.sort_values("p_value").reset_index(drop=True)
        return result_df

    def get_significant_features(self, result_df: pd.DataFrame, use_fdr: bool = False) -> List[str]:
        """从分析结果中提取具有统计学差异的特征列表。

        Parameters
        ----------
        result_df : pd.DataFrame
            `run` 方法返回的结果表。
        use_fdr : bool, default False
            是否使用 FDR 校正后的 P 值作为筛选标准；默认使用原始 P 值（<0.05）。

        Returns
        -------
        List[str]
            具有统计学差异的特征名列表。
        """
        col = "significant_fdr" if use_fdr else "significant_raw"
        return result_df.loc[result_df[col], "variable"].tolist()

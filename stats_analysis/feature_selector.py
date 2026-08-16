# -*- coding: utf-8 -*-
"""
feature_selector.py
从 9 个衍生地贫指数中挑选出 2~3 个最终入模的指标。

挑选逻辑：
    1. 依据单因素分析的 P 值（越小越具统计学意义）对 9 个衍生指标排序；
    2. 依次遍历排序后的指标，若某指标与"已挑选指标"的 Pearson 相关系数
       绝对值超过阈值（默认 0.9），则认为信息高度重复，予以剔除；
    3. 直至凑够 settings.N_DERIVED_INDEX_TO_KEEP 个指标，或候选列表遍历完毕。

这样既保证了入选指标本身具有统计学意义，又避免了多重共线性对模型（尤其是 LR）的影响。
"""

import pandas as pd

from config import settings


def select_derived_indices(df, univariate_result, max_keep=None, corr_threshold=None):
    """
    从 9 个衍生指标候选集中挑出最终入模的指标。

    Parameters
    ----------
    df : pandas.DataFrame
        清洗后的完整数据（用于计算指标间相关性）。
    univariate_result : pandas.DataFrame
        univariate.run_univariate_analysis() 的输出，须包含 variable 与 p_value 两列。
    max_keep : int, optional
        最多保留的衍生指标个数，默认使用 settings.N_DERIVED_INDEX_TO_KEEP。
    corr_threshold : float, optional
        判定"信息重复"的相关系数阈值，默认使用 settings.DERIVED_INDEX_CORR_THRESHOLD。

    Returns
    -------
    dict
        {
            "selected": list[str]                 # 最终入选的衍生指标
            "selection_log": pandas.DataFrame      # 每个候选指标的取舍过程记录，便于写入报告/复核
        }
    """
    max_keep = max_keep or settings.N_DERIVED_INDEX_TO_KEEP
    corr_threshold = corr_threshold or settings.DERIVED_INDEX_CORR_THRESHOLD

    candidates = settings.DERIVED_INDEX_CANDIDATES
    ranked = (
        univariate_result[univariate_result["variable"].isin(candidates)]
        .sort_values("p_value")
        .reset_index(drop=True)
    )

    corr_matrix = df[candidates].corr(method="pearson").abs()

    selected = []
    log_rows = []
    for _, row in ranked.iterrows():
        var = row["variable"]
        if len(selected) >= max_keep:
            log_rows.append({"variable": var, "p_value": row["p_value"],
                             "kept": False, "reason": "Reached maximum number of selected indices"})
            continue

        redundant_with = [
            s for s in selected if corr_matrix.loc[var, s] >= corr_threshold
        ]
        if redundant_with:
            log_rows.append({
                "variable": var, "p_value": row["p_value"], "kept": False,
                "reason": f"Highly correlated (|r|>={corr_threshold}) with selected index {redundant_with}, redundant",
            })
        else:
            selected.append(var)
            log_rows.append({"variable": var, "p_value": row["p_value"],
                             "kept": True, "reason": "Statistically significant and independent, selected"})

    selection_log = pd.DataFrame(log_rows)
    return {"selected": selected, "selection_log": selection_log}


def build_final_feature_list(selected_derived_indices):
    """
    拼接最终入模特征列表 = 全部原始血常规指标 + 挑选出的衍生指标。

    Parameters
    ----------
    selected_derived_indices : list[str]
        select_derived_indices() 返回的 selected 列表。

    Returns
    -------
    list[str]
        最终入模的完整特征列表。
    """
    return list(settings.BASE_FEATURES) + list(selected_derived_indices)

# -*- coding: utf-8 -*-
"""
loader.py
负责数据的读取与基础清洗，是整个流水线的第一步。
只做"脏活累活"（编码、去空格、类型转换、缺失值检查），不做统计分析或建模，
以便后续任何一步出问题时都能快速定位到具体环节。
"""

import pandas as pd

from config import settings


def load_raw_data(path=None):
    """
    读取原始 CSV 数据。

    Parameters
    ----------
    path : str or Path, optional
        CSV 文件路径。默认使用 settings.DATA_PATH。

    Returns
    -------
    pandas.DataFrame
        原始数据（含全部原始列、衍生指标列、诊断标签列）。
    """
    path = path or settings.DATA_PATH
    # utf-8-sig 用于自动剔除 Excel 导出 csv 常见的 BOM 头
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_data(df):
    """
    对原始数据做基础清洗：
        1. 去除全部字符串字段中的多余空格；
        2. 将建模相关列强制转换为数值类型；
        3. 丢弃编号列（对预测无意义）；
        4. 检查并报告缺失值情况（本数据集经确认无缺失，但保留通用性）。

    Parameters
    ----------
    df : pandas.DataFrame
        load_raw_data() 返回的原始数据。

    Returns
    -------
    pandas.DataFrame
        清洗后的数据，仅保留分析所需列。
    """
    df = df.copy()

    id_like_cols = [c for c in df.columns if c in ("编号", "id", "ID", "No", "No.")]
    df = df.drop(columns=id_like_cols, errors="ignore")

    numeric_cols = (
            settings.BASE_FEATURES
            + settings.DERIVED_INDEX_CANDIDATES
            + [settings.TARGET_COL]
    )
    for col in numeric_cols:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_missing = df[numeric_cols].isna().sum().sum()
    if n_missing > 0:
        print(f"[WARN] {n_missing} missing values remain after cleaning; affected rows have been dropped.")
        df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    df[settings.TARGET_COL] = df[settings.TARGET_COL].astype(int)
    return df


def load_clean_dataset(path=None):
    """
    一步到位：读取 + 清洗。是对外暴露的主入口函数。

    Parameters
    ----------
    path : str or Path, optional
        CSV 文件路径。

    Returns
    -------
    pandas.DataFrame
        可直接用于统计分析和建模的干净数据。
    """
    return clean_data(load_raw_data(path))

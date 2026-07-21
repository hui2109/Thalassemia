# -*- coding: utf-8 -*-
"""
data_loader.py
===============
负责原始数据的读取、基础校验与拆分为特征矩阵 / 标签向量。
本模块只做“读进来、看看干不干净”，不涉及任何统计检验或建模逻辑，
方便客户后续替换数据源（例如改成读数据库）而不影响下游代码。
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd


def load_raw_data(csv_path: Path, encoding: str = "utf-8-sig") -> pd.DataFrame:
    """读取原始 csv 数据。

    Parameters
    ----------
    csv_path : Path
        csv 文件路径。
    encoding : str, default "utf-8-sig"
        文件编码。默认按带 BOM 的 utf-8 读取（Excel 导出的 csv 常见此情况）。

    Returns
    -------
    pd.DataFrame
        原始数据框，列名已去除首尾空白。

    Raises
    ------
    FileNotFoundError
        文件不存在时抛出。
    ValueError
        数据为空时抛出。
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {csv_path}")

    df = pd.read_csv(csv_path, encoding=encoding)
    df.columns = [str(c).strip() for c in df.columns]

    if df.empty:
        raise ValueError(f"数据文件为空: {csv_path}")

    return df


def validate_data(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> None:
    """对读入的数据做基本合法性校验，发现问题直接抛异常，尽早暴露数据质量问题。

    Parameters
    ----------
    df : pd.DataFrame
        待校验数据框。
    target_col : str
        结局列名。
    feature_cols : List[str]
        特征列名列表。

    Returns
    -------
    None
    """
    missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"数据中缺少以下列: {missing_cols}")

    if df[target_col].isna().any():
        raise ValueError(f"结局列 '{target_col}' 存在缺失值，请先处理。")

    n_missing_features = df[feature_cols].isna().sum()
    total_missing = int(n_missing_features.sum())
    if total_missing > 0:
        print(f"[提示] 特征列中共发现 {total_missing} 个缺失值，将在预处理阶段处理:\n"
              f"{n_missing_features[n_missing_features > 0]}")


def get_features_and_target(
    df: pd.DataFrame, feature_cols: List[str], target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """从数据框中拆分出特征矩阵 X 与标签向量 y。

    Parameters
    ----------
    df : pd.DataFrame
        完整数据框。
    feature_cols : List[str]
        特征列名列表。
    target_col : str
        结局列名。

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y)，X 为特征矩阵（float 类型），y 为标签向量（int 类型）。
    """
    X = df[feature_cols].astype(float).copy()
    y = df[target_col].astype(int).copy()
    return X, y

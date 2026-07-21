# -*- coding: utf-8 -*-
"""
data_split.py
==============
负责训练集/测试集的分层随机划分，以及构建标准化预处理器。

设计说明：
- LR / SVC / MLP 等对特征尺度敏感的模型需要标准化；
- Decision Tree / Random Forest / XGBoost / LightGBM 等树模型对尺度不敏感，
  为保持树模型的可解释性（分裂阈值即原始单位），不对其做标准化。
- 因此标准化被封装为独立的 `build_scaler`，由每个模型自己的 Pipeline 决定是否使用，
  而不是在数据层面统一处理，避免信息泄漏也便于灵活调整。
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """按结局类别分层，随机划分训练集与测试集。

    Parameters
    ----------
    X : pd.DataFrame
        特征矩阵。
    y : pd.Series
        标签向量。
    test_size : float
        测试集比例，如 0.3 表示 7:3 划分。
    random_state : int
        随机种子，保证可复现。

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)，索引保持原始 df 索引以便溯源。
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_scaler() -> StandardScaler:
    """构建一个新的标准化器实例。

    Returns
    -------
    StandardScaler
        未拟合的 sklearn StandardScaler 实例，供 Pipeline 内部使用
        （在交叉验证/训练集上 fit，测试集上仅 transform，避免数据泄漏）。
    """
    return StandardScaler()

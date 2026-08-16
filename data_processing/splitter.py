# -*- coding: utf-8 -*-
"""
splitter.py
负责按 7:3 比例做分层随机划分，保证训练集/测试集中四个类别的比例与全量数据保持一致。
"""

from sklearn.model_selection import train_test_split

from config import settings


def stratified_split(df, feature_cols, target_col=None, test_size=None, random_state=None):
    """
    对数据集做分层随机划分（训练集 / 测试集）。

    Parameters
    ----------
    df : pandas.DataFrame
        清洗后的完整数据集。
    feature_cols : list[str]
        最终入模的特征列名列表。
    target_col : str, optional
        标签列名，默认使用 settings.TARGET_COL。
    test_size : float, optional
        测试集比例，默认使用 settings.TEST_SIZE (0.3)。
    random_state : int, optional
        随机种子，默认使用 settings.RANDOM_STATE。

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test) —— 均为 pandas 对象，
        其中 X_* 仅保留 feature_cols 对应的列。
    """
    target_col = target_col or settings.TARGET_COL
    test_size = settings.TEST_SIZE if test_size is None else test_size
    random_state = settings.RANDOM_STATE if random_state is None else random_state

    X = df[feature_cols].reset_index(drop=True)
    y = df[target_col].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )

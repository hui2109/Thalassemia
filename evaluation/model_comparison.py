# -*- coding: utf-8 -*-
"""
model_comparison.py
汇总多个模型的表现，生成两张核心对比表：
    1. 训练集 vs 测试集 7 模型对比表 —— 用于直观判断每个模型是否存在过拟合；
    2. 测试集 Bootstrap（1000 次）重采样后的 7 模型对比表 —— 每个指标均带 95% CI，
       是模型最终横向比较、选出"最优模型"的主要依据。
"""

import pandas as pd

from config import settings
from evaluation.bootstrap import point_estimate_with_bootstrap_ci
from evaluation.metrics import compute_all_metrics

METRIC_DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "macro_auc": "Macro AUC",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "ppv": "PPV",
    "npv": "NPV",

    "acc_per_class_0": 'Accuracy_0',
    "auc_per_class_0": 'AUC_0',
    "f1_per_class_0": 'F1_0',
    "sensitivity_0": 'Sensitivity_0',
    "specificity_0": 'Specificity_0',
    "ppv_0": 'PPV_0',
    "npv_0": 'NPV_0',

    "acc_per_class_1": 'Accuracy_1',
    "auc_per_class_1": 'AUC_1',
    "f1_per_class_1": 'F1_1',
    "sensitivity_1": 'Sensitivity_1',
    "specificity_1": 'Specificity_1',
    "ppv_1": 'PPV_1',
    "npv_1": 'NPV_1',

    "acc_per_class_2": 'Accuracy_2',
    "auc_per_class_2": 'AUC_2',
    "f1_per_class_2": 'F1_2',
    "sensitivity_2": 'Sensitivity_2',
    "specificity_2": 'Specificity_2',
    "ppv_2": 'PPV_2',
    "npv_2": 'NPV_2',

    "acc_per_class_3": 'Accuracy_3',
    "auc_per_class_3": 'AUC_3',
    "f1_per_class_3": 'F1_3',
    "sensitivity_3": 'Sensitivity_3',
    "specificity_3": 'Specificity_3',
    "ppv_3": 'PPV_3',
    "npv_3": 'NPV_3',
}


def _predict_proba_ordered(estimator, X, classes=None):
    """
    获取按 settings.CLASS_ORDER 顺序排列的预测概率矩阵。
    （sklearn 的 predict_proba 列顺序遵循 estimator.classes_，需要重新对齐。）

    Parameters
    ----------
    estimator : 已训练好的 sklearn 兼容分类器（或 Pipeline）
    X : array-like
    classes : list, optional

    Returns
    -------
    numpy.ndarray, shape (n_samples, n_classes)
    """
    classes = classes or settings.CLASS_ORDER
    proba = estimator.predict_proba(X)
    col_order = [list(estimator.classes_).index(c) for c in classes]
    return proba[:, col_order]


def collect_test_predictions(trainers, X_test, y_test):
    """
    统一收集每个模型在测试集上的预测结果，供后续所有可视化函数（ROC / 校准曲线 /
    决策曲线 / 混淆矩阵 / SHAP）复用，避免重复调用 predict / predict_proba。

    Parameters
    ----------
    trainers : dict[str, models.trainer.ModelTrainer]
    X_test, y_test : pandas.DataFrame / Series

    Returns
    -------
    dict[str, dict]
        模型展示名 -> {"y_true":.., "y_pred":.., "y_proba":..}
    """
    results = {}
    for name in settings.MODEL_ORDER:
        if name not in trainers:
            continue
        model = trainers[name].best_estimator_
        display_name = settings.MODEL_DISPLAY_NAMES[name]
        results[display_name] = {
            "y_true": y_test.values if hasattr(y_test, "values") else y_test,
            "y_pred": model.predict(X_test),
            "y_proba": _predict_proba_ordered(model, X_test),
        }
    return results


def build_train_test_performance_table(trainers, X_train, y_train, X_test, y_test):
    """
    生成"训练集 vs 测试集 7 模型对比表"。

    Parameters
    ----------
    trainers : dict[str, models.trainer.ModelTrainer]
        train_all_models() 的输出。
    X_train, y_train, X_test, y_test : pandas.DataFrame / Series
        划分好的训练集与测试集。

    Returns
    -------
    pandas.DataFrame
        每行 = 一个模型，列为 训练集_指标 / 测试集_指标 交替排列。
    """
    rows = []
    for name in settings.MODEL_ORDER:
        if name not in trainers:
            continue
        model = trainers[name].best_estimator_

        train_pred = model.predict(X_train)
        train_proba = _predict_proba_ordered(model, X_train)
        train_metrics = compute_all_metrics(y_train, train_pred, train_proba)

        test_pred = model.predict(X_test)
        test_proba = _predict_proba_ordered(model, X_test)
        test_metrics = compute_all_metrics(y_test, test_pred, test_proba)

        row = {"Model": settings.MODEL_DISPLAY_NAMES[name],
               "5-fold CV Macro-F1": round(trainers[name].best_cv_score_, 4)}
        for key, display_name in METRIC_DISPLAY_NAMES.items():
            row[f"Train_{display_name}"] = round(train_metrics[key], 4)
            row[f"Test_{display_name}"] = round(test_metrics[key], 4)
        rows.append(row)

    return pd.DataFrame(rows)


def build_bootstrap_comparison_table(trainers, X_test, y_test, n_bootstrap=None):
    """
    生成"测试集 Bootstrap 重采样 1000 次后的模型对比表"，每个指标以
    "点估计 (95%CI下限-95%CI上限)" 的形式展示，便于直接放入论文/报告。

    Parameters
    ----------
    trainers : dict[str, models.trainer.ModelTrainer]
    X_test, y_test : pandas.DataFrame / Series
    n_bootstrap : int, optional
        默认使用 settings.N_BOOTSTRAP。

    Returns
    -------
    pandas.DataFrame
        每行 = 一个模型，列为各指标的 "点估计 (95% CI)" 字符串。
    dict[str, pandas.DataFrame]
        模型名 -> 完整的 bootstrap CI 明细表（供需要原始数值时使用）。
    """
    rows = []
    detail_tables = {}
    for name in settings.MODEL_ORDER:
        if name not in trainers:
            continue
        model = trainers[name].best_estimator_
        y_pred = model.predict(X_test)
        y_proba = _predict_proba_ordered(model, X_test)

        ci_table = point_estimate_with_bootstrap_ci(
            y_test, y_pred, y_proba, n_bootstrap=n_bootstrap
        )
        detail_tables[name] = ci_table

        row = {"Model": settings.MODEL_DISPLAY_NAMES[name]}
        ci_lookup = ci_table.set_index("metric")["display"]
        for key, display_name in METRIC_DISPLAY_NAMES.items():
            row[display_name] = ci_lookup[key]
        rows.append(row)

    return pd.DataFrame(rows), detail_tables

# -*- coding: utf-8 -*-
"""
metrics.py
定义多分类场景下所有需要报告的统计学指标的"纯计算函数"。

设计原则：所有函数都是无状态的纯函数（输入数组，输出数值/字典），
不涉及任何模型训练、绘图或文件读写，方便在 bootstrap 重采样中被反复、快速调用。
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
)

from config import settings


def per_class_confusion_counts(y_true, y_pred, classes):
    """
    以 one-vs-rest 方式计算每个类别的 TP / FP / FN / TN。

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    classes : list
        类别标签列表（决定输出顺序）。

    Returns
    -------
    dict[label, dict]
        每个类别对应 {"TP":.., "FP":.., "FN":.., "TN":..}
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    total = cm.sum()
    counts = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        counts[c] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
    return counts


def sensitivity_specificity_ppv_npv(y_true, y_pred, classes=None):
    """
    计算每个类别的灵敏度(Sensitivity/Recall)、特异度(Specificity)、
    阳性预测值(PPV/Precision)、阴性预测值(NPV)，以及四者的宏平均。

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    classes : list, optional
        默认使用 settings.CLASS_ORDER。

    Returns
    -------
    dict
        {
          "per_class": {label: {"sensitivity":.., "specificity":.., "ppv":.., "npv":..}},
          "macro": {"sensitivity":.., "specificity":.., "ppv":.., "npv":..}
        }
    """
    classes = classes or settings.CLASS_ORDER
    counts = per_class_confusion_counts(y_true, y_pred, classes)

    per_class = {}
    for c in classes:
        tp, fp, fn, tn = counts[c]["TP"], counts[c]["FP"], counts[c]["FN"], counts[c]["TN"]
        per_class[c] = {
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
            "ppv": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
            "npv": tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        }

    macro = {
        key: float(np.nanmean([per_class[c][key] for c in classes]))
        for key in ("sensitivity", "specificity", "ppv", "npv")
    }
    return {"per_class": per_class, "macro": macro}


def macro_auc_ovr(y_true, y_proba, classes=None):
    """
    宏平均 One-vs-Rest AUC（多分类场景标准做法）。

    Parameters
    ----------
    y_true : array-like
    y_proba : array-like, shape (n_samples, n_classes)
        模型输出的各类别概率，列顺序须与 classes 一致。
    classes : list, optional

    Returns
    -------
    float
    """
    classes = classes or settings.CLASS_ORDER
    return roc_auc_score(
        y_true, y_proba, multi_class="ovr", average="macro", labels=classes
    )


def compute_all_metrics(y_true, y_pred, y_proba, classes=None):
    """
    一次性计算全部需要报告的模型评价指标（单点估计，不含置信区间；
    置信区间由 evaluation.bootstrap 模块通过反复调用本函数得到）。

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    y_proba : array-like, shape (n_samples, n_classes)
    classes : list, optional

    Returns
    -------
    dict
        扁平化的指标字典，key 包括：
        accuracy, macro_auc, macro_f1, weighted_f1,
        sensitivity, specificity, ppv, npv （均为宏平均）
    """
    classes = classes or settings.CLASS_ORDER

    ssp = sensitivity_specificity_ppv_npv(y_true, y_pred, classes)["macro"]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_auc": macro_auc_ovr(y_true, y_proba, classes),
        "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, labels=classes, average="weighted"),
        "sensitivity": ssp["sensitivity"],
        "specificity": ssp["specificity"],
        "ppv": ssp["ppv"],
        "npv": ssp["npv"],
    }

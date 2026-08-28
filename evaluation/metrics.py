# -*- coding: utf-8 -*-
"""
metrics.py
定义多分类场景下所有需要报告的统计学指标的"纯计算函数"。

设计原则：所有函数都是无状态的纯函数（输入数组，输出数值/字典），
不涉及任何模型训练、绘图或文件读写，方便在 bootstrap 重采样中被反复、快速调用。
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score,
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

    sspn = sensitivity_specificity_ppv_npv(y_true, y_pred, classes)
    sspn_macro = sspn["macro"]
    sspn_per_class = sspn["per_class"]

    f1_per_class = f1_score(y_true, y_pred, labels=classes, average=None)
    auc_per_class = roc_auc_score(y_true, y_proba, multi_class="ovr", average=None, labels=classes)
    acc_per_class = recall_score(y_true, y_pred, labels=classes, average=None)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_auc": macro_auc_ovr(y_true, y_proba, classes),
        "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, labels=classes, average="weighted"),
        "sensitivity": sspn_macro["sensitivity"],
        "specificity": sspn_macro["specificity"],
        "ppv": sspn_macro["ppv"],
        "npv": sspn_macro["npv"],

        "sensitivity_0": sspn_per_class[0]["sensitivity"],
        "specificity_0": sspn_per_class[0]["specificity"],
        "ppv_0": sspn_per_class[0]["ppv"],
        "npv_0": sspn_per_class[0]["npv"],
        "f1_per_class_0": f1_per_class[0],
        "auc_per_class_0": auc_per_class[0],
        "acc_per_class_0": acc_per_class[0],

        "sensitivity_1": sspn_per_class[1]["sensitivity"],
        "specificity_1": sspn_per_class[1]["specificity"],
        "ppv_1": sspn_per_class[1]["ppv"],
        "npv_1": sspn_per_class[1]["npv"],
        "f1_per_class_1": f1_per_class[1],
        "auc_per_class_1": auc_per_class[1],
        "acc_per_class_1": acc_per_class[1],

        "sensitivity_2": sspn_per_class[2]["sensitivity"],
        "specificity_2": sspn_per_class[2]["specificity"],
        "ppv_2": sspn_per_class[2]["ppv"],
        "npv_2": sspn_per_class[2]["npv"],
        "f1_per_class_2": f1_per_class[2],
        "auc_per_class_2": auc_per_class[2],
        "acc_per_class_2": acc_per_class[2],

        "sensitivity_3": sspn_per_class[3]["sensitivity"],
        "specificity_3": sspn_per_class[3]["specificity"],
        "ppv_3": sspn_per_class[3]["ppv"],
        "npv_3": sspn_per_class[3]["npv"],
        "f1_per_class_3": f1_per_class[3],
        "auc_per_class_3": auc_per_class[3],
        "acc_per_class_3": acc_per_class[3],
    }

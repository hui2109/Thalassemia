# -*- coding: utf-8 -*-
"""
plot_roc.py
============
多分类 ROC 曲线绘制。采用 one-vs-rest 方式为每个类别绘制一条 ROC 曲线，
并叠加宏平均 ROC 曲线（先对各类别 FPR 网格插值，再对 TPR 取平均，
这是 scikit-learn 官方推荐的宏平均 ROC 计算方式）。
"""

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


def _macro_average_roc(y_true_bin: np.ndarray, y_proba: np.ndarray, n_classes: int):
    """计算宏平均 ROC 曲线坐标。

    Parameters
    ----------
    y_true_bin : np.ndarray
        one-hot 编码后的真实标签，shape (n_samples, n_classes)。
    y_proba : np.ndarray
        预测概率矩阵，shape (n_samples, n_classes)。
    n_classes : int
        类别数。

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        (fpr_grid, mean_tpr, macro_auc)
    """
    fpr_dict, tpr_dict = {}, {}
    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])

    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, macro_auc


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: List[int],
    class_labels: Dict[int, str],
    model_name: str,
    save_path: Path,
) -> None:
    """为单个模型绘制多分类 ROC 曲线（各类别 one-vs-rest + 宏平均）。

    Parameters
    ----------
    y_true : np.ndarray
        真实标签。
    y_proba : np.ndarray
        预测概率矩阵，列顺序与 `classes` 一致。
    classes : List[int]
        类别列表。
    class_labels : Dict[int, str]
        类别编码到中文名的映射。
    model_name : str
        模型名称，用于图标题与文件名。
    save_path : Path
        图片保存路径。

    Returns
    -------
    None
    """
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc_i = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=1.8, label=f"{class_labels.get(c, c)} (AUC={roc_auc_i:.3f})")

    fpr_macro, tpr_macro, macro_auc = _macro_average_roc(y_true_bin, y_proba, n_classes)
    ax.plot(fpr_macro, tpr_macro, color="black", lw=2.2, linestyle="--", label=f"宏平均 (AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle=":")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("1 - 特异度 (False Positive Rate)")
    ax.set_ylabel("灵敏度 (True Positive Rate)")
    ax.set_title(f"{model_name} ROC 曲线")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_comparison(
    models_proba: Dict[str, np.ndarray],
    y_true: np.ndarray,
    classes: List[int],
    save_path: Path,
) -> None:
    """将多个模型的宏平均 ROC 曲线绘制在同一张图上，便于模型间比较。

    Parameters
    ----------
    models_proba : Dict[str, np.ndarray]
        {模型名: 预测概率矩阵}。
    y_true : np.ndarray
        真实标签（对全部模型通用，即同一测试集）。
    classes : List[int]
        类别列表。
    save_path : Path
        图片保存路径。

    Returns
    -------
    None
    """
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=150)
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_proba)))

    for (name, proba), color in zip(models_proba.items(), colors):
        fpr_macro, tpr_macro, macro_auc = _macro_average_roc(y_true_bin, proba, n_classes)
        ax.plot(fpr_macro, tpr_macro, color=color, lw=1.8, label=f"{name} (宏平均AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle=":")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("1 - 特异度 (False Positive Rate)")
    ax.set_ylabel("灵敏度 (True Positive Rate)")
    ax.set_title("各模型宏平均 ROC 曲线对比")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

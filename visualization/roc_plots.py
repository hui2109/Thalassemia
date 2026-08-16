# -*- coding: utf-8 -*-
"""
roc_plots.py
多分类 One-vs-Rest ROC 曲线绘制。

每个模型一张子图，子图内部画出 4 个类别各自的 ROC 曲线（含 AUC），
最终拼成 1 张包含全部 7 个模型的总图，方便横向比较。
"""

import math

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from config import settings


def _plot_single_model_roc(ax, y_true, y_proba, classes, class_labels):
    """
    在给定的 matplotlib Axes 上绘制单个模型的多类别 One-vs-Rest ROC 曲线。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    y_true : array-like
    y_proba : array-like, shape (n_samples, n_classes)
    classes : list
    class_labels : dict
        类别编码到中文标签的映射。
    """
    for i, c in enumerate(classes):
        y_bin = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.8, label=f"{class_labels.get(c, c)} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.legend(loc="lower right", fontsize=8)


def plot_roc_grid(model_predictions, classes=None, class_labels=None, save_path=None):
    """
    为多个模型批量绘制 ROC 曲线网格图。

    Parameters
    ----------
    model_predictions : dict[str, tuple]
        模型展示名 -> (y_true, y_proba)。
    classes : list, optional
    class_labels : dict, optional
    save_path : str or Path, optional
        若提供，则保存图片到该路径（同时返回 Figure 对象）。

    Returns
    -------
    matplotlib.figure.Figure
    """
    classes = classes or settings.CLASS_ORDER
    class_labels = class_labels or settings.CLASS_LABELS

    n_models = len(model_predictions)
    n_cols = min(4, n_models)
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.2 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for ax, (model_name, (y_true, y_proba)) in zip(axes, model_predictions.items()):
        _plot_single_model_roc(ax, y_true, y_proba, classes, class_labels)
        ax.set_title(model_name, fontsize=11, fontweight="bold")

    for ax in axes[len(model_predictions):]:
        ax.axis("off")

    fig.suptitle("Multiclass One-vs-Rest ROC Curves by Model (Test Set)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

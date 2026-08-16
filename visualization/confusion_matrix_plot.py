# -*- coding: utf-8 -*-
"""
confusion_matrix_plot.py
混淆矩阵热力图绘制。
"""

import math

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import settings


def plot_confusion_matrix_grid(model_predictions, classes=None, class_labels=None,
                               normalize=True, save_path=None):
    """
    为多个模型批量绘制混淆矩阵热力图网格。

    Parameters
    ----------
    model_predictions : dict[str, tuple]
        模型展示名 -> (y_true, y_pred)。注意这里传入的是类别预测值而非概率。
    classes : list, optional
    class_labels : dict, optional
    normalize : bool, optional
        True 时按真实类别行归一化（展示各类召回率），更适合类别不均衡的数据。
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    classes = classes or settings.CLASS_ORDER
    class_labels = class_labels or settings.CLASS_LABELS
    tick_labels = [class_labels.get(c, c) for c in classes]

    n_models = len(model_predictions)
    n_cols = min(4, n_models)
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for ax, (model_name, (y_true, y_pred)) in zip(axes, model_predictions.items()):
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        if normalize:
            cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt = ".2%"
        else:
            cm_display = cm
            fmt = "d"

        sns.heatmap(
            cm_display, annot=True, fmt=fmt, cmap="Blues", cbar=False, ax=ax,
            xticklabels=tick_labels, yticklabels=tick_labels,
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(model_name, fontsize=11, fontweight="bold")

    for ax in axes[len(model_predictions):]:
        ax.axis("off")

    fig.suptitle("Confusion Matrices by Model (Test Set)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

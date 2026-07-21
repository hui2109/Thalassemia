# -*- coding: utf-8 -*-
"""
plot_confusion_matrix.py
==========================
混淆矩阵可视化。同时提供"数量"与"按行归一化的比例"两种展示，
便于直观判断模型在各类别上的误判方向（例如是否倾向于将非静止型误判为静止型）。
"""

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[int],
    class_labels: Dict[int, str],
    model_name: str,
    save_path: Path,
) -> None:
    """绘制并保存混淆矩阵图（左：原始计数，右：按真实类别行归一化比例）。

    Parameters
    ----------
    y_true : np.ndarray
        真实标签。
    y_pred : np.ndarray
        预测标签。
    classes : List[int]
        类别列表。
    class_labels : Dict[int, str]
        类别编码到中文名映射。
    model_name : str
        模型名称。
    save_path : Path
        图片保存路径。

    Returns
    -------
    None
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels = [class_labels.get(c, str(c)) for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)

    for ax, matrix, title, fmt in zip(
        axes, [cm, cm_norm], ["计数", "按真实类别归一化比例"], ["d", ".2f"]
    ):
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title(title)
        thresh = matrix.max() / 2.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                text = f"{value:{fmt}}"
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if value > thresh else "black", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{model_name} 混淆矩阵")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

# -*- coding: utf-8 -*-
"""
plot_dca.py
============
决策曲线分析（Decision Curve Analysis, DCA）。
对每个类别按 one-vs-rest 方式计算净获益（Net Benefit），并与
"全部判阳性(Treat All)"和"全部判阴性(Treat None)"两条参照策略比较，
用以评估模型在不同风险阈值下的临床实用价值。

净获益公式：
    NB(pt) = TP/n - FP/n * (pt / (1 - pt))
其中 pt 为风险阈值概率，TP/FP 为在该阈值下将预测概率 >= pt 判为"阳性"时的真阳/假阳数。
"""

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize


def compute_net_benefit(y_true_binary: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """计算一组风险阈值下的净获益。

    Parameters
    ----------
    y_true_binary : np.ndarray
        0/1 真实标签。
    y_prob : np.ndarray
        预测概率。
    thresholds : np.ndarray
        风险阈值数组，取值范围 (0, 1)（不含端点，端点处 odds 无定义/无意义）。

    Returns
    -------
    np.ndarray
        与 thresholds 等长的净获益数组。
    """
    n = len(y_true_binary)
    net_benefits = np.zeros_like(thresholds)
    for i, pt in enumerate(thresholds):
        predicted_positive = y_prob >= pt
        tp = np.sum(predicted_positive & (y_true_binary == 1))
        fp = np.sum(predicted_positive & (y_true_binary == 0))
        net_benefits[i] = (tp / n) - (fp / n) * (pt / (1 - pt))
    return net_benefits


def compute_treat_all_net_benefit(y_true_binary: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """计算"全部判阳性"策略在各风险阈值下的净获益，作为 DCA 图的参照曲线。

    Parameters
    ----------
    y_true_binary : np.ndarray
        0/1 真实标签。
    thresholds : np.ndarray
        风险阈值数组。

    Returns
    -------
    np.ndarray
        与 thresholds 等长的净获益数组。
    """
    n = len(y_true_binary)
    prevalence = np.sum(y_true_binary == 1) / n
    return prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))


def plot_dca_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: List[int],
    class_labels: Dict[int, str],
    model_name: str,
    save_path: Path,
    threshold_range: tuple = (0.01, 0.99),
    n_points: int = 99,
) -> None:
    """为单个模型绘制各类别（one-vs-rest）的决策曲线。

    Parameters
    ----------
    y_true : np.ndarray
        真实标签。
    y_proba : np.ndarray
        预测概率矩阵。
    classes : List[int]
        类别列表。
    class_labels : Dict[int, str]
        类别编码到中文名映射。
    model_name : str
        模型名称。
    save_path : Path
        图片保存路径。
    threshold_range : tuple, default (0.01, 0.99)
        风险阈值绘制范围。
    n_points : int, default 99
        阈值采样点数。

    Returns
    -------
    None
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_points)
    y_true_bin = label_binarize(y_true, classes=classes)

    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4.5), dpi=150)
    if len(classes) == 1:
        axes = [axes]

    for i, c in enumerate(classes):
        yt = y_true_bin[:, i]
        yp = y_proba[:, i]
        nb_model = compute_net_benefit(yt, yp, thresholds)
        nb_all = compute_treat_all_net_benefit(yt, thresholds)

        ax = axes[i]
        ax.plot(thresholds, nb_model, color="crimson", lw=2, label=model_name)
        ax.plot(thresholds, nb_all, color="grey", lw=1.2, linestyle="--", label="All positive")
        ax.axhline(0, color="black", lw=1.2, linestyle="-", label="All negative")
        ax.set_ylim(bottom=min(-0.05, np.nanmin(nb_model) - 0.02))
        ax.set_xlabel("风险阈值 (Threshold Probability)")
        ax.set_ylabel("净获益 (Net Benefit)")
        ax.set_title(f"{class_labels.get(c, c)} vs. 其余")
        ax.legend(fontsize=8)

    fig.suptitle(f"{model_name} 决策曲线分析 (DCA)")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

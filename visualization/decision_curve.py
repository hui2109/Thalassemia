# -*- coding: utf-8 -*-
"""
decision_curve.py
决策曲线分析（Decision Curve Analysis, DCA）。

DCA 用于评估"使用模型辅助临床决策"相比"全部治疗 (Treat All)"或
"全部不治疗 (Treat None)"两种朴素策略，在不同的风险阈值下能带来多少净获益（Net Benefit）。
多分类问题同样按 One-vs-Rest 拆解为若干个二分类 DCA 子问题。

净获益公式：
    NB(threshold) = TP/N - FP/N * (threshold / (1 - threshold))
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from config import settings


def compute_net_benefit(y_true_bin, y_prob, thresholds):
    """
    计算模型在一系列风险阈值下的净获益，以及 Treat-All 策略的净获益。

    Parameters
    ----------
    y_true_bin : array-like, 取值 0/1
    y_prob : array-like
        预测为阳性的概率。
    thresholds : array-like
        风险阈值序列，例如 np.arange(0.01, 0.99, 0.01)。

    Returns
    -------
    dict
        {"thresholds":.., "model":.., "treat_all":.., "treat_none":..}
        三条曲线的净获益数组，可直接用于绘图。
    """
    y_true_bin = np.asarray(y_true_bin)
    y_prob = np.asarray(y_prob)
    n = len(y_true_bin)
    prevalence = y_true_bin.mean()

    model_nb, all_nb = [], []
    for t in thresholds:
        predicted_positive = y_prob >= t
        tp = np.sum(predicted_positive & (y_true_bin == 1))
        fp = np.sum(predicted_positive & (y_true_bin == 0))
        model_nb.append(tp / n - fp / n * (t / (1 - t)))
        # Treat-All 策略：假设全部人都为阳性
        all_nb.append(prevalence - (1 - prevalence) * (t / (1 - t)))

    return {
        "thresholds": np.asarray(thresholds),
        "model": np.asarray(model_nb),
        "treat_all": np.asarray(all_nb),
        "treat_none": np.zeros(len(thresholds)),
    }


def plot_decision_curve_grid(model_predictions, classes=None, class_labels=None,
                             thresholds=None, save_path=None):
    """
    为多个模型批量绘制决策曲线网格图（每个子图 = 一个模型，图内含各类别 DCA 曲线）。

    Parameters
    ----------
    model_predictions : dict[str, tuple]
        模型展示名 -> (y_true, y_proba)。
    classes : list, optional
    class_labels : dict, optional
    thresholds : array-like, optional
        默认 0.01 ~ 0.99，步长 0.01。
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    classes = classes or settings.CLASS_ORDER
    class_labels = class_labels or settings.CLASS_LABELS
    thresholds = thresholds if thresholds is not None else np.arange(0.01, 0.99, 0.01)

    n_models = len(model_predictions)
    n_cols = min(4, n_models)
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.2 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for ax, (model_name, (y_true, y_proba)) in zip(axes, model_predictions.items()):
        for i, c in enumerate(classes):
            y_bin = (np.asarray(y_true) == c).astype(int)
            result = compute_net_benefit(y_bin, y_proba[:, i], thresholds)
            ax.plot(result["thresholds"], result["model"], lw=1.6,
                    label=f"{class_labels.get(c, c)}")
        # Treat-all / Treat-none 仅需画一次代表性参考线（用最后一个类别的结果示意）
        ax.plot(result["thresholds"], result["treat_none"], color="black",
                lw=1, linestyle="--", label="Treat None")
        ax.set_ylim(-0.1, max(0.5, np.nanmax(result["model"]) + 0.05))
        ax.set_xlabel("Threshold Probability")
        ax.set_ylabel("Net Benefit")
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")

    for ax in axes[len(model_predictions):]:
        ax.axis("off")

    fig.suptitle("Decision Curve Analysis by Model (One-vs-Rest, Test Set)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

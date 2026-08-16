# -*- coding: utf-8 -*-
"""
calibration_plots.py
校准曲线（Calibration Curve）绘制，以及校准斜率（slope）、截距（intercept）的计算。

多分类问题的校准评估同样采用 One-vs-Rest 思路：对每个类别单独评估
"模型预测该类别的概率" 与 "该类别实际发生频率" 是否一致。

校准斜率/截距的计算方法（Cox 校准回归，临床预测模型验证的标准做法）：
    对 logit(p_pred) 做自变量，真实的二分类结局 (0/1) 做因变量，拟合一元 Logistic 回归：
        logit(P(y=1)) = intercept + slope * logit(p_pred)
    - slope = 1 且 intercept = 0 代表完美校准；
    - slope < 1 提示预测概率"过于极端"（over-fitting 常见表现）；
    - intercept != 0 提示系统性高估或低估。
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

from config import settings


def _logit(p, eps=1e-6):
    """将概率值转换为 logit（做数值截断避免 log(0)）。"""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y_true_bin, y_prob):
    """
    计算单个类别（one-vs-rest 二分类）的校准斜率与截距（Cox 校准回归）。

    Parameters
    ----------
    y_true_bin : array-like, 取值 0/1
    y_prob : array-like
        模型预测该类别为阳性的概率。

    Returns
    -------
    dict
        {"slope": float, "intercept": float}
    """
    x = _logit(np.asarray(y_prob)).reshape(-1, 1)
    y = np.asarray(y_true_bin)

    # C=np.inf 等价于不加正则化的普通 Logistic 回归（Cox 校准回归要求无正则化偏倚估计）
    lr = LogisticRegression(C=np.inf, max_iter=1000)
    lr.fit(x, y)
    return {"slope": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0])}


def compute_calibration_summary(model_predictions, classes=None, class_labels=None):
    """
    批量计算多个模型、多个类别的校准斜率/截距，汇总为一张表。

    Parameters
    ----------
    model_predictions : dict[str, tuple]
        模型展示名 -> (y_true, y_proba)。
    classes : list, optional
    class_labels : dict, optional

    Returns
    -------
    pandas.DataFrame
        列：模型, 类别, 校准斜率, 校准截距
    """
    classes = classes or settings.CLASS_ORDER
    class_labels = class_labels or settings.CLASS_LABELS

    rows = []
    for model_name, (y_true, y_proba) in model_predictions.items():
        for i, c in enumerate(classes):
            y_bin = (np.asarray(y_true) == c).astype(int)
            result = calibration_slope_intercept(y_bin, y_proba[:, i])
            rows.append({
                "Model": model_name,
                "Class": class_labels.get(c, c),
                "Calibration Slope": round(result["slope"], 3),
                "Calibration Intercept": round(result["intercept"], 3),
            })
    return pd.DataFrame(rows)


def plot_calibration_grid(model_predictions, classes=None, class_labels=None,
                          n_bins=10, save_path=None):
    """
    为多个模型批量绘制校准曲线网格图（每个子图 = 一个模型，图内含 4 条类别校准曲线）。

    Parameters
    ----------
    model_predictions : dict[str, tuple]
        模型展示名 -> (y_true, y_proba)。
    classes : list, optional
    class_labels : dict, optional
    n_bins : int, optional
        校准曲线分箱数。
    save_path : str or Path, optional

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
        for i, c in enumerate(classes):
            y_bin = (np.asarray(y_true) == c).astype(int)
            frac_pos, mean_pred = calibration_curve(
                y_bin, y_proba[:, i], n_bins=n_bins, strategy="quantile"
            )
            ax.plot(mean_pred, frac_pos, marker="o", markersize=3,
                    label=f"{class_labels.get(c, c)}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1, label="Perfect Calibration")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")

    for ax in axes[len(model_predictions):]:
        ax.axis("off")

    fig.suptitle("Calibration Curves by Model (One-vs-Rest, Test Set)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

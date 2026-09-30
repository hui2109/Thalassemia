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

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

from config import settings

# 与 ROC 图使用完全相同的类别配色和线型，保持论文插图的一致性。
CLASS_COLORS = ("#3C5488", "#E64B35", "#00A087", "#7E6148")
CLASS_LINESTYLES = ("solid", (0, (5, 2)), (0, (2, 1.4)), (0, (5, 1.5, 1.5, 1.5)))


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


def _plot_single_model_calibration(ax, y_true, y_proba, classes, class_labels, n_bins):
    """在单个面板中绘制四类分箱校准曲线与理想校准参考线。"""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_true.ndim != 1 or y_true.size == 0:
        raise ValueError("y_true must be a nonempty one-dimensional array.")
    if y_proba.shape != (y_true.size, len(classes)):
        raise ValueError(
            "y_proba must have shape (len(y_true), len(classes)); "
            "its columns must follow the order of classes."
        )
    if not np.isfinite(y_proba).all() or np.any((y_proba < 0) | (y_proba > 1)):
        raise ValueError("y_proba must contain finite probabilities in [0, 1].")

    ax.plot(
        (0, 1), (0, 1), color="#AAB3BA", linewidth=1.15,
        linestyle=(0, (3, 3)), label="Perfect calibration", zorder=1,
    )

    for i, cls in enumerate(classes):
        positive = y_true == cls
        if not positive.any() or positive.all():
            raise ValueError(
                f"Class {cls!r} must have both positive and negative examples "
                "in the test set to calculate its calibration curve."
            )
        frac_pos, mean_pred = calibration_curve(
            positive.astype(int), y_proba[:, i], n_bins=n_bins,
            strategy="quantile",
        )
        ax.plot(
            mean_pred, frac_pos, color=CLASS_COLORS[i],
            linestyle=CLASS_LINESTYLES[i], linewidth=2.05,
            marker="o", markersize=4, markerfacecolor="white",
            markeredgewidth=1.25, solid_capstyle="round", zorder=3,
            label=str(class_labels.get(cls, cls)),
        )

    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(which="major", color="#E4E9EC", linewidth=0.75, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#68747D")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", labelsize=8.5,
                   length=3.5, width=0.8, colors="#45515A")
    ax.tick_params(axis="both", which="minor", length=2, width=0.6,
                   colors="#87929A")
    # 白底图例减轻曲线与文字的相互遮挡，并与 ROC 面板保持同样的边框样式。
    legend = ax.legend(
        loc="upper left", fontsize=8.1, frameon=True, fancybox=False,
        framealpha=0.96, facecolor="white", edgecolor="#D6DEE3",
        handlelength=2.5, handletextpad=0.65,
        labelspacing=0.35, borderpad=0.55,
    )
    legend.get_frame().set_linewidth(0.65)


def plot_calibration_grid(model_predictions, classes=None, class_labels=None,
                          n_bins=10, save_path=None, dpi=600):
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

    参数补充
    --------
    dpi：导出位图的分辨率，默认 600；保存时至少为 600。
    概率矩阵的列顺序必须与 classes 一致。
    """
    if not model_predictions:
        raise ValueError("model_predictions must contain at least one model.")
    classes = tuple(settings.CLASS_ORDER if classes is None else classes)
    class_labels = settings.CLASS_LABELS if class_labels is None else class_labels
    if len(classes) != 4 or len(set(classes)) != 4:
        raise ValueError("This four-class figure requires four distinct classes.")
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least 2.")
    if save_path is not None and dpi < 600:
        raise ValueError("Publication exports require dpi >= 600.")

    n_models = len(model_predictions)
    n_cols = min(4, n_models)
    n_rows = math.ceil(n_models / n_cols)
    preferred_font = (
        "Times New Roman" if any(f.name == "Times New Roman"
                                 for f in font_manager.fontManager.ttflist)
        else "DejaVu Serif"
    )
    with plt.rc_context({
        "font.family": preferred_font,
        "font.size": 10,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }):
        fig = plt.figure(figsize=(4.1 * n_cols, 4.05 * n_rows + 0.45),
                         layout="constrained")
        # 与 ROC 图相同：网格列数加倍，使不足四张的最后一行居中。
        grid = fig.add_gridspec(n_rows, 2 * n_cols)
        fig.patch.set_facecolor("white")
        fig.suptitle("One-vs-rest calibration curves (Test set)",
                     fontsize=15, fontweight="bold", color="#253746")

        try:
            for index, (name, (y_true, y_proba)) in enumerate(model_predictions.items()):
                row, col = divmod(index, n_cols)
                panels_in_row = min(n_cols, n_models - row * n_cols)
                offset = n_cols - panels_in_row
                ax = fig.add_subplot(grid[row, offset + 2 * col:offset + 2 * col + 2])
                _plot_single_model_calibration(
                    ax, y_true, y_proba, classes, class_labels, n_bins,
                )
                display_name = settings.MODEL_DISPLAY_NAMES.get(name, name)
                ax.set_title(f"{chr(65 + index)}  {display_name}", loc="left",
                             fontsize=10.5, fontweight="bold", color="#253746", pad=10)
                ax.set_xlabel("Predicted probability", fontsize=9.2, labelpad=6)
                ax.set_ylabel("Observed frequency", fontsize=9.2, labelpad=6)

            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
        except Exception:
            plt.close(fig)
            raise
        return fig

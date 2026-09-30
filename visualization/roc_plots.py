# -*- coding: utf-8 -*-
"""
roc_plots.py
多分类 One-vs-Rest ROC 曲线绘制。

每个模型一张子图，子图内部画出 4 个类别各自的 ROC 曲线（含 AUC），
最终拼成 1 张包含全部 7 个模型的总图，方便横向比较。

概率矩阵 y_proba 的列顺序 应与 classes 一致；默认使用 settings.CLASS_ORDER。
保留原有 plot_roc_grid 调用方式，支持高分辨率位图和矢量图导出。
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from sklearn.metrics import auc, roc_curve

from config import settings

# 参考 Nature 系列常用科研配色；所有模型的类别颜色保持一致。
CLASS_COLORS = ("#3C5488", "#E64B35", "#00A087", "#7E6148")
# 使用不同线型，便于在灰度打印时区分四个类别。
CLASS_LINESTYLES = ("solid", (0, (5, 2)), (0, (2, 1.4)), (0, (5, 1.5, 1.5, 1.5)))


def _plot_single_model_roc(ax, y_true, y_proba, classes, class_labels):
    """
    在给定的 matplotlib Axes 上绘制单个模型的多类别 One-vs-Rest ROC 曲线。

    参数
    ----
    ax：matplotlib.axes.Axes
        用于绘制单个模型 ROC 曲线的坐标轴。
    y_true：一维数组
        测试集真实类别。
    y_proba：二维数组，形状为 (n_samples, n_classes)
        按 classes 顺序排列的类别预测概率。
    classes：列表
        类别编码及概率矩阵的列顺序。
    class_labels：字典
        类别编码到中文标签的映射。
        当前论文配置中的实际展示标签为英文。
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_true.ndim != 1 or y_true.size == 0:
        raise ValueError("y_true must be a nonempty one-dimensional array.")
    if y_proba.shape != (y_true.size, len(classes)):
        raise ValueError(
            "y_proba must have shape (len(y_true), len(classes)); "
            "its columns must follow the order of classes."
        )
    if not np.isfinite(y_proba).all():
        raise ValueError("y_proba contains NaN or infinite values.")

    ax.plot(
        (0, 1), (0, 1), color="#AAB3BA", linewidth=1.15,
        linestyle=(0, (3, 3)), label="Chance (AUC = 0.50)", zorder=1,
    )

    for i, cls in enumerate(classes):
        positive = y_true == cls
        if not positive.any() or positive.all():
            raise ValueError(
                f"Class {cls!r} must have both positive and negative examples "
                "in the test set to calculate its ROC curve."
            )
        fpr, tpr, _ = roc_curve(positive, y_proba[:, i], drop_intermediate=False)
        area = auc(fpr, tpr)
        ax.plot(
            fpr, tpr, color=CLASS_COLORS[i % len(CLASS_COLORS)],
            linestyle=CLASS_LINESTYLES[i % len(CLASS_LINESTYLES)],
            linewidth=2.05, solid_capstyle="round", zorder=3,
            label=f"{class_labels.get(cls, cls)} (AUC {area:.2f})",
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
    # 每张子图分别展示当前模型的四个 AUC 值；半透明白底避免曲线遮挡文字。
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles[1:], labels[1:], loc="lower right", fontsize=8.1,
        frameon=True, fancybox=False, framealpha=0.96,
        facecolor="white", edgecolor="#D6DEE3",
        handlelength=2.5, handletextpad=0.65,
        labelspacing=0.35, borderpad=0.55,
    )
    legend.get_frame().set_linewidth(0.65)


def plot_roc_grid(model_predictions, classes=None, class_labels=None,
                  save_path=None, dpi=600):
    """
    为多个模型批量绘制 ROC 曲线网格图。

    参数
    ----
    model_predictions：模型名到预测结果的映射
        模型展示名 -> (y_true, y_proba)。
        其中 y_proba 的列顺序必须与 classes 一致。
    classes：可选列表
        类别顺序；默认读取 settings.CLASS_ORDER。
    class_labels：可选字典
        用于图内展示的英文类别名；默认读取 settings.CLASS_LABELS。
    save_path：可选路径
        若提供，则保存图片到该路径（同时返回 Figure 对象）。
        根据文件后缀导出 PNG、TIFF、PDF 或 SVG；后两者为矢量图。
    dpi：整数，默认为 600
        位图的导出分辨率；投稿图像至少使用 600 dpi。

    返回
    ----
    matplotlib.figure.Figure
        使用结束后可通过 plt.close(fig) 释放图像内存。
    """
    if not model_predictions:
        raise ValueError("model_predictions must contain at least one model.")
    classes = tuple(settings.CLASS_ORDER if classes is None else classes)
    class_labels = settings.CLASS_LABELS if class_labels is None else class_labels
    if len(classes) != 4 or len(set(classes)) != 4:
        raise ValueError("This four-class figure requires four distinct classes.")
    if dpi < 600 and save_path is not None:
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
        # 将网格列数加倍，使最后一行居中，同时保持每张子图的宽度一致。
        grid = fig.add_gridspec(n_rows, 2 * n_cols)
        fig.patch.set_facecolor("white")
        fig.suptitle("One-vs-rest ROC curves (Test set)",
                     fontsize=15, fontweight="bold", color="#253746")

        try:
            for index, (name, (y_true, y_proba)) in enumerate(model_predictions.items()):
                row, col = divmod(index, n_cols)
                panels_in_row = min(n_cols, n_models - row * n_cols)
                offset = n_cols - panels_in_row
                ax = fig.add_subplot(grid[row, offset + 2 * col:offset + 2 * col + 2])
                _plot_single_model_roc(ax, y_true, y_proba, classes, class_labels)
                display_name = settings.MODEL_DISPLAY_NAMES.get(name, name)
                ax.set_title(f"{chr(65 + index)}  {display_name}", loc="left",
                             fontsize=10.5, fontweight="bold", color="#253746", pad=10)
                ax.set_xlabel("False positive rate",
                              fontsize=9.2, labelpad=6)
                ax.set_ylabel("True positive rate",
                              fontsize=9.2, labelpad=6)

            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
        except Exception:
            plt.close(fig)
            raise
        return fig

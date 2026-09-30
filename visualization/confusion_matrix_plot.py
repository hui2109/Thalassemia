# -*- coding: utf-8 -*-
"""
confusion_matrix_plot.py
混淆矩阵热力图绘制。
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, font_manager
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix

from config import settings

# 与 roc_plots.py 的四类颜色保持一致，色条使用其中的海军蓝。
CLASS_COLORS = ("#3C5488", "#E64B35", "#00A087", "#7E6148")
TEXT_COLOR = "#253746"
MUTED_COLOR = "#68747D"


def _draw_single_matrix(ax, cm, classes, class_labels, normalize, cmap, norm):
    """在单张子图内绘制四分类矩阵、类别标记和单元格数值。"""
    row_totals = cm.sum(axis=1)
    if normalize:
        values = np.divide(
            cm, row_totals[:, None], out=np.zeros_like(cm, dtype=float),
            where=row_totals[:, None] != 0,
        )
        # 测试集中不存在的类别没有可定义的行百分比，使用灰底与短横线标明。
        values = np.ma.masked_where(
            np.broadcast_to(row_totals[:, None] == 0, cm.shape), values
        )
    else:
        values = cm

    image = ax.imshow(values, cmap=cmap, norm=norm, interpolation="nearest")
    tick_labels = [str(class_labels.get(c, c)) for c in classes]
    ax.set_xticks(range(4), labels=tick_labels)
    ax.set_yticks(range(4), labels=tick_labels)
    ax.tick_params(axis="both", which="major", length=0, pad=7,
                   labelsize=8.5, colors=MUTED_COLOR)
    ax.set_xlabel("Predicted class", fontsize=9.2, color=TEXT_COLOR, labelpad=8)
    ax.set_ylabel("True class", fontsize=9.2, color=TEXT_COLOR, labelpad=8)
    ax.set_box_aspect(1)

    # 明确关闭外部样式可能启用的主网格，避免每格正中出现十字线。
    ax.grid(False, which="both", axis="both")
    # 仅在相邻格子之间绘制浅色分界；行列旁的细色标对应 ROC 图的类别颜色。
    boundaries = np.arange(0.5, 3.5, 1)
    ax.hlines(boundaries, -0.5, 3.5, colors="white", linewidths=1.6, zorder=2)
    ax.vlines(boundaries, -0.5, 3.5, colors="white", linewidths=1.6, zorder=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for index, accent in enumerate(CLASS_COLORS):
        ax.add_patch(Rectangle((-0.59, index - 0.42), 0.035, 0.84,
                               facecolor=accent, edgecolor="none", clip_on=False))
        ax.add_patch(Rectangle((index - 0.42, -0.59), 0.84, 0.035,
                               facecolor=accent, edgecolor="none", clip_on=False))

    for row in range(4):
        for col in range(4):
            if normalize and row_totals[row] == 0:
                ax.text(col, row, "—", ha="center", va="center",
                        fontsize=10, color=MUTED_COLOR)
                continue
            value = float(values[row, col])
            foreground = "white" if norm(value) >= 0.59 else TEXT_COLOR
            if normalize:
                # 使用同一行的真实样本数作分母，第二行仍给出可核查的原始计数。
                percent = "<0.1%" if 0 < value < 0.0005 else f"{value:.1%}"
                ax.text(col, row - 0.12, percent, ha="center", va="center",
                        fontsize=9.2, fontweight="bold", color=foreground)
                ax.text(col, row + 0.20, f"n = {cm[row, col]}",
                        ha="center", va="center", fontsize=7.5, color=foreground)
            else:
                ax.text(col, row, f"{cm[row, col]:,}", ha="center", va="center",
                        fontsize=9.7, fontweight="bold", color=foreground)
    return image


def plot_confusion_matrix_grid(model_predictions, classes=None, class_labels=None,
                               normalize=True, save_path=None, dpi=600):
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

    补充说明
    --------
    默认每格标注行百分比和样本数；normalize=False 时标注原始样本数。
    dpi 默认为 600，支持按保存路径后缀导出 PNG、TIFF、PDF 或 SVG。
    """
    if not model_predictions:
        raise ValueError("model_predictions must contain at least one model.")
    classes = tuple(settings.CLASS_ORDER if classes is None else classes)
    class_labels = settings.CLASS_LABELS if class_labels is None else class_labels
    if len(classes) != 4 or len(set(classes)) != 4:
        raise ValueError("This four-class figure requires four distinct classes.")
    if save_path is not None and dpi < 600:
        raise ValueError("Publication exports require dpi >= 600.")

    # 先收集并核对所有矩阵，确保色条的范围在不同模型之间可直接比较。
    matrices = []
    for name, (y_true, y_pred) in model_predictions.items():
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        if (y_true.ndim != 1 or y_pred.ndim != 1 or not y_true.size
                or y_true.shape != y_pred.shape):
            raise ValueError(f"{name}: y_true and y_pred must be nonempty, equally sized 1-D arrays.")
        if not np.isin(y_true, classes).all() or not np.isin(y_pred, classes).all():
            raise ValueError(f"{name}: predictions contain labels outside classes.")
        matrices.append(confusion_matrix(y_true, y_pred, labels=classes))

    n_models = len(matrices)
    n_cols = min(4, n_models)
    n_rows = math.ceil(n_models / n_cols)
    preferred_font = (
        "Times New Roman" if any(f.name == "Times New Roman"
                                 for f in font_manager.fontManager.ttflist)
        else "DejaVu Serif"
    )
    if normalize:
        norm = colors.Normalize(vmin=0, vmax=1)
    else:
        norm = colors.Normalize(vmin=0, vmax=max(1, *(int(cm.max()) for cm in matrices)))
    cmap = colors.LinearSegmentedColormap.from_list(
        "roc_navy_matrix", ("#F2F6F8", "#DDE7EE", "#A7BDD0", "#6485A9", "#3C5488", "#233A68")
    ).copy()
    cmap.set_bad("#F0F1F2")

    with plt.rc_context({
        "font.family": preferred_font,
        "font.size": 10,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }):
        fig = plt.figure(figsize=(4.0 * n_cols, 4.05 * n_rows + 1.0),
                         facecolor="white")
        grid = fig.add_gridspec(n_rows, 2 * n_cols, left=0.08, right=0.965,
                                top=0.87, bottom=0.18, wspace=0.58, hspace=0.67)
        fig.suptitle("Confusion matrices (Test set)", y=0.96,
                     fontsize=15, fontweight="bold", color=TEXT_COLOR)

        try:
            for index, ((name, _), cm) in enumerate(zip(model_predictions.items(), matrices)):
                row, col = divmod(index, n_cols)
                panels_in_row = min(n_cols, n_models - row * n_cols)
                offset = n_cols - panels_in_row
                ax = fig.add_subplot(grid[row, offset + 2 * col:offset + 2 * col + 2])
                image = _draw_single_matrix(ax, cm, classes, class_labels,
                                            normalize, cmap, norm)
                display_name = settings.MODEL_DISPLAY_NAMES.get(name, name)
                panel_letter = chr(65 + index) if index < 26 else str(index + 1)
                ax.set_title(f"{panel_letter}  {display_name}", loc="left",
                             fontsize=10.5, fontweight="bold", color=TEXT_COLOR, pad=13)

            # 全图共用固定色标，避免各模型分别缩放导致深浅失去可比性。
            colorbar_ax = fig.add_axes((0.355, 0.075, 0.29, 0.018))
            colorbar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal")
            colorbar.outline.set_visible(False)
            colorbar.ax.tick_params(length=2.5, width=0.65, labelsize=8,
                                    colors=MUTED_COLOR, pad=3)
            if normalize:
                colorbar.set_ticks(np.linspace(0, 1, 5))
                colorbar.ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
                colorbar.set_label("Share within true class", fontsize=8.8,
                                   color=TEXT_COLOR, labelpad=5)
            else:
                colorbar.set_label("Number of samples", fontsize=8.8,
                                   color=TEXT_COLOR, labelpad=5)

            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
        except Exception:
            plt.close(fig)
            raise
        return fig

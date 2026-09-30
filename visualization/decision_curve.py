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

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MaxNLocator, MultipleLocator

from config import settings

# 与同一论文的 ROC 图片使用完全相同的类别颜色和线型。
CLASS_COLORS = ("#3C5488", "#E64B35", "#00A087", "#7E6148")
CLASS_LINESTYLES = ("solid", (0, (5, 2)), (0, (2, 1.4)), (0, (5, 1.5, 1.5, 1.5)))


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
    y_prob = np.asarray(y_prob, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if y_true_bin.ndim != 1 or y_true_bin.size == 0:
        raise ValueError("y_true_bin must be a nonempty one-dimensional array.")
    if y_prob.shape != y_true_bin.shape:
        raise ValueError("y_prob must have the same one-dimensional shape as y_true_bin.")
    if not np.isin(y_true_bin, (0, 1)).all():
        raise ValueError("y_true_bin must contain only 0 and 1.")
    if not np.isfinite(y_prob).all() or np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("y_prob must contain finite probabilities between 0 and 1.")
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("thresholds must be a nonempty one-dimensional array.")
    if (not np.isfinite(thresholds).all()
            or np.any((thresholds < 0) | (thresholds >= 1))
            or np.any(np.diff(thresholds) <= 0)):
        raise ValueError("thresholds must be strictly increasing and lie in [0, 1).")

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
        "thresholds": thresholds,
        "model": np.asarray(model_nb),
        "treat_all": np.asarray(all_nb),
        "treat_none": np.zeros(len(thresholds)),
    }


def _style_axis(ax):
    """为所有模型设置一致的论文图表样式。"""
    ax.set_xlim(0, 1)
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
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


def plot_decision_curve_grid(model_predictions, classes=None, class_labels=None,
                             thresholds=None, save_path=None, dpi=600):
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

    概率矩阵的列顺序必须与 classes 一致；四个类别分别进行一对其余类别的分析。
    类别对应的浅色点线表示该类别的 Treat All；灰色水平线表示 Treat None。
    高阈值处 Treat All 可能急剧下降，参考线仅在共同纵轴范围内显示。
    dpi 默认为 600；保存为 PDF 或 SVG 时还可获得矢量图。
    """
    if not model_predictions:
        raise ValueError("model_predictions must contain at least one model.")
    classes = tuple(settings.CLASS_ORDER if classes is None else classes)
    class_labels = settings.CLASS_LABELS if class_labels is None else class_labels
    thresholds = (np.arange(0.01, 0.99, 0.01) if thresholds is None
                  else np.asarray(thresholds, dtype=float))
    if len(classes) != 4 or len(set(classes)) != 4:
        raise ValueError("This four-class figure requires four distinct classes.")
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
        fig.patch.set_facecolor("white")
        grid = fig.add_gridspec(n_rows, 2 * n_cols)
        fig.suptitle("One-vs-rest decision curves (Test set)",
                     fontsize=15, fontweight="bold", color="#253746")
        # Treat-all / Treat-none 仅需画一次代表性参考线（用最后一个类别的结果示意）
        # 原注释对应旧版；四个类别的患病率不同，Treat All 必须分别计算和绘制。
        try:
            model_bounds = []
            panels = []
            for index, (name, (y_true, y_proba)) in enumerate(model_predictions.items()):
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
                    raise ValueError("y_proba must contain finite probabilities between 0 and 1.")

                row, col = divmod(index, n_cols)
                ax = fig.add_subplot(grid[row, 2 * col:2 * col + 2])
                for i, c in enumerate(classes):
                    y_bin = (y_true == c).astype(int)
                    if not y_bin.any() or y_bin.all():
                        raise ValueError(
                            f"Class {c!r} must have both positive and negative examples "
                            "in the test set for one-vs-rest DCA."
                        )
                    result = compute_net_benefit(y_bin, y_proba[:, i], thresholds)
                    model_bounds.extend((np.min(result["model"]), np.max(result["model"])))
                    # 同色浅点线表示该类别的全部干预策略，避免误用其他类别的患病率。
                    ax.plot(result["thresholds"], result["treat_all"],
                            color=CLASS_COLORS[i], linestyle=(0, (1, 2.4)),
                            linewidth=1.2, alpha=0.50, zorder=1)
                    ax.plot(result["thresholds"], result["model"],
                            color=CLASS_COLORS[i],
                            linestyle=CLASS_LINESTYLES[i],
                            linewidth=2.05, solid_capstyle="round", zorder=3,
                            label=f"{class_labels.get(c, c)}")
                ax.axhline(0, color="#7D8992", linewidth=1.1,
                           linestyle=(0, (3, 3)), zorder=2)
                _style_axis(ax)
                display_name = settings.MODEL_DISPLAY_NAMES.get(name, name)
                ax.set_title(f"{chr(65 + index)}  {display_name}", loc="left",
                             fontsize=10.5, fontweight="bold", color="#253746", pad=10)
                ax.set_xlabel("Threshold probability", fontsize=9.2, labelpad=6)
                ax.set_ylabel("Net benefit", fontsize=9.2, labelpad=6)
                # 图例仅列出四类模型曲线；统一说明解释两种参考策略。
                panels.append(ax)

            # 最后一行未填满时，把整图图例放在右下角空出的面板位置。
            handles, labels = panels[0].get_legend_handles_labels()
            handles += [
                Line2D([], [], color="#839099", linestyle=(0, (1, 2.4)),
                       linewidth=1.2, label="Treat all (class matched)"),
                Line2D([], [], color="#7D8992", linestyle=(0, (3, 3)),
                       linewidth=1.1, label="Treat none"),
            ]
            labels += ["Treat all (class matched)", "Treat none"]
            if n_models % n_cols:
                legend_ax = fig.add_subplot(grid[-1, -2:])
                legend_ax.set_axis_off()
                legend_anchor = (0.5, 0.5)
                legend_transform = legend_ax.transAxes
                legend_location = "center"
            else:
                # 面板恰好填满时，将共用图例放到整图下方。
                legend_anchor = (0.5, -0.02)
                legend_transform = fig.transFigure
                legend_location = "upper center"
            legend = fig.legend(
                handles, labels, loc=legend_location,
                bbox_to_anchor=legend_anchor, bbox_transform=legend_transform,
                fontsize=8.5, frameon=True, fancybox=False, framealpha=0.96,
                facecolor="white", edgecolor="#D6DEE3",
                handlelength=2.4, handletextpad=0.65,
                labelspacing=0.42, borderpad=0.7,
            )
            legend.get_frame().set_linewidth(0.65)

            # 用模型曲线的完整取值统一纵轴，使不同模型可以直接比较。
            y_min = min(-0.10, min(model_bounds) - 0.025)
            y_max = max(0.45, max(model_bounds) + 0.045)
            for ax in panels:
                ax.set_ylim(y_min, y_max)

            # 统一收紧各面板的纵轴范围，突出具有解释价值的净获益区间
            for ax in fig.axes:
                if ax.has_data():
                    ax.set_ylim(-0.30, 0.50)
                    ax.set_yticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

            if save_path is not None:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
        except Exception:
            plt.close(fig)
            raise
        return fig

# -*- coding: utf-8 -*-
"""
plot_calibration.py
=====================
校准曲线（Calibration Curve）绘制，以及校准斜率（calibration slope）与
校准截距（calibration intercept）的计算。

计算方法：将预测概率转换为对数几率 logit(p)，以其为唯一自变量，
对真实的二分类结局（one-vs-rest）拟合一个不带正则化的逻辑回归：
    logit(P(y=1)) = intercept + slope * logit(p_predicted)
理想模型应满足 intercept ≈ 0，slope ≈ 1。
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize


def compute_calibration_slope_intercept(y_true_binary: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """计算单一类别（二分类视角）的校准斜率与截距。

    Parameters
    ----------
    y_true_binary : np.ndarray
        0/1 真实标签。
    y_prob : np.ndarray
        该类别的预测概率，取值范围 (0, 1)。

    Returns
    -------
    Tuple[float, float]
        (slope, intercept)
    """
    eps = 1e-6
    p_clipped = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p_clipped / (1 - p_clipped)).reshape(-1, 1)

    if len(np.unique(y_true_binary)) < 2:
        return np.nan, np.nan

    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    lr.fit(logit_p, y_true_binary)
    slope = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])
    return slope, intercept


def plot_calibration_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: List[int],
    class_labels: Dict[int, str],
    model_name: str,
    save_path: Path,
    n_bins: int = 10,
) -> Dict[int, Tuple[float, float]]:
    """为单个模型绘制各类别（one-vs-rest）的校准曲线，并返回各类别校准斜率/截距。

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
    n_bins : int, default 10
        校准曲线分箱数。

    Returns
    -------
    Dict[int, Tuple[float, float]]
        {类别: (slope, intercept)}
    """
    y_true_bin = label_binarize(y_true, classes=classes)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    slope_intercept = {}

    ax.plot([0, 1], [0, 1], linestyle=":", color="grey", label="理想校准线")

    for i, c in enumerate(classes):
        yt = y_true_bin[:, i]
        yp = y_proba[:, i]
        prob_true, prob_pred = calibration_curve(yt, yp, n_bins=n_bins, strategy="quantile")
        slope, intercept = compute_calibration_slope_intercept(yt, yp)
        slope_intercept[c] = (slope, intercept)
        ax.plot(
            prob_pred, prob_true, marker="o", color=colors[i], lw=1.6,
            label=f"{class_labels.get(c, c)} (斜率={slope:.2f}, 截距={intercept:.2f})",
        )

    ax.set_xlabel("预测概率")
    ax.set_ylabel("实际观测频率")
    ax.set_title(f"{model_name} 校准曲线")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return slope_intercept

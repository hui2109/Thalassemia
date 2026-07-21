# -*- coding: utf-8 -*-
"""
shap_analysis.py
==================
基于 SHAP (SHapley Additive exPlanations) 的模型可解释性分析。

设计说明：
- 树模型（Decision Tree / Random Forest / XGBoost / LightGBM）使用 shap.TreeExplainer，
  直接对 Pipeline 中的 clf 步骤计算（这些模型未做标准化，clf 接收原始特征）。
- 非树模型（LR / SVC / MLP）使用 shap.Explainer 包裹整个 Pipeline 的 predict_proba，
  以背景数据集做核估计（Kernel/Permutation），计算量较大，因此背景与解释样本量需要控制。
- 三分类场景下 SHAP 值维度为 (n_samples, n_features, n_classes)，本模块统一标准化为
  该形状后再绘图，避免不同 shap 版本 API 差异带来的麻烦；绘图函数为自实现的
  蜂群摘要图（beeswarm）与条形图，不依赖 shap 内置绘图函数，方便客户自行调整样式。
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


TREE_MODEL_NAMES = {"Decision Tree", "Random Forest", "XGBoost", "LightGBM"}


def _standardize_shap_output(raw_output, n_samples: int, n_features: int) -> np.ndarray:
    """将不同版本 shap 库返回的多分类结果统一标准化为 ndarray, shape (n_samples, n_features, n_classes)。

    Parameters
    ----------
    raw_output : object
        shap.Explainer(...)(X) 或 explainer.shap_values(X) 的原始返回值，
        可能是 shap.Explanation、list[np.ndarray] 或 np.ndarray。
    n_samples, n_features : int
        期望的样本数与特征数，用于校验/reshape。

    Returns
    -------
    np.ndarray
        shape (n_samples, n_features, n_classes)。

    Raises
    ------
    TypeError
        无法识别的返回类型时抛出。
    """
    if _HAS_SHAP and isinstance(raw_output, shap.Explanation):
        values = raw_output.values
    else:
        values = raw_output

    if isinstance(values, list):
        # legacy 版本: list[ (n_samples, n_features) ]，长度为类别数
        values = np.stack(values, axis=-1)

    values = np.asarray(values)

    if values.ndim == 2:
        # 二分类或单输出情形，补齐为 3 维，类别维度为 1
        values = values[:, :, np.newaxis]

    if values.shape[0] != n_samples or values.shape[1] != n_features:
        raise TypeError(f"SHAP 输出维度异常: {values.shape}, 期望第一维={n_samples}, 第二维={n_features}")

    return values


def compute_shap_values(
    model: Pipeline,
    model_name: str,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    max_background: int = 100,
) -> np.ndarray:
    """计算给定模型在 X_explain 上的 SHAP 值。

    Parameters
    ----------
    model : Pipeline
        已训练好的模型 Pipeline（含或不含 scaler 步骤）。
    model_name : str
        模型名称，用于判断走 TreeExplainer 还是通用 Explainer。
    X_background : pd.DataFrame
        背景数据集（用于估计特征的边际分布），一般取训练集的一个子样本。
    X_explain : pd.DataFrame
        需要计算 SHAP 值并可视化的样本集（一般为测试集，或其子样本）。
    max_background : int, default 100
        非树模型使用核估计时，背景样本量上限（越大越准但越慢）。

    Returns
    -------
    np.ndarray
        SHAP 值数组，shape (n_samples_explain, n_features, n_classes)。

    Raises
    ------
    ImportError
        未安装 shap 库时抛出。
    """
    if not _HAS_SHAP:
        raise ImportError("未检测到 shap 库，请先执行: pip install shap")

    n_explain, n_features = X_explain.shape

    if model_name in TREE_MODEL_NAMES:
        clf = model.named_steps["clf"]
        explainer = shap.TreeExplainer(clf)
        raw = explainer.shap_values(X_explain)
    else:
        background = X_background.sample(
            n=min(max_background, len(X_background)), random_state=42
        )
        explainer = shap.Explainer(model.predict_proba, background)
        raw = explainer(X_explain)

    return _standardize_shap_output(raw, n_explain, n_features)


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: List[str],
    class_labels: Dict[int, str],
    classes: List[int],
    model_name: str,
    save_path: Path,
    top_n: int = 15,
) -> None:
    """绘制 SHAP 特征重要性条形图：按各类别分组展示平均绝对 SHAP 值。

    Parameters
    ----------
    shap_values : np.ndarray
        shape (n_samples, n_features, n_classes)。
    feature_names : List[str]
        特征名列表，顺序与 shap_values 第二维一致。
    class_labels : Dict[int, str]
        类别编码到中文名映射。
    classes : List[int]
        类别列表，顺序与 shap_values 第三维一致。
    model_name : str
        模型名称。
    save_path : Path
        图片保存路径。
    top_n : int, default 15
        展示重要性最高的前 N 个特征。

    Returns
    -------
    None
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)  # shape (n_features, n_classes)
    overall_importance = mean_abs.mean(axis=1)
    order = np.argsort(overall_importance)[::-1][:top_n]
    ordered_features = [feature_names[i] for i in order][::-1]

    n_classes = len(classes)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * top_n)), dpi=150)
    bar_height = 0.8 / n_classes
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for k, c in enumerate(classes):
        y_positions = np.arange(len(ordered_features)) + k * bar_height
        values = [mean_abs[i, k] for i in order][::-1]
        ax.barh(y_positions, values, height=bar_height, color=colors[k], label=class_labels.get(c, c))

    ax.set_yticks(np.arange(len(ordered_features)) + bar_height * (n_classes - 1) / 2)
    ax.set_yticklabels(ordered_features)
    ax.set_xlabel("平均绝对 SHAP 值 (mean |SHAP value|)")
    ax.set_title(f"{model_name} SHAP 特征重要性条形图")
    ax.legend(title="类别", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_shap_summary_beeswarm(
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    class_labels: Dict[int, str],
    classes: List[int],
    model_name: str,
    save_path: Path,
    class_index: Optional[int] = None,
    top_n: int = 15,
) -> None:
    """绘制 SHAP 摘要图（蜂群图 / beeswarm plot），展示每个特征取值高低对预测的影响方向与幅度。

    Parameters
    ----------
    shap_values : np.ndarray
        shape (n_samples, n_features, n_classes)。
    X_explain : pd.DataFrame
        对应的原始特征取值（用于按特征值高低着色）。
    class_labels : Dict[int, str]
        类别编码到中文名映射。
    classes : List[int]
        类别列表。
    model_name : str
        模型名称。
    save_path : Path
        图片保存路径。
    class_index : Optional[int], default None
        指定绘制哪个类别（取 shap_values 第三维的索引）；为 None 时默认取最后一个类别
        （本项目中通常对应"非静止型地贫"，临床上最关注的阳性类别）。
    top_n : int, default 15
        展示重要性最高的前 N 个特征。

    Returns
    -------
    None
    """
    if class_index is None:
        class_index = shap_values.shape[2] - 1
    class_code = classes[class_index]

    sv = shap_values[:, :, class_index]  # shape (n_samples, n_features)
    feature_names = list(X_explain.columns)

    mean_abs = np.mean(np.abs(sv), axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * top_n)), dpi=150)
    # 说明: matplotlib>=3.9 已移除 plt.cm.get_cmap()，改用 matplotlib.colormaps[...] 获取色图。
    cmap = matplotlib.colormaps["coolwarm"]

    for row, feat_idx in enumerate(order[::-1]):
        feature_vals = X_explain.iloc[:, feat_idx].values.astype(float)
        shap_vals = sv[:, feat_idx]

        # 特征取值归一化到 [0, 1] 用于着色（红=高，蓝=低）
        vmin, vmax = np.nanmin(feature_vals), np.nanmax(feature_vals)
        norm_vals = (feature_vals - vmin) / (vmax - vmin) if vmax > vmin else np.full_like(feature_vals, 0.5)

        # 轻微垂直抖动，避免同一行的点完全重叠
        jitter = (np.random.RandomState(0).rand(len(shap_vals)) - 0.5) * 0.6
        ax.scatter(shap_vals, np.full_like(shap_vals, row) + jitter, c=norm_vals, cmap=cmap,
                   s=10, alpha=0.75, vmin=0, vmax=1, edgecolors="none")

    ax.axvline(0, color="grey", lw=1, linestyle="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order[::-1]])
    ax.set_xlabel("SHAP 值 (对预测概率的边际贡献)")
    ax.set_title(f"{model_name} SHAP 摘要图 —— 类别: {class_labels.get(class_code, class_code)}")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("特征取值（低 → 高）")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

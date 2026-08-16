# -*- coding: utf-8 -*-
"""
shap_plots.py
模型可解释性分析：SHAP (SHapley Additive exPlanations) 摘要图与条形图。

策略：
    - 树模型（Decision Tree / Random Forest / XGBoost / LightGBM）使用 shap.TreeExplainer，
      速度快、精确，可直接对 Pipeline 中的树模型本体（跳过无标准化步骤）计算；
    - 其余模型若需要，可使用 shap.Explainer 的通用（Permutation）后端，
      对整条 Pipeline（含标准化）计算，速度较慢，因此默认只对 config.settings.SHAP_MODELS
      中列出的树模型做 SHAP 分析（可按需在 settings 中增删）。

由于本任务是四分类问题，SHAP 值形状为 (n_samples, n_features, n_classes)；
默认展示"非静止型地贫"（类别 2，通常是临床最关心的阳性类别）对应的 SHAP 贡献，
可通过 class_index 参数自由切换。
"""

import matplotlib.pyplot as plt
import numpy as np
import shap

from config import settings


def compute_tree_shap_values(pipeline, X_background, X_explain):
    """
    使用 TreeExplainer 计算树模型的 SHAP 值。

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        训练好的模型 Pipeline，其最后一步（'clf'）须是树模型。
    X_background : pandas.DataFrame
        背景数据集（保留参数以兼容调用方，本函数默认不使用它 —— 见下方说明）。
    X_explain : pandas.DataFrame
        需要被解释的数据（一般是测试集，或测试集的子样本）。

    Returns
    -------
    shap.Explanation
        形状 (n_samples, n_features, n_classes) 的 SHAP 解释对象。

    Notes
    -----
    使用 feature_perturbation="tree_path_dependent"（不需要背景数据集，直接利用树本身
    统计的路径覆盖率估计条件期望）而不是默认的 "interventional" 模式。原因：
    interventional 模式在新版 shap 中不支持 XGBoost 的类别型分裂（categorical split），
    多分类树模型上会直接抛出 NotImplementedError；tree_path_dependent 模式兼容性更好，
    是 shap 官方文档中对 XGBoost / LightGBM / sklearn 树模型的推荐默认选项。
    多分类场景下 SHAP 值输出在"原始得分/对数几率(margin)"空间而非概率空间——
    数值符号与相对大小依然可以正常解释特征的正/负向贡献，只是不能直接读作"概率变化量"。
    """
    clf = pipeline.named_steps.get("clf", pipeline)
    explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
    return explainer(X_explain, check_additivity=False)


def _extract_class_slice(shap_explanation, class_index):
    """
    从多分类 SHAP Explanation 中取出指定类别的切片，兼容不同 shap 版本的返回形状。

    Parameters
    ----------
    shap_explanation : shap.Explanation
    class_index : int
        目标类别在 predict_proba 输出中的列索引。

    Returns
    -------
    shap.Explanation
        形状 (n_samples, n_features) 的单类别切片。
    """
    values = shap_explanation.values
    if values.ndim == 3:
        return shap.Explanation(
            values=values[:, :, class_index],
            base_values=shap_explanation.base_values[:, class_index]
            if np.ndim(shap_explanation.base_values) > 1 else shap_explanation.base_values,
            data=shap_explanation.data,
            feature_names=shap_explanation.feature_names,
        )
    return shap_explanation


def plot_shap_summary(shap_explanation, class_index, class_label, save_path=None):
    """
    绘制 SHAP 摘要图（蜂群图 beeswarm），展示每个特征对预测该类别概率的影响方向与大小。

    Parameters
    ----------
    shap_explanation : shap.Explanation
        compute_tree_shap_values() 的输出。
    class_index : int
        目标类别索引。
    class_label : str
        用于图标题的类别中文名称。
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    class_slice = _extract_class_slice(shap_explanation, class_index)

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(class_slice, show=False)
    plt.title(f"SHAP Summary Plot — Class: {class_label}", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_shap_bar(shap_explanation, class_index, class_label, save_path=None):
    """
    绘制 SHAP 条形图（各特征平均绝对 SHAP 值排序），直观展示特征重要性排名。

    Parameters
    ----------
    shap_explanation : shap.Explanation
    class_index : int
    class_label : str
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    class_slice = _extract_class_slice(shap_explanation, class_index)

    fig = plt.figure(figsize=(8, 6))
    shap.plots.bar(class_slice, show=False)
    plt.title(f"SHAP Feature Importance — Class: {class_label}", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def generate_shap_report(pipeline, X_background, X_explain, feature_names,
                         model_name, class_index=2, class_label=None,
                         figure_dir=None):
    """
    一站式生成某个模型的 SHAP 摘要图 + 条形图，并保存到磁盘。

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        已训练模型。
    X_background : pandas.DataFrame
        SHAP 背景数据集（建议从训练集中抽样 50~100 条，兼顾速度与精度）。
    X_explain : pandas.DataFrame
        待解释数据集（建议使用测试集，若样本量很大可抽样）。
    feature_names : list[str]
    model_name : str
        模型名称，用于文件命名。
    class_index : int, optional
        默认解释类别索引 2（非静止型地贫），可按需修改。
    class_label : str, optional
        默认根据 settings.CLASS_LABELS 自动推断。
    figure_dir : str or Path, optional
        默认使用 settings.FIGURE_DIR。

    Returns
    -------
    dict
        {"summary_fig":.., "bar_fig":.., "shap_values":..}
    """
    figure_dir = figure_dir or settings.FIGURE_DIR
    class_label = class_label or settings.CLASS_LABELS.get(
        settings.CLASS_ORDER[class_index], str(class_index)
    )

    X_explain = X_explain.copy()
    X_explain.columns = feature_names
    X_background = X_background.copy()
    X_background.columns = feature_names

    shap_values = compute_tree_shap_values(pipeline, X_background, X_explain)

    summary_fig = plot_shap_summary(
        shap_values, class_index, class_label,
        save_path=figure_dir / f"shap_summary_{model_name}.png",
    )
    plt.close(summary_fig)

    bar_fig = plot_shap_bar(
        shap_values, class_index, class_label,
        save_path=figure_dir / f"shap_bar_{model_name}.png",
    )
    plt.close(bar_fig)

    return {"summary_fig": summary_fig, "bar_fig": bar_fig, "shap_values": shap_values}

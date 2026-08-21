# -*- coding: utf-8 -*-
"""
shap_plots.py
模型可解释性分析：SHAP (SHapley Additive exPlanations) 摘要图与条形图。

策略（v2，支持全部 7 个模型）：
    - 树模型（Decision Tree / Random Forest / XGBoost / LightGBM）
      使用 shap.TreeExplainer，直接对 Pipeline 中的树模型本体计算，速度快、结果精确，
      feature_perturbation="tree_path_dependent" 不需要背景数据集。
    - 非树模型（Logistic Regression / SVC，经 CalibratedClassifierCV 校准 / MLP 等）
      TreeExplainer 不支持这些模型（会抛出 InvalidModelError: Model type not yet
      supported by TreeExplainer），改用模型无关的 Permutation Explainer
      （shap.Explainer(pipeline.predict_proba, background, algorithm="permutation")），
      直接对整条 Pipeline 的 predict_proba 做黑盒解释，天然兼容 Pipeline 内部的
      标准化等预处理步骤，不需要手动处理特征变换。

      注意：模型无关方法比 TreeExplainer 慢得多（每个样本需要多次调用 predict_proba）。
      本模块默认 max_background / max_explain 均为 None（不限制样本量，使用你传入的
      全部背景数据 / 待解释数据），如果机器性能一般、跑起来太慢，可以在调用
      generate_shap_report() 时显式传入 max_background / max_explain（如各设为 100）
      做下采样以提速。

由于本任务是四分类问题，SHAP 值形状为 (n_samples, n_features, n_classes)；
默认展示"静止型地贫"（类别 1，通常是临床最关心的阳性类别）对应的 SHAP 贡献，
可通过 class_index 参数自由切换。
"""

import matplotlib.pyplot as plt
import numpy as np
import shap

from config import settings

# TreeExplainer 支持的模型类名，按类名字符串匹配（而非 isinstance 导入），
# 这样即使当前环境没装 xgboost / lightgbm 也不会因为 import 失败而报错。
_TREE_MODEL_CLASS_NAMES = {
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "XGBClassifier",
    "LGBMClassifier",
}


def _get_clf_step(pipeline):
    """从 Pipeline 中取出最后一步（'clf'）的模型本体；若传入的本身就不是 Pipeline，原样返回。"""
    return pipeline.named_steps.get("clf", pipeline) if hasattr(pipeline, "named_steps") else pipeline


def _is_tree_model(clf):
    """判断给定的模型对象是否为 TreeExplainer 支持的树模型。"""
    return type(clf).__name__ in _TREE_MODEL_CLASS_NAMES


def compute_tree_shap_values(pipeline, X_background, X_explain):
    """
    使用 TreeExplainer 计算树模型的 SHAP 值。

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        训练好的模型 Pipeline，其最后一步（'clf'）须是树模型。
    X_background : pandas.DataFrame
        背景数据集（保留参数以统一函数签名，tree_path_dependent 模式实际不使用它）。
    X_explain : pandas.DataFrame
        需要被解释的数据（一般是测试集，或测试集的子样本）。

    Returns
    -------
    shap.Explanation
        形状 (n_samples, n_features, n_classes) 的 SHAP 解释对象。

    Notes
    -----
    使用 feature_perturbation="tree_path_dependent"（不需要背景数据集，直接利用树本身
    统计的路径覆盖率估计条件期望），兼容 XGBoost 的类别型分裂（categorical split），
    是 shap 官方文档中对 XGBoost / LightGBM / sklearn 树模型的推荐默认选项。
    多分类场景下 SHAP 值输出在"原始得分/对数几率(margin)"空间而非概率空间——
    数值符号与相对大小依然可以正常解释特征的正/负向贡献，只是不能直接读作"概率变化量"。
    """
    clf = _get_clf_step(pipeline)
    explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
    return explainer(X_explain, check_additivity=False)


def compute_model_agnostic_shap_values(pipeline, X_background, X_explain,
                                       max_background=None, max_explain=None,
                                       algorithm="permutation"):
    """
    对非树模型（Logistic Regression / SVC / MLP 等）使用模型无关的 SHAP 方法。

    直接对整条 Pipeline 的 predict_proba 函数做黑盒解释（而不是只解释 'clf' 这一步），
    这样 Pipeline 内部的标准化等预处理步骤会被自动、正确地考虑进去，不需要手动
    对 X_background / X_explain 做任何变换。

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        训练好的完整 Pipeline（预处理 + 分类器）。
    X_background : pandas.DataFrame
        背景数据集。若 max_background 为 None（默认），使用全部传入样本，
        不做任何下采样；否则下采样到最多 max_background 条。
    X_explain : pandas.DataFrame
        待解释数据集。若 max_explain 为 None（默认），使用全部传入样本；
        否则下采样到最多 max_explain 条。
    max_background : int or None, optional
        背景集下采样上限，默认 None（不限制，使用全部背景数据）。
        注意：这里同时会显式构造 shap.maskers.Independent(background,
        max_samples=len(background))，避免 shap 内部 masker 自身默认
        max_samples=100 的硬编码上限把背景集又偷偷截断回 100 条。
    max_explain : int or None, optional
        待解释集下采样上限，默认 None（不限制，使用全部传入样本）。
        样本量、特征数都较大时模型无关方法会比较慢，机器性能足够可以不设限。
    algorithm : str, optional
        传给 shap.Explainer 的算法名，默认 "permutation"（速度与精度较均衡，
        对多分类 predict_proba 输出兼容性好）。

    Returns
    -------
    shap.Explanation
        形状 (n_samples, n_features, n_classes) 的 SHAP 解释对象。
    """
    if max_background is not None and len(X_background) > max_background:
        background = shap.sample(X_background, max_background, random_state=settings.RANDOM_STATE)
    else:
        background = X_background

    if max_explain is not None and len(X_explain) > max_explain:
        X_explain = X_explain.sample(n=max_explain, random_state=settings.RANDOM_STATE)

    # 显式构造 masker 并把 max_samples 设成背景集实际大小，
    # 否则 shap.Explainer 内部会自己再套一层 max_samples=100 的默认上限，
    # 即使传入的 background 有几百条也会被悄悄截断到 100 条（日志里的
    # "Subsampling to 100 samples" 提示就是这里触发的）。
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(pipeline.predict_proba, masker, algorithm=algorithm)

    # Permutation explainer 需要足够的 max_evals（至少 2*n_features+1），否则部分
    # shap 版本会直接报错要求你手动指定；这里按特征数自动给出一个安全值。
    n_features = X_explain.shape[1]
    max_evals = max(2 * n_features + 1, 500)

    return explainer(X_explain, max_evals=max_evals)


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
        compute_tree_shap_values() / compute_model_agnostic_shap_values() 的输出。
    class_index : int
        目标类别索引。
    class_label : str
        用于图标题的类别名称。
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    class_slice = _extract_class_slice(shap_explanation, class_index)

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(class_slice, show=False, rng=settings.RANDOM_STATE)
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
                         model_name, class_index=1, class_label=None,
                         figure_dir=None, max_background=None, max_explain=None,
                         algorithm="permutation"):
    """
    一站式生成某个模型的 SHAP 摘要图 + 条形图，并保存到磁盘。
    自动根据模型类型选择 TreeExplainer（树模型）或模型无关的 Permutation
    Explainer（非树模型：Logistic Regression / SVC / MLP 等）。

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        已训练模型（完整 Pipeline，含预处理 + 分类器）。
    X_background : pandas.DataFrame
        SHAP 背景数据集（树模型不使用；非树模型会被自动下采样到 max_background 条）。
    X_explain : pandas.DataFrame
        待解释数据集（建议使用测试集；非树模型会被自动下采样到 max_explain 条）。
    feature_names : list[str]
    model_name : str
        模型名称，用于文件命名。
    class_index : int, optional
        默认解释类别索引 1（静止型地贫），可按需修改。
    class_label : str, optional
        默认根据 settings.CLASS_LABELS 自动推断。
    figure_dir : str or Path, optional
        默认使用 settings.FIGURE_DIR。
    max_background : int or None, optional
        仅对非树模型生效，背景集下采样上限，默认 None（不限制，使用全部背景数据）。
    max_explain : int or None, optional
        仅对非树模型生效，待解释集下采样上限，默认 None（不限制，使用全部待解释数据）。
    algorithm : str, optional
        仅对非树模型生效，传给 shap.Explainer 的算法名，默认 "permutation"。

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

    clf = _get_clf_step(pipeline)
    if _is_tree_model(clf):
        shap_values = compute_tree_shap_values(pipeline, X_background, X_explain)
    else:
        shap_values = compute_model_agnostic_shap_values(
            pipeline, X_background, X_explain,
            max_background=max_background, max_explain=max_explain,
            algorithm=algorithm,
        )

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

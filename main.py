# -*- coding: utf-8 -*-
"""
main.py
========
地中海贫血三分类预测模型 —— 主流程编排脚本。

运行方式：
    python main.py

流程：
    Step 1  读取并校验数据
    Step 2  单因素分析（正态性/方差齐性检验 -> ANOVA或Kruskal-Wallis -> FDR校正）
    Step 3  按 7:3 分层随机划分训练集/测试集，并生成训练测试基线对比表
    Step 4  训练集 5 折交叉验证 + 网格/随机搜索调参，训练 7 个模型
    Step 5  训练集交叉验证性能 与 测试集性能评估（含 95% CI）
    Step 6  绘制 ROC / 校准曲线 / DCA / 混淆矩阵 / SHAP 图
    Step 7  测试集 Bootstrap 1000 次模型对比
    Step 8  汇总导出全部表格与模型文件

各步骤均调用独立子模块的函数完成，本文件只负责"编排顺序 + 落盘"，
不包含具体算法实现，方便客户单独调试/替换任意一个环节。
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from config import Config
from data.data_loader import get_features_and_target, load_raw_data, validate_data
from evaluation.bootstrap_compare import bootstrap_model_metrics
from evaluation.metrics import bootstrap_ci_for_metrics, compute_point_metrics, metrics_dict_to_dataframe
from evaluation.plot_calibration import plot_calibration_curves
from evaluation.plot_confusion_matrix import plot_confusion_matrix
from evaluation.plot_dca import plot_dca_curves
from evaluation.plot_roc import plot_multiclass_roc, plot_roc_comparison
from evaluation.shap_analysis import compute_shap_values, plot_shap_bar, plot_shap_summary_beeswarm
from models.model_trainer import ModelTrainer
from models.model_zoo import build_model_specs
from preprocessing.data_split import stratified_split
from stats.univariate_analysis import UnivariateAnalyzer
from tables.table_builder import (
    build_baseline_table,
    build_bootstrap_summary_table,
    build_train_test_comparison_table,
    build_train_vs_test_model_table,
)

warnings.filterwarnings("ignore")

# 中文字体设置：从系统已安装字体中挑选可用的中文字体，避免图中中文显示为方块
import matplotlib.font_manager as _fm

_PREFERRED_CJK_FONTS = [
    "Noto Sans CJK SC", "Noto Sans CJK", "Source Han Sans SC", "Source Han Sans CN",
    "PingFang SC", "Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
]
_available_font_names = {f.name for f in _fm.fontManager.ttflist}
_chosen_font = next((f for f in _PREFERRED_CJK_FONTS if f in _available_font_names), None)

plt.rcParams["axes.unicode_minus"] = False
if _chosen_font:
    plt.rcParams["font.sans-serif"] = [_chosen_font]
    print(f"[字体设置] 图表中文字体使用: {_chosen_font}")
else:
    print(
        "[提示] 系统未检测到中文字体，图表中的中文可能显示为方块。\n"
        "       建议安装思源黑体等中文字体后重新运行，例如 (Ubuntu/Debian):\n"
        "       sudo apt-get install fonts-noto-cjk"
    )


def run_univariate_analysis(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """执行单因素分析并保存表1（三分类基线特征对照表）。

    Parameters
    ----------
    df : pd.DataFrame
        完整数据集。
    cfg : Config
        全局配置。

    Returns
    -------
    pd.DataFrame
        单因素分析原始结果（供后续筛选特征使用）。
    """
    feature_cols = cfg.feature_cols(df.columns.tolist())
    analyzer = UnivariateAnalyzer(group_col=cfg.target_col, class_labels=cfg.class_labels)
    result_df = analyzer.run(df, feature_cols)

    baseline_table = build_baseline_table(result_df, cfg.class_labels)
    baseline_table.to_csv(cfg.table_dir / "table1_baseline_characteristics.csv", index=False, encoding="utf-8-sig")
    print(f"[表格已保存] {cfg.table_dir / 'table1_baseline_characteristics.csv'}")

    return result_df


def compute_train_cv_metrics(
    model_specs: dict, fitted_models: dict, X_train: pd.DataFrame, y_train: pd.Series, cfg: Config
) -> dict:
    """使用与调参时相同的 5 折分层交叉验证，计算训练集的"样本外"性能，避免用重代入(resubstitution)夸大表现。

    Parameters
    ----------
    model_specs : dict
        模型规格字典（用于获取未 refit 的估计器结构以做 cross_val_predict）。
    fitted_models : dict
        已用最优超参数在全训练集上 refit 的模型（用于取其超参数配置）。
    X_train, y_train : pd.DataFrame, pd.Series
        训练集。
    cfg : Config
        全局配置。

    Returns
    -------
    dict
        {模型名: {"accuracy":.., "macro_auc":.., "macro_f1":.., "weighted_f1":..}}
    """
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    results = {}
    for name, best_pipeline in fitted_models.items():
        # 用最优超参数的一份"未训练"克隆，在 5 折上分别 fit/predict，得到样本外预测
        estimator_clone = clone(best_pipeline)
        y_pred = cross_val_predict(estimator_clone, X_train, y_train, cv=cv, method="predict", n_jobs=cfg.n_jobs)
        y_proba = cross_val_predict(estimator_clone, X_train, y_train, cv=cv, method="predict_proba", n_jobs=cfg.n_jobs)
        point = compute_point_metrics(y_train.values, y_pred, y_proba, cfg.class_order)
        results[name] = {
            "accuracy": point["accuracy"],
            "macro_auc": point["macro_auc"],
            "macro_f1": point["macro_f1"],
            "weighted_f1": point["weighted_f1"],
        }
    return results


def evaluate_on_test_set(fitted_models: dict, X_test: pd.DataFrame, y_test: pd.Series, cfg: Config) -> dict:
    """在测试集上计算每个模型的全部指标及 95% Bootstrap 置信区间，并绘制各类评估图。

    Parameters
    ----------
    fitted_models : dict
        已训练好的模型字典。
    X_test, y_test : pd.DataFrame, pd.Series
        测试集。
    cfg : Config
        全局配置。

    Returns
    -------
    dict
        {模型名: bootstrap_ci_for_metrics 的输出}，供后续汇总表使用。
    """
    test_metrics = {}
    models_proba_for_comparison = {}

    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        models_proba_for_comparison[name] = y_proba

        metrics_ci = bootstrap_ci_for_metrics(
            y_test.values, y_pred, y_proba, cfg.class_order,
            n_bootstrap=cfg.n_bootstrap, alpha=cfg.ci_alpha, random_state=cfg.random_state,
        )
        test_metrics[name] = metrics_ci

        metrics_long_df = metrics_dict_to_dataframe(metrics_ci, cfg.class_labels, name)
        safe_name = name.replace(" ", "_")
        metrics_long_df.to_csv(cfg.table_dir / f"metrics_{safe_name}.csv", index=False, encoding="utf-8-sig")

        # --- 绘图：ROC / 校准曲线 / DCA / 混淆矩阵 ---
        plot_multiclass_roc(y_test.values, y_proba, cfg.class_order, cfg.class_labels, name,
                             cfg.figure_dir / f"roc_{safe_name}.png")
        plot_calibration_curves(y_test.values, y_proba, cfg.class_order, cfg.class_labels, name,
                                 cfg.figure_dir / f"calibration_{safe_name}.png")
        plot_dca_curves(y_test.values, y_proba, cfg.class_order, cfg.class_labels, name,
                         cfg.figure_dir / f"dca_{safe_name}.png")
        plot_confusion_matrix(y_test.values, y_pred, cfg.class_order, cfg.class_labels, name,
                               cfg.figure_dir / f"confusion_matrix_{safe_name}.png")
        print(f"[评估完成] {name}")

    plot_roc_comparison(models_proba_for_comparison, y_test.values, cfg.class_order,
                         cfg.figure_dir / "roc_comparison_all_models.png")

    return test_metrics


def run_shap_analysis(fitted_models: dict, X_train: pd.DataFrame, X_test: pd.DataFrame, cfg: Config) -> None:
    """对每个模型执行 SHAP 分析并保存摘要图（蜂群图）与条形图。

    Parameters
    ----------
    fitted_models : dict
        已训练好的模型字典。
    X_train, X_test : pd.DataFrame
        训练集（作为背景数据）与测试集（作为解释样本）。
    cfg : Config
        全局配置。

    Returns
    -------
    None
    """
    # 非树模型的核估计计算量大，解释样本量适当降采样以控制运行时间
    max_explain_samples = 200
    X_explain = X_test if len(X_test) <= max_explain_samples else X_test.sample(max_explain_samples, random_state=cfg.random_state)

    for name, model in fitted_models.items():
        safe_name = name.replace(" ", "_")
        try:
            shap_values = compute_shap_values(model, name, X_train, X_explain)
        except Exception as exc:  # noqa: BLE001 —— SHAP 对不同模型/版本的兼容性问题较多，单模型失败不应中断整体流程
            print(f"[警告] {name} 的 SHAP 分析失败，已跳过。原因: {exc}")
            continue

        plot_shap_bar(shap_values, list(X_explain.columns), cfg.class_labels, cfg.class_order, name,
                      cfg.figure_dir / f"shap_bar_{safe_name}.png")
        plot_shap_summary_beeswarm(shap_values, X_explain, cfg.class_labels, cfg.class_order, name,
                                    cfg.figure_dir / f"shap_summary_{safe_name}.png")
        print(f"[SHAP完成] {name}")


def main() -> None:
    """主入口：按顺序执行完整建模与评估流程。"""
    cfg = Config()

    # ---------- Step 1: 读取数据 ----------
    df = load_raw_data(cfg.raw_data_path)
    feature_cols_all = cfg.feature_cols(df.columns.tolist())
    validate_data(df, cfg.target_col, feature_cols_all)
    print(f"[数据加载完成] 共 {len(df)} 条样本，{len(feature_cols_all)} 个候选特征。")
    print(f"类别分布:\n{df[cfg.target_col].value_counts().sort_index()}")

    # ---------- Step 2: 单因素分析 ----------
    univariate_result = run_univariate_analysis(df, cfg)
    analyzer = UnivariateAnalyzer(group_col=cfg.target_col, class_labels=cfg.class_labels)
    significant_features = analyzer.get_significant_features(univariate_result, use_fdr=False)
    print(f"[单因素分析] 具有统计学差异(P<0.05)的特征共 {len(significant_features)} 个: {significant_features}")

    # 默认使用具有统计学差异的特征建模；如需使用全部特征，把下一行替换为 feature_cols_all
    modeling_features = significant_features if significant_features else feature_cols_all

    # ---------- Step 3: 数据集划分 ----------
    X, y = get_features_and_target(df, modeling_features, cfg.target_col)
    X_train, X_test, y_train, y_test = stratified_split(X, y, cfg.test_size, cfg.random_state)
    print(f"[数据集划分完成] 训练集 {len(X_train)} 例，测试集 {len(X_test)} 例。")

    train_test_table = build_train_test_comparison_table(
        X_train, X_test, y_train, y_test, modeling_features, cfg.target_col, cfg.class_labels
    )
    train_test_table.to_csv(cfg.table_dir / "table2_train_test_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"[表格已保存] {cfg.table_dir / 'table2_train_test_comparison.csv'}")

    # ---------- Step 4: 模型训练与调参 ----------
    model_specs = build_model_specs(random_state=cfg.random_state)
    trainer = ModelTrainer(cv_folds=cfg.cv_folds, random_state=cfg.random_state, n_jobs=cfg.n_jobs)
    fitted_models, best_params, cv_results = trainer.train_all(model_specs, X_train, y_train)
    trainer.save_models(fitted_models, cfg.model_dir)

    best_params_df = pd.DataFrame([{"模型": k, "最优超参数": str(v)} for k, v in best_params.items()])
    best_params_df.to_csv(cfg.table_dir / "best_hyperparameters.csv", index=False, encoding="utf-8-sig")

    # ---------- Step 5: 训练集(CV)与测试集性能评估 ----------
    train_cv_metrics = compute_train_cv_metrics(model_specs, fitted_models, X_train, y_train, cfg)
    test_metrics = evaluate_on_test_set(fitted_models, X_test, y_test, cfg)

    model_comparison_table = build_train_vs_test_model_table(train_cv_metrics, test_metrics)
    model_comparison_table.to_csv(cfg.table_dir / "table3_train_vs_test_model_performance.csv", index=False, encoding="utf-8-sig")
    print(f"[表格已保存] {cfg.table_dir / 'table3_train_vs_test_model_performance.csv'}")

    # ---------- Step 6: SHAP 可解释性分析 ----------
    run_shap_analysis(fitted_models, X_train, X_test, cfg)

    # ---------- Step 7: 测试集 Bootstrap 1000 次模型对比 ----------
    bootstrap_df = bootstrap_model_metrics(
        fitted_models, X_test, y_test, cfg.class_order,
        n_bootstrap=cfg.n_bootstrap, alpha=cfg.ci_alpha, random_state=cfg.random_state,
    )
    bootstrap_table = build_bootstrap_summary_table(bootstrap_df)
    bootstrap_table.to_csv(cfg.table_dir / "table4_bootstrap_model_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"[表格已保存] {cfg.table_dir / 'table4_bootstrap_model_comparison.csv'}")

    print("\n[全部流程完成] 结果已保存至:", cfg.output_dir)


if __name__ == "__main__":
    main()

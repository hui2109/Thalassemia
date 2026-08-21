# -*- coding: utf-8 -*-
"""
main.py
============================================================
地中海贫血四分类预测模型 —— 主运行脚本
============================================================

本脚本按照标准临床预测模型建模流程，依次完成：
    Step 1: Data Loading and Cleaning
    Step 2: Univariate Analysis (Normality -> Homogeneity of Variance -> ANOVA / Kruskal-Wallis)
    Step 3: Select 2-3 Best Derived Thalassemia Indices out of 9 Candidates
    Step 4: Baseline Characteristics Table Across 4 Diagnosis Groups (with Univariate P-values)
    Step 5: Stratified Train/Test Split (7:3)
    Step 6: Baseline Characteristics Comparison - Train vs Test
    Step 7: Train 7 Models with 5-fold CV Hyperparameter Tuning
    Step 8: Model Performance Comparison Table - Train vs Test
    Step 9: Test-set Bootstrap (1000 Resamples) Model Comparison Table
    Step 10: ROC Curves / Calibration Curves (with Slope & Intercept) / Decision Curves / Confusion Matrices
    Step 11: SHAP Interpretability Analysis (Summary + Bar Plots)

本脚本刻意保持 '平铺直叙' 的写法（而非包一层 run_pipeline() 函数），
方便直接阅读、在任意一步之后插入自己的代码、或注释掉不需要的步骤。
所有具体计算逻辑都已拆分到各个子模块中，本文件只负责"调用 + 编排 + 保存结果"。
"""

from config import settings
from data_processing.loader import load_clean_dataset
from data_processing.splitter import stratified_split
from evaluation.model_comparison import (
    collect_test_predictions,
    build_train_test_performance_table,
    build_bootstrap_comparison_table,
)
from models.trainer import train_all_models
from stats_analysis.baseline_tables import build_group_baseline_table, build_split_baseline_table
from stats_analysis.feature_selector import select_derived_indices, build_final_feature_list
from stats_analysis.univariate import run_univariate_analysis
from visualization.calibration_plots import plot_calibration_grid, compute_calibration_summary
from visualization.confusion_matrix_plot import plot_confusion_matrix_grid
from visualization.decision_curve import plot_decision_curve_grid
from visualization.roc_plots import plot_roc_grid
from visualization.shap_plots import generate_shap_report
from visualization.style import apply_style

apply_style()

# =============================================================================
# Step 1: Data Loading and Cleaning
# =============================================================================
print("=" * 60)
print("Step 1: Loading and cleaning data")
print("=" * 60)

raw_df = load_clean_dataset()
print(f"Dataset size: {len(raw_df)}; diagnosis group distribution:\n{raw_df[settings.TARGET_COL].value_counts()}")

# =============================================================================
# Step 2: Univariate analysis (all candidate variables: 11 raw CBC indices + 9 derived indices)
# =============================================================================
print("\n" + "=" * 60)
print("Step 2: Univariate analysis")
print("=" * 60)

all_candidate_vars = settings.BASE_FEATURES + settings.DERIVED_INDEX_CANDIDATES
univariate_result = run_univariate_analysis(raw_df, all_candidate_vars)
univariate_result.to_csv(settings.TABLE_DIR / "table0_univariate_analysis.csv",
                         index=False, encoding="utf-8-sig")
print(univariate_result[["variable", "test_method", "p_value_display", "significant"]])

# =============================================================================
# Step 3: Select 2-3 best derived thalassemia indices out of 9 candidates
# =============================================================================
print("\n" + "=" * 60)
print("Step 3: Derived index selection")
print("=" * 60)

selection = select_derived_indices(raw_df, univariate_result)
selected_derived = selection["selected"]
selection["selection_log"].to_csv(
    settings.TABLE_DIR / "table0b_derived_index_selection_log.csv",
    index=False, encoding="utf-8-sig",
)
print(f"Final selected derived indices: {selected_derived}")

FEATURE_COLS = build_final_feature_list(selected_derived)
print(f"Final feature set ({len(FEATURE_COLS)} features): {FEATURE_COLS}")

# =============================================================================
# Step 4: Baseline Characteristics Table Across 4 Diagnosis Groups (with Univariate P-values)
# =============================================================================
print("\n" + "=" * 60)
print("Step 4: Baseline characteristics table (by diagnosis group)")
print("=" * 60)

# 展示表中同时列出全部 9 个衍生指数（供临床参考对比），但只有 FEATURE_COLS 会入模
group_baseline_table = build_group_baseline_table(raw_df, all_candidate_vars)
group_baseline_table.to_csv(
    settings.TABLE_DIR / "table1_baseline_by_diagnosis_group.csv",
    index=False, encoding="utf-8-sig",
)
print(group_baseline_table)

# =============================================================================
# Step 5: Stratified train/test split (7:3)
# =============================================================================
print("\n" + "=" * 60)
print("Step 5: Train/test split (7:3, stratified)")
print("=" * 60)

X_train, X_test, y_train, y_test = stratified_split(raw_df, FEATURE_COLS)
print(f"Train set size: {len(X_train)}; Test set size: {len(X_test)}")

# =============================================================================
# Step 6: Baseline Characteristics Comparison - Train vs Test
# =============================================================================
print("\n" + "=" * 60)
print("Step 6: Baseline Characteristics Comparison - Train vs Test")
print("=" * 60)

train_full = X_train.copy()
train_full[settings.TARGET_COL] = y_train.values
test_full = X_test.copy()
test_full[settings.TARGET_COL] = y_test.values

split_baseline_table = build_split_baseline_table(train_full, test_full, FEATURE_COLS)
split_baseline_table.to_csv(
    settings.TABLE_DIR / "table2_baseline_train_vs_test.csv",
    index=False, encoding="utf-8-sig",
)
print(split_baseline_table)

# =============================================================================
# Step 7: Train 7 Models with 5-fold CV Hyperparameter Tuning
# =============================================================================
print("\n" + "=" * 60)
print("Step 7: Model training (5-fold CV tuning: GridSearchCV / RandomizedSearchCV)")
print("=" * 60)

trainers = train_all_models(X_train, y_train)

for name, trainer in trainers.items():
    trainer.save()  # 保存到 settings.MODEL_DIR/{name}.joblib，方便直接加载复用

# =============================================================================
# Step 8: Model Performance Comparison Table - Train vs Test
# =============================================================================
print("\n" + "=" * 60)
print("Step 8: Model Performance Comparison Table - Train vs Test")
print("=" * 60)

train_test_table = build_train_test_performance_table(trainers, X_train, y_train, X_test, y_test)
train_test_table.to_csv(
    settings.TABLE_DIR / "table3_train_vs_test_model_performance.csv",
    index=False, encoding="utf-8-sig",
)
print(train_test_table)

# =============================================================================
# Step 9: Test-set Bootstrap (1000 Resamples) Model Comparison Table
# =============================================================================
print("\n" + "=" * 60)
print("Step 9: Test-set bootstrap resampling model comparison table (95% CI)")
print("=" * 60)

bootstrap_table, bootstrap_detail = build_bootstrap_comparison_table(trainers, X_test, y_test)
bootstrap_table.to_csv(
    settings.TABLE_DIR / "table4_bootstrap_model_comparison.csv",
    index=False, encoding="utf-8-sig",
)
for model_name, detail_df in bootstrap_detail.items():
    detail_df.to_csv(
        settings.TABLE_DIR / f"table4b_bootstrap_detail_{model_name}.csv",
        index=False, encoding="utf-8-sig",
    )
print(bootstrap_table)

# =============================================================================
# Step 10: Visualization - ROC / calibration / decision curves / confusion matrices
# =============================================================================
print("\n" + "=" * 60)
print("Step 10: Plotting evaluation figures")
print("=" * 60)

test_predictions = collect_test_predictions(trainers, X_test, y_test)
roc_input = {name: (v["y_true"], v["y_proba"]) for name, v in test_predictions.items()}
cm_input = {name: (v["y_true"], v["y_pred"]) for name, v in test_predictions.items()}

plot_roc_grid(roc_input, save_path=settings.FIGURE_DIR / "fig1_roc_curves.png")
plot_calibration_grid(roc_input, save_path=settings.FIGURE_DIR / "fig2_calibration_curves.png")
plot_decision_curve_grid(roc_input, save_path=settings.FIGURE_DIR / "fig3_decision_curves.png")
plot_confusion_matrix_grid(cm_input, save_path=settings.FIGURE_DIR / "fig4_confusion_matrices.png")

calibration_summary = compute_calibration_summary(roc_input)
calibration_summary.to_csv(
    settings.TABLE_DIR / "table5_calibration_slope_intercept.csv",
    index=False, encoding="utf-8-sig",
)
print(calibration_summary)

# =============================================================================
# Step 11: SHAP interpretability analysis (tree models: Decision Tree / RF / XGBoost / LightGBM)
# =============================================================================
print("\n" + "=" * 60)
print("Step 11: SHAP interpretability analysis")
print("=" * 60)

# 背景数据从训练集中抽样，兼顾计算速度与精度；如需更精确可放大样本量
# 注意：非树模型（LR/SVC/MLP）走模型无关的 Permutation Explainer，
# shap_plots.py 内部还会再自动下采样到 max_background 条，这里只需给出一个不太大的候选池即可
shap_background = X_train.sample(n=len(X_train), random_state=settings.RANDOM_STATE)

for model_name in settings.SHAP_MODELS:
    if model_name not in trainers:
        continue
    print(f"Computing SHAP values for {model_name} ...")
    generate_shap_report(
        pipeline=trainers[model_name].best_estimator_,
        X_background=shap_background,
        X_explain=X_test,
        feature_names=FEATURE_COLS,
        model_name=model_name,
        class_index=1,  # 默认解释"静止型地贫"类别, 可按需修改为 0/1/2/3
    )

print("\n" + "=" * 60)
print("Pipeline completed! Results saved to:")
print(f"  Tables: {settings.TABLE_DIR}")
print(f"  Figures: {settings.FIGURE_DIR}")
print(f"  Models: {settings.MODEL_DIR}")
print("=" * 60)

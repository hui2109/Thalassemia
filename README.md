# 地中海贫血三分类预测模型

基于血常规指标（RBC / HB / HCT / MCV / MCH / MCHC / RDW 等）预测地中海贫血分型的
机器学习项目：**对照组(0)/ 静止型地贫(1)/ 非静止型地贫(2)** 三分类。

---

## 1. 目录结构

```
thalassemia_prediction/
├── config.py                    # 全局配置（路径/随机种子/类别定义等）
├── main.py                      # 主流程编排脚本（唯一运行入口）
├── requirements.txt             # 依赖清单
├── data/
│   ├── thalassemia_raw.csv      # 原始数据
│   └── data_loader.py           # 数据读取与校验
├── stats/
│   ├── normality_tests.py       # 正态性 + 方差齐性检验
│   └── univariate_analysis.py   # 单因素分析（ANOVA / Kruskal-Wallis + FDR校正）
├── preprocessing/
│   └── data_split.py            # 分层随机划分 + 标准化器构建
├── models/
│   ├── model_zoo.py             # 7个模型的估计器与超参数搜索空间定义
│   └── model_trainer.py         # 5折交叉验证调参（GridSearchCV/RandomizedSearchCV）
├── evaluation/
│   ├── metrics.py                # 核心指标 + Bootstrap 95%CI
│   ├── bootstrap_compare.py      # 测试集配对 Bootstrap 模型对比
│   ├── plot_roc.py               # ROC 曲线
│   ├── plot_calibration.py       # 校准曲线 + 校准斜率/截距
│   ├── plot_dca.py               # 决策曲线分析 (DCA)
│   ├── plot_confusion_matrix.py  # 混淆矩阵图
│   └── shap_analysis.py          # SHAP 摘要图 / 条形图
├── tables/
│   └── table_builder.py         # 各类汇总表格生成
└── outputs/                     # 运行产出（自动生成）
    ├── figures/                  # 全部图表 (png)
    ├── tables/                   # 全部表格 (csv)
    └── models/                   # 训练好的模型 (joblib)
```

## 2. 快速开始

```bash
pip install -r requirements.txt
python main.py
```

运行完成后，全部图表、表格、模型文件会保存在 `outputs/` 目录下。

## 3. 方法学要点与关键假设（客户/审稿人常问）

1. **单因素分析**：对每个候选特征先做正态性检验（Shapiro-Wilk，n>5000 时改用
   D'Agostino-Pearson）与方差齐性检验（Levene）；若三组均正态且方差齐，使用单因素方差
   分析（ANOVA），否则使用 Kruskal-Wallis 秩和检验。多重比较使用 Benjamini-Hochberg
   法做 FDR 校正（`stats/univariate_analysis.py`）。

2. **特征筛选**：`main.py` 默认使用单因素分析中 **原始 P<0.05** 的特征进入建模。
   如需改为使用全部特征，或改为 FDR 校正后 P<0.05，只需修改 `main.py` 中
   `modeling_features` 一行的赋值逻辑，其余代码无需改动。

3. **"多模态LR"的实现说明**：本数据集为单一模态的表格化血常规指标，因此按
   "多分类（multinomial）逻辑回归" 实现。详见 `models/model_zoo.py` 顶部注释。

4. **是否标准化**：LR / SVC / MLP 在 Pipeline 中加入 `StandardScaler`；
   决策树 / 随机森林 / XGBoost / LightGBM 不做标准化（树模型对尺度不敏感，
   保留原始单位有利于后续 SHAP 解释）。

5. **调参目标**：默认以 **宏平均 F1 (f1_macro)** 作为 GridSearchCV /
   RandomizedSearchCV 的优化目标，兼顾三个类别（尤其是样本量最少的"静止型地贫"）的
   识别能力，而非仅优化总体准确率。可在 `ModelTrainer(scoring=...)` 中修改。

6. **训练集性能的计算方式**：为避免"重代入"（用训练好的模型直接预测训练集）夸大
   模型表现，表3 中训练集指标是在与调参时相同的 5 折分层交叉验证下，用
   `cross_val_predict` 得到的"样本外"预测计算所得，而非简单重代入。

7. **指标口径**：灵敏度/特异度/PPV/NPV 均按照 **one-vs-rest**（该类别为"阳性"，
   其余两类合并为"阴性"）计算；宏平均 AUC 为三个 one-vs-rest AUC 的宏平均。

8. **校准斜率/截距**：对每个类别的 one-vs-rest 预测概率取对数几率(logit)，
   以其为唯一自变量对真实二分类结局拟合一个不带正则化的逻辑回归，回归系数即为
   校准斜率，截距即为校准截距（理想值分别为 1 和 0）。

9. **决策曲线分析 (DCA)**：净获益公式 `NB(pt) = TP/n - FP/n × pt/(1-pt)`，
   与"全部阳性"和"全部阴性"两条参照策略对比。

10. **SHAP 分析**：树模型使用 `TreeExplainer`（对 Pipeline 中的 clf 直接解释，
    因为树模型未做标准化）；LR/SVC/MLP 使用 `shap.Explainer` 包裹整个 Pipeline 的
    `predict_proba`。摘要图（蜂群图）与条形图为项目自实现（不调用 `shap.plots.*`），
    以保证在不同 shap 版本下行为一致、且方便客户自行调整绘图样式。

11. **测试集 Bootstrap 对比**：对全部 7 个模型使用 **完全相同** 的 1000 组重采样
    索引（配对 Bootstrap），保证模型间比较的公平性；同时提供两两模型指标差值的
    95% CI（`evaluation/bootstrap_compare.py` 中 `pairwise_metric_difference`），
    可用于判断模型间差异是否具有统计学意义。

## 4. 二次开发建议

- 所有函数均标注了参数类型、返回类型与功能说明（Google 风格 docstring），
  可直接用于生成 API 文档。
- 每个功能模块相互独立，替换任意一个环节（如更换特征筛选逻辑、增删模型、
  修改评价指标）都不会影响其他模块，只需保证输入输出接口不变。
- 新增模型：在 `models/model_zoo.py` 的 `build_model_specs()` 中按现有格式
  增加一个 `ModelSpec` 即可，`ModelTrainer` 会自动识别并训练。
- 如需替换数据源：只需保证新数据经过 `data/data_loader.py` 处理后，得到的
  `X (DataFrame)` / `y (Series)` 格式与现有一致，下游全部代码无需改动。

## 5. 已知限制

- SHAP 对 LR/SVC/MLP 使用核估计（Permutation Explainer），计算较慢，
  `main.py` 中默认将解释样本量降采样至 200 例以控制运行时间，可在
  `run_shap_analysis()` 中调整 `max_explain_samples`。
- 若某次 Bootstrap 重采样未覆盖到全部三个类别（小概率事件，测试集类别越不均衡
  越容易出现），该次重采样会被跳过，不计入置信区间计算。

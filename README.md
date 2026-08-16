# 地中海贫血四分类预测模型

基于血常规原始指标与衍生地贫筛查指数，构建 7 个机器学习模型，
对 **对照组(0) / 静止型地贫(1) / 非静止型地贫(2) / 缺铁性贫血(3)** 进行四分类预测。

## 运行环境

- Python 3.13（在 3.10+ 环境下同样可运行）
- 依赖见 `requirements.txt`，安装方式：

```bash
pip install -r requirements.txt
```

## 运行方式

直接用你熟悉的 IDE（PyCharm / VS Code）打开 `main.py`，按需分段运行即可，
不需要在命令行传参。脚本按"第 1 步 ~ 第 11 步"顺序清晰注释，可以整段跑完，
也可以只跑其中某几步（例如只重新画图，不用重新训练模型）。

## 目录结构

```
thalassemia_pred/
├── main.py                     # 主运行脚本（从数据到全部结果的完整流程）
├── requirements.txt
├── data/
│   └── thalassemia_data.csv    # 原始数据
├── config/
│   └── settings.py             # 全局配置：路径、随机种子、列名、类别标签等
├── data_processing/
│   ├── loader.py                # 数据读取与清洗
│   └── splitter.py              # 分层随机划分训练/测试集
├── stats_analysis/
│   ├── univariate.py            # 单因素分析（正态性/方差齐性检验 + ANOVA/Kruskal-Wallis）
│   ├── feature_selector.py      # 9 个衍生地贫指数的筛选逻辑
│   └── baseline_tables.py       # 基线特征对照表（分组对比 / 训练测试集对比）
├── models/
│   ├── model_factory.py         # 7 个模型的 Pipeline 与超参数搜索空间定义
│   └── trainer.py               # ModelTrainer 类：5 折 CV 调参 + 训练 + 保存
├── evaluation/
│   ├── metrics.py                # 各类评价指标的纯计算函数
│   ├── bootstrap.py              # Bootstrap 重采样置信区间计算
│   └── model_comparison.py       # 训练/测试对比表、Bootstrap 对比表
├── visualization/
│   ├── style.py                  # 全局绘图风格（中文字体等）
│   ├── roc_plots.py              # ROC 曲线
│   ├── calibration_plots.py      # 校准曲线 + 校准斜率/截距
│   ├── decision_curve.py         # 决策曲线分析 (DCA)
│   ├── confusion_matrix_plot.py  # 混淆矩阵热力图
│   └── shap_plots.py             # SHAP 摘要图 / 条形图
└── outputs/
    ├── tables/                   # 所有 csv 结果表
    ├── figures/                  # 所有 png 图片
    └── models/                   # 训练好的模型（joblib 格式）
```

## 二次开发建议

- 所有函数均为"输入 -> 输出"的纯函数风格（模型训练类除外），每个函数职责单一，
  互相之间没有隐藏的全局状态依赖，可以放心单独修改、替换、单元测试。
- 新增一个模型：只需在 `models/model_factory.py` 中新增一个 `_build_xxx()` 函数并注册到
  `MODEL_BUILDERS` 字典，`main.py` 无需任何改动。
- 调整入模特征：修改 `config/settings.py` 中的 `BASE_FEATURES` /
  `N_DERIVED_INDEX_TO_KEEP` / `DERIVED_INDEX_CORR_THRESHOLD` 即可。
- 所有中间结果（模型、表格、图片）均已落盘到 `outputs/` 目录，可以只重跑某一步，
  复用之前保存好的结果。

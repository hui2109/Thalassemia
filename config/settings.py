# -*- coding: utf-8 -*-
"""
settings.py
全局配置文件：路径、随机种子、列名定义、类别标签等所有可能需要自行调整的常量都集中于此。
把所有"魔法数字/字符串"都放在这里,方便后续二次开发时统一修改,而不必去每个模块里翻找。
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. 路径配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "thalassemia_data.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

for _dir in (OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, MODEL_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. 随机性与重复实验控制
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_CV_FOLDS = 5  # 训练集内部交叉验证折数
N_BOOTSTRAP = 1000  # 测试集 bootstrap 重采样次数
TEST_SIZE = 0.3  # 训练/测试集划分比例 (7:3)
CI_ALPHA = 0.05  # 95% 置信区间对应的显著性水平

# ---------------------------------------------------------------------------
# 3. 数据列名定义
# ---------------------------------------------------------------------------
TARGET_COL = "diagnosis"

# 血常规原始指标（这些一定会被纳入模型）
BASE_FEATURES = [
    "age", "RBC", "HB", "HCT", "MCV", "MCH",
    "MCHC", "RDW-SD", "RDW-CV", "PLT", "PDW",
]

# 9 个衍生地贫筛查指数：由血常规原始指标计算得出，彼此高度共线，
# 不能全部入模，需通过单因素分析 + 相关性分析挑选 2~3 个最具统计学意义且信息冗余最小的指标。
DERIVED_INDEX_CANDIDATES = [
    "England & Fraser index (E&F)",
    "Green & King index (G&K)",
    "Red Cell Distribution Width index (RDWI)",
    "Ricerca index (RI)",
    "Siridah index (SI)",
    "Srivastava formula (SF)",
    "Kandhro 1 formula (KF1)",
    "Shine and Lal (S&L)",
    "Mentzer (MI)",
]

# 从 9 个衍生指标中最终挑选进入模型的数量上限
N_DERIVED_INDEX_TO_KEEP = 3
# 衍生指标之间若相关系数（Pearson |r|）超过该阈值，则视为信息重复，只保留统计学证据更强的一个
DERIVED_INDEX_CORR_THRESHOLD = 0.9

# ---------------------------------------------------------------------------
# 4. 类别标签定义（SCI 论文图表要求全英文，因此标签统一使用英文）
# ---------------------------------------------------------------------------
CLASS_LABELS = {
    0: "Control",
    1: "Silent Thalassemia",
    2: "Non-silent Thalassemia",
    3: "Iron Deficiency Anemia",
}
CLASS_ORDER = [0, 1, 2, 3]

# ---------------------------------------------------------------------------
# 5. 模型清单（用于结果表统一排序与展示名，全部使用英文）
# ---------------------------------------------------------------------------
MODEL_DISPLAY_NAMES = {
    "LogisticRegression": "Multinomial Logistic Regression",
    "SVC": "Support Vector Classifier",
    "DecisionTree": "Decision Tree",
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "MLP": "Multilayer Perceptron",
}
MODEL_ORDER = list(MODEL_DISPLAY_NAMES.keys())

# 需要做 SHAP 解释的模型（树模型速度快，优先展示；如需全部展示可自行把列表补全）
# SHAP_MODELS = ["DecisionTree", "RandomForest", "XGBoost", "LightGBM"]
SHAP_MODELS = list(MODEL_DISPLAY_NAMES.keys())

# -*- coding: utf-8 -*-
"""
model_zoo.py
=============
集中定义本项目使用的 7 个三分类模型：
    1. LR  —— 多分类逻辑回归（Logistic Regression, multinomial）
    2. SVC —— 支持向量分类器
    3. Decision Tree —— 决策树
    4. Random Forest —— 随机森林
    5. XGBoost
    6. LightGBM
    7. MLP —— 多层感知机（神经网络）

每个模型以 `ModelSpec` 描述：估计器实例、超参数搜索空间、
调参方式（GridSearchCV 穷举 或 RandomizedSearchCV 随机搜索）、
以及是否需要标准化。客户如需增删模型或调整搜索空间，只需修改本文件，
不影响训练、评估等其他模块。

【关于"多模态LR"的说明】
本数据集为单一模态的表格化血常规指标，因此这里将其实现为
标准的"多分类（多项式）逻辑回归"（multinomial Logistic Regression）。
若客户后续接入影像等其他模态数据，可在 build_model_specs 中
将 LR 替换为对多模态特征拼接后的输入，接口保持不变。
"""

from dataclasses import dataclass
from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False


@dataclass
class ModelSpec:
    """单个模型的规格说明。

    Attributes
    ----------
    name : str
        模型名称（用于图表、表格标注）。
    estimator : object
        sklearn 兼容的未拟合估计器实例。
    param_distributions : dict
        超参数搜索空间。
    search_type : str
        "grid" 使用 GridSearchCV；"random" 使用 RandomizedSearchCV。
    needs_scaling : bool
        是否需要在 Pipeline 中加入 StandardScaler。
    n_iter : int
        当 search_type="random" 时，随机搜索的采样次数。
    """

    name: str
    estimator: object
    param_distributions: dict
    search_type: str = "grid"
    needs_scaling: bool = False
    n_iter: int = 30


def build_model_specs(random_state: int = 42) -> Dict[str, ModelSpec]:
    """构建全部 7 个模型的规格字典。

    Parameters
    ----------
    random_state : int, default 42
        传递给各模型/搜索器的随机种子，保证结果可复现。

    Returns
    -------
    Dict[str, ModelSpec]
        {模型名: ModelSpec}，遍历顺序即报告中呈现的模型顺序。

    Raises
    ------
    ImportError
        若环境中未安装 xgboost 或 lightgbm，会在调用本函数时抛出，
        并提示客户执行 `pip install xgboost lightgbm`。
    """
    if not _HAS_XGBOOST:
        raise ImportError("未检测到 xgboost，请先执行: pip install xgboost")
    if not _HAS_LIGHTGBM:
        raise ImportError("未检测到 lightgbm，请先执行: pip install lightgbm")

    specs: Dict[str, ModelSpec] = {}

    # 1. 多分类逻辑回归 -----------------------------------------------------
    # 说明: 较新版本 scikit-learn (>=1.7) 已移除 LogisticRegression 的 multi_class 参数，
    # lbfgs/newton-cg 等求解器在多分类场景下会自动按多项式(multinomial)方式求解，无需显式指定。
    specs["LR"] = ModelSpec(
        name="Logistic Regression",
        estimator=LogisticRegression(max_iter=5000, random_state=random_state),
        param_distributions={
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs", "newton-cg"],
        },
        search_type="grid",
        needs_scaling=True,
    )

    # 2. 支持向量分类器 -------------------------------------------------------
    specs["SVC"] = ModelSpec(
        name="SVC",
        estimator=SVC(probability=True, random_state=random_state),
        param_distributions={
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__kernel": ["rbf", "linear", "poly"],
            "clf__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        },
        search_type="grid",
        needs_scaling=True,
    )

    # 3. 决策树 ---------------------------------------------------------------
    specs["Decision Tree"] = ModelSpec(
        name="Decision Tree",
        estimator=DecisionTreeClassifier(random_state=random_state),
        param_distributions={
            "clf__max_depth": [3, 4, 5, 6, 8, 10, None],
            "clf__min_samples_split": [2, 5, 10, 20],
            "clf__min_samples_leaf": [1, 2, 5, 10],
            "clf__criterion": ["gini", "entropy"],
        },
        search_type="grid",
        needs_scaling=False,
    )

    # 4. 随机森林 ---------------------------------------------------------------
    specs["Random Forest"] = ModelSpec(
        name="Random Forest",
        estimator=RandomForestClassifier(random_state=random_state, n_jobs=1),
        param_distributions={
            "clf__n_estimators": [100, 200, 300, 500],
            "clf__max_depth": [3, 5, 8, 10, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", None],
        },
        search_type="random",
        needs_scaling=False,
        n_iter=40,
    )

    # 5. XGBoost -----------------------------------------------------------
    specs["XGBoost"] = ModelSpec(
        name="XGBoost",
        # 说明: use_label_encoder 参数在 xgboost>=2.0 中已被彻底移除，不能再传入。
        estimator=XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=1,
        ),
        param_distributions={
            "clf__n_estimators": [100, 200, 300, 500],
            "clf__max_depth": [3, 4, 5, 6, 8],
            "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "clf__subsample": [0.6, 0.8, 1.0],
            "clf__colsample_bytree": [0.6, 0.8, 1.0],
            "clf__reg_lambda": [0.1, 1, 5, 10],
        },
        search_type="random",
        needs_scaling=False,
        n_iter=50,
    )

    # 6. LightGBM ------------------------------------------------------------
    specs["LightGBM"] = ModelSpec(
        name="LightGBM",
        estimator=LGBMClassifier(
            objective="multiclass",
            num_class=3,
            random_state=random_state,
            n_jobs=1,
            verbosity=-1,
        ),
        param_distributions={
            "clf__n_estimators": [100, 200, 300, 500],
            "clf__max_depth": [-1, 3, 5, 8],
            "clf__num_leaves": [15, 31, 63, 127],
            "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "clf__subsample": [0.6, 0.8, 1.0],
            "clf__colsample_bytree": [0.6, 0.8, 1.0],
        },
        search_type="random",
        needs_scaling=False,
        n_iter=50,
    )

    # 7. 多层感知机（神经网络） -----------------------------------------------
    specs["MLP"] = ModelSpec(
        name="MLP",
        estimator=MLPClassifier(max_iter=2000, random_state=random_state, early_stopping=True),
        param_distributions={
            "clf__hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32), (64, 32, 16)],
            "clf__alpha": [0.0001, 0.001, 0.01, 0.1],
            "clf__learning_rate_init": [0.001, 0.005, 0.01],
            "clf__activation": ["relu", "tanh"],
        },
        search_type="random",
        needs_scaling=True,
        n_iter=30,
    )

    return specs

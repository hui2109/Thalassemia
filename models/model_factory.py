# -*- coding: utf-8 -*-
"""
model_factory.py
模型工厂：集中定义 7 个分类器各自的 Pipeline（含必要的标准化步骤）与超参数搜索空间。

设计原则：
    - 每个模型的构建逻辑独立成一个 `_build_xxx()` 函数，互不影响；新增/替换模型时
      只需要新增一个函数并注册到 MODEL_BUILDERS 字典即可，不需要改动其它代码。
    - 需要特征标准化的模型（LR / SVC / MLP）用 sklearn Pipeline 自动完成 StandardScaler，
      避免在训练/测试集之间发生数据泄漏（scaler 只在训练折上 fit）。
    - 每个 builder 返回 (pipeline, param_distributions, search_strategy)，
      search_strategy ∈ {"grid", "random"}，由 trainer.py 据此选择 GridSearchCV / RandomizedSearchCV。
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import settings


def _build_logistic_regression():
    """多分类 Logistic Regression（'多模态 LR'：multinomial softmax）。"""
    # 注：sklearn >=1.5 起 LogisticRegression 不再需要显式传 multi_class 参数，
    # lbfgs/newton-cg 等求解器在多分类场景下会自动采用 multinomial（softmax）损失。
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            random_state=settings.RANDOM_STATE,
        )),
    ])
    # 不显式搜索 penalty：不同 sklearn 版本对 penalty 参数的支持方式差异较大，
    # 默认的 L2 正则化配合 C 的搜索已能覆盖绝大多数场景。
    param_grid = {
        "clf__C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        "clf__solver": ["lbfgs", "newton-cg"],
    }
    return pipeline, param_grid, "grid"


def _build_svc():
    """
    支持向量机分类器。

    sklearn >= 1.9 起 SVC(probability=True) 已弃用（内部实现效率低且results不稳定），
    官方推荐做法是用 CalibratedClassifierCV 包装一个不开启 probability 的 SVC，
    通过交叉验证做概率校准，得到 predict_proba —— 这也是 AUC / 校准曲线 / DCA 分析
    所必需的概率输出方式，同时比旧的 probability=True 更稳健。
    """
    from sklearn.calibration import CalibratedClassifierCV

    base_svc = SVC(random_state=settings.RANDOM_STATE)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(base_svc, method="sigmoid", cv=3, ensemble=False)),
    ])
    # CalibratedClassifierCV 包装后，基分类器的超参数需要通过 estimator__ 前缀访问
    param_grid = {
        "clf__estimator__C": [0.1, 1, 10, 100],
        "clf__estimator__gamma": ["scale", "auto", 0.01, 0.1],
        "clf__estimator__kernel": ["rbf", "linear"],
    }
    return pipeline, param_grid, "grid"


def _build_decision_tree():
    """单棵决策树，作为可解释性基线模型。"""
    pipeline = Pipeline([
        ("clf", DecisionTreeClassifier(random_state=settings.RANDOM_STATE)),
    ])
    param_grid = {
        "clf__max_depth": [3, 4, 5, 6, 8, 10, None],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__criterion": ["gini", "entropy"],
    }
    return pipeline, param_grid, "grid"


def _build_random_forest():
    """随机森林。搜索空间较大，使用 RandomizedSearchCV 提升调参效率。"""
    pipeline = Pipeline([
        ("clf", RandomForestClassifier(random_state=settings.RANDOM_STATE)),
    ])
    param_distributions = {
        "clf__n_estimators": [100, 200, 300, 500, 800],
        "clf__max_depth": [3, 5, 8, 10, 15, None],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__max_features": ["sqrt", "log2", None],
    }
    return pipeline, param_distributions, "grid"


def _build_xgboost():
    """XGBoost 多分类（softprob）。延迟导入，避免环境未安装时影响其它模型运行。"""
    from xgboost import XGBClassifier

    n_classes = len(settings.CLASS_ORDER)
    pipeline = Pipeline([
        ("clf", XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=settings.RANDOM_STATE,
            tree_method="hist",
        )),
    ])
    param_distributions = {
        "clf__n_estimators": [100, 200, 300, 500],
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__reg_lambda": [0.1, 1, 5, 10],
    }
    return pipeline, param_distributions, "grid"


def _build_lightgbm():
    """LightGBM 多分类（multiclass softmax）。延迟导入。"""
    from lightgbm import LGBMClassifier

    pipeline = Pipeline([
        ("clf", LGBMClassifier(
            objective="multiclass",
            num_class=len(settings.CLASS_ORDER),
            random_state=settings.RANDOM_STATE,
            verbosity=-1,
        )),
    ])
    param_distributions = {
        "clf__n_estimators": [100, 200, 300, 500],
        "clf__max_depth": [-1, 3, 5, 8],
        "clf__num_leaves": [15, 31, 63, 127],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__reg_lambda": [0.0, 0.1, 1, 5],
    }
    return pipeline, param_distributions, "random"


def _build_mlp():
    """多层感知机（前馈神经网络）。"""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            max_iter=4000,
            early_stopping=True,
            random_state=settings.RANDOM_STATE,
        )),
    ])
    param_distributions = {
        "clf__hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32), (64, 32, 16)],
        "clf__activation": ["relu", "tanh"],
        "clf__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
        "clf__learning_rate_init": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
    }
    return pipeline, param_distributions, "random"


# 模型名 -> 构建函数。新增模型时只需在此处添加一行注册。
MODEL_BUILDERS = {
    "LogisticRegression": _build_logistic_regression,
    "SVC": _build_svc,
    "DecisionTree": _build_decision_tree,
    "RandomForest": _build_random_forest,
    "XGBoost": _build_xgboost,
    "LightGBM": _build_lightgbm,
    "MLP": _build_mlp,
}


def get_model_spec(model_name):
    """
    根据模型名获取其 (pipeline, 参数搜索空间, 搜索策略)。

    Parameters
    ----------
    model_name : str
        必须是 settings.MODEL_ORDER 中的名称之一。

    Returns
    -------
    tuple
        (sklearn.pipeline.Pipeline, dict, str)
    """
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"未知模型名: {model_name}，可选值: {list(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[model_name]()

# -*- coding: utf-8 -*-
"""
trainer.py
统一负责"5 折交叉验证 + 超参数搜索 + 训练"这一整套流程。

用类结构 ModelTrainer 封装,是因为训练过程需要维护一些状态（已训练模型、
CV 最优参数、CV 分数等），用类比一堆散装函数更清晰，也方便以后
想单独重训某一个模型，或者把训练好的模型序列化保存/加载。
"""

import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from config import settings
from models.model_factory import get_model_spec


class ModelTrainer:
    """
    单个模型的训练器：给定模型名称，自动完成 5 折 CV 调参 + 用全部训练集重新拟合最优参数。

    Attributes
    ----------
    model_name : str
        模型名称（须在 settings.MODEL_ORDER 中）。
    best_estimator_ : sklearn.pipeline.Pipeline or None
        调参完成后，用全部训练数据重新拟合出的最优模型；fit() 之前为 None。
    best_params_ : dict or None
        网格/随机搜索得到的最优超参数。
    cv_results_ : pandas.DataFrame or None
        完整的交叉验证结果（每组超参数在 5 折上的表现），便于复核调参过程。
    best_cv_score_ : float or None
        最优超参数对应的 5 折交叉验证平均得分（macro-F1）。
    """

    def __init__(self, model_name, n_cv_folds=None, n_iter_random=80, scoring="f1_macro"):
        """
        Parameters
        ----------
        model_name : str
            模型名称。
        n_cv_folds : int, optional
            交叉验证折数，默认使用 settings.N_CV_FOLDS。
        n_iter_random : int, optional
            当使用 RandomizedSearchCV 时的采样次数。
        scoring : str, optional
            调参优化目标，默认宏平均 F1（对不均衡的四分类问题比 accuracy 更合理）。
        """
        self.model_name = model_name
        self.n_cv_folds = n_cv_folds or settings.N_CV_FOLDS
        self.n_iter_random = n_iter_random
        self.scoring = scoring

        self.best_estimator_ = None
        self.best_params_ = None
        self.cv_results_ = None
        self.best_cv_score_ = None

    def fit(self, X_train, y_train):
        """
        执行 5 折交叉验证调参，并用最优超参数在全部训练集上重新拟合。

        Parameters
        ----------
        X_train : pandas.DataFrame
        y_train : pandas.Series

        Returns
        -------
        ModelTrainer
            返回 self，便于链式调用。
        """
        pipeline, param_space, strategy = get_model_spec(self.model_name)
        cv = StratifiedKFold(
            n_splits=self.n_cv_folds, shuffle=True, random_state=settings.RANDOM_STATE
        )

        if strategy == "grid":
            searcher = GridSearchCV(
                pipeline, param_grid=param_space, scoring=self.scoring,
                cv=cv, n_jobs=-1, refit=True,
            )
        else:
            searcher = RandomizedSearchCV(
                pipeline, param_distributions=param_space, n_iter=self.n_iter_random,
                scoring=self.scoring, cv=cv, n_jobs=-1, refit=True,
                random_state=settings.RANDOM_STATE,
            )

        searcher.fit(X_train, y_train)

        self.best_estimator_ = searcher.best_estimator_
        self.best_params_ = searcher.best_params_
        self.best_cv_score_ = searcher.best_score_
        self.cv_results_ = searcher.cv_results_
        return self

    def save(self, path=None):
        """
        将最优模型序列化保存到磁盘（joblib 格式）。

        Parameters
        ----------
        path : str or Path, optional
            保存路径，默认 settings.MODEL_DIR / f"{model_name}.joblib"。
        """
        path = path or (settings.MODEL_DIR / f"{self.model_name}.joblib")
        joblib.dump(self, path)
        return path


def train_all_models(X_train, y_train, model_names=None):
    """
    批量训练全部模型（每个模型独立完成 5 折 CV 调参）。

    Parameters
    ----------
    X_train : pandas.DataFrame
    y_train : pandas.Series
    model_names : list[str], optional
        默认训练 settings.MODEL_ORDER 中的全部 7 个模型。

    Returns
    -------
    dict[str, ModelTrainer]
        模型名 -> 已完成训练的 ModelTrainer 实例。
    """
    model_names = model_names or settings.MODEL_ORDER
    trainers = {}
    for name in model_names:
        print(f"[INFO] Tuning and training model: {name} ...")
        trainer = ModelTrainer(name).fit(X_train, y_train)
        print(f"[INFO] {name} done. Best 5-fold CV Macro-F1 = {trainer.best_cv_score_:.4f}")
        print(f"[INFO] {name} best hyperparameters: {trainer.best_params_}")
        trainers[name] = trainer
    return trainers

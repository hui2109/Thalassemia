# -*- coding: utf-8 -*-
"""
model_trainer.py
==================
将 model_zoo 中定义的模型规格转化为可训练的 sklearn Pipeline，
并使用 GridSearchCV / RandomizedSearchCV 在训练集上做 5 折分层交叉验证调参。

输出统一为 {模型名: 已 fit 的最优 Pipeline}，供 evaluation 模块直接调用
`.predict()` / `.predict_proba()`。
"""

from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from models.model_zoo import ModelSpec
from preprocessing.data_split import build_scaler


class ModelTrainer:
    """模型训练器：负责把一个 ModelSpec 训练成调参后的最优模型。"""

    def __init__(self, cv_folds: int, random_state: int, scoring: str = "f1_macro", n_jobs: int = -1):
        """
        Parameters
        ----------
        cv_folds : int
            交叉验证折数。
        random_state : int
            随机种子。
        scoring : str, default "f1_macro"
            调参优化的目标指标。三分类任务下宏平均 F1 兼顾了各类别（含少数类）的表现。
        n_jobs : int, default -1
            并行搜索使用的进程数。
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.scoring = scoring
        self.n_jobs = n_jobs

    def _build_pipeline(self, spec: ModelSpec) -> Pipeline:
        """根据 ModelSpec 构建 sklearn Pipeline（按需插入标准化步骤）。

        Parameters
        ----------
        spec : ModelSpec
            模型规格。

        Returns
        -------
        Pipeline
            未拟合的 Pipeline，step 名固定为 "scaler"（可选）与 "clf"，
            与 model_zoo 中 param_distributions 的 "clf__xxx" 键保持一致。
        """
        steps = []
        if spec.needs_scaling:
            steps.append(("scaler", build_scaler()))
        steps.append(("clf", spec.estimator))
        return Pipeline(steps)

    def tune_one_model(
        self, spec: ModelSpec, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[Pipeline, dict, pd.DataFrame]:
        """对单个模型执行 5 折交叉验证调参，并用全部训练集拟合最优模型。

        Parameters
        ----------
        spec : ModelSpec
            模型规格。
        X_train : pd.DataFrame
            训练集特征。
        y_train : pd.Series
            训练集标签。

        Returns
        -------
        Tuple[Pipeline, dict, pd.DataFrame]
            (best_estimator, best_params, cv_results_df)
            best_estimator 已经在整个训练集上 refit；
            cv_results_df 为交叉验证过程的完整结果，供审计/复现使用。
        """
        pipeline = self._build_pipeline(spec)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        if spec.search_type == "grid":
            searcher = GridSearchCV(
                estimator=pipeline,
                param_grid=spec.param_distributions,
                scoring=self.scoring,
                cv=cv,
                n_jobs=self.n_jobs,
                refit=True,
            )
        elif spec.search_type == "random":
            searcher = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=spec.param_distributions,
                n_iter=spec.n_iter,
                scoring=self.scoring,
                cv=cv,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                refit=True,
            )
        else:
            raise ValueError(f"未知的 search_type: {spec.search_type}，应为 'grid' 或 'random'")

        searcher.fit(X_train, y_train)
        cv_results_df = pd.DataFrame(searcher.cv_results_)
        return searcher.best_estimator_, searcher.best_params_, cv_results_df

    def train_all(
        self, specs: Dict[str, ModelSpec], X_train: pd.DataFrame, y_train: pd.Series, verbose: bool = True
    ) -> Tuple[Dict[str, Pipeline], Dict[str, dict], Dict[str, pd.DataFrame]]:
        """批量训练全部模型。

        Parameters
        ----------
        specs : Dict[str, ModelSpec]
            由 `model_zoo.build_model_specs()` 得到的模型规格字典。
        X_train, y_train : pd.DataFrame, pd.Series
            训练集。
        verbose : bool, default True
            是否打印训练进度。

        Returns
        -------
        Tuple[Dict[str, Pipeline], Dict[str, dict], Dict[str, pd.DataFrame]]
            (fitted_models, best_params, cv_results)，三者的 key 均为模型名。
        """
        fitted_models: Dict[str, Pipeline] = {}
        best_params: Dict[str, dict] = {}
        cv_results: Dict[str, pd.DataFrame] = {}

        for name, spec in specs.items():
            if verbose:
                print(f"[训练中] {name} ...")
            model, params, cv_df = self.tune_one_model(spec, X_train, y_train)
            fitted_models[name] = model
            best_params[name] = params
            cv_results[name] = cv_df
            if verbose:
                print(f"[完成] {name} 最优参数: {params}")

        return fitted_models, best_params, cv_results

    @staticmethod
    def save_models(fitted_models: Dict[str, Pipeline], save_dir) -> None:
        """将全部已训练模型持久化到磁盘。

        Parameters
        ----------
        fitted_models : Dict[str, Pipeline]
            训练好的模型字典。
        save_dir : Path or str
            保存目录，每个模型保存为 "<模型名>.joblib"。

        Returns
        -------
        None
        """
        from pathlib import Path

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, model in fitted_models.items():
            safe_name = name.replace(" ", "_")
            joblib.dump(model, save_dir / f"{safe_name}.joblib")

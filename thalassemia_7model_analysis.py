# -*- coding: utf-8 -*-
"""
Three-class thalassemia prediction: 7-model analysis pipeline
Outcome:
    diagnosis = 0 Control
    diagnosis = 1 Silent thalassemia (重点关注)
    diagnosis = 2 Non-silent thalassemia

Models:
    1) Multinomial Logistic Regression (Softmax)
    2) SVM / SVC (RBF kernel)
    3) Decision Tree
    4) Random Forest
    5) XGBoost
    6) LightGBM
    7) MLP

Recommended Python packages:
    pip install pandas numpy scipy scikit-learn statsmodels matplotlib openpyxl
    pip install xgboost lightgbm shap imbalanced-learn

Notes:
    - Column "no" is treated as an ID and excluded from predictors.
    - All tuning occurs in the training set only.
    - The independent test set is used once for final evaluation.
    - Class 1 is emphasized with AUC, sensitivity, specificity, F1, PPV, NPV,
      calibration, DCA and SHAP.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    shapiro, levene, f_oneway, kruskal, ttest_ind, mannwhitneyu,
    loguniform, randint, uniform
)

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, roc_auc_score,
    f1_score, recall_score, precision_score, confusion_matrix,
    roc_curve, auc
)
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import statsmodels.api as sm
import shap

warnings.filterwarnings("ignore")

# =========================================================
# 0. Global settings
# =========================================================

RANDOM_STATE = 2026
TEST_SIZE = 0.30
N_BOOTSTRAP = 1000
N_SPLITS_CV = 5
N_ITER_RANDOM_SEARCH = 30

CLASSES = np.array([0, 1, 2])
CLASS_NAMES = {
    0: "Control",
    1: "Silent thalassemia",
    2: "Non-silent thalassemia",
}

TARGET = "diagnosis"
ID_COLS = ["no"]

# Change this path if necessary.
DATA_PATH = Path("./个人自行清洗数据结果汇总.csv")

OUTPUT_DIR = Path("./thalassemia_7model_results")
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. Load and audit data
# =========================================================

def load_data(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    allowed = set(CLASSES.tolist())
    observed = set(pd.Series(df[TARGET]).dropna().unique().tolist())
    if not observed.issubset(allowed):
        raise ValueError(
            f"Unexpected diagnosis values: {observed}. "
            f"Expected only {allowed}."
        )

    feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET]]
    X = df[feature_cols].copy()
    y = df[TARGET].astype(int).copy()

    # This dataset is numeric. Fail loudly if an unexpected object column appears.
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            "Non-numeric predictors detected. Add encoding before modeling: "
            + ", ".join(non_numeric)
        )

    print("=" * 70)
    print("DATA AUDIT")
    print("=" * 70)
    print(f"N = {len(df)}")
    print(f"Number of raw columns = {df.shape[1]}")
    print(f"Number of model predictors = {X.shape[1]}")
    print(f"Predictors = {feature_cols}")
    print("\nClass distribution:")
    counts = y.value_counts().sort_index()
    for c in CLASSES:
        n = int(counts.get(c, 0))
        pct = n / len(y) * 100
        print(f"  {c} ({CLASS_NAMES[c]}): {n} ({pct:.1f}%)")

    print(f"\nTotal missing values = {int(df.isna().sum().sum())}")

    # Duplicate ID check if ID exists
    for col in ID_COLS:
        if col in df.columns:
            print(f"Duplicate values in {col}: {int(df[col].duplicated().sum())}")

    return df, X, y


# =========================================================
# 2. Baseline Table 1: 3-class comparison
# =========================================================

def _safe_shapiro(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    # scipy Shapiro can handle this sample size; cap for general robustness.
    if len(x) > 5000:
        x = np.random.default_rng(RANDOM_STATE).choice(x, 5000, replace=False)
    return float(shapiro(x).pvalue)


def baseline_three_groups(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str = TARGET
) -> pd.DataFrame:
    """
    Automated descriptive table:
    - If each group's Shapiro p>0.05 and Levene p>0.05:
        mean ± SD + one-way ANOVA
    - Otherwise:
        median [Q1, Q3] + Kruskal-Wallis

    For a manuscript, also inspect histograms/Q-Q plots because normality tests
    can be overly sensitive in moderate/large samples.
    """
    rows = []

    for var in feature_cols:
        groups = [
            df.loc[df[target] == c, var].dropna().astype(float).values
            for c in CLASSES
        ]

        shapiro_ps = [_safe_shapiro(g) for g in groups]
        try:
            levene_p = float(levene(*groups, center="median").pvalue)
        except Exception:
            levene_p = np.nan

        is_normal = (
            all(np.isfinite(p) and p > 0.05 for p in shapiro_ps)
            and np.isfinite(levene_p)
            and levene_p > 0.05
        )

        if is_normal:
            desc = [
                f"{np.mean(g):.3f} ± {np.std(g, ddof=1):.3f}"
                for g in groups
            ]
            test_name = "One-way ANOVA"
            p_value = float(f_oneway(*groups).pvalue)
        else:
            desc = [
                f"{np.median(g):.3f} "
                f"[{np.quantile(g, 0.25):.3f}, {np.quantile(g, 0.75):.3f}]"
                for g in groups
            ]
            test_name = "Kruskal-Wallis"
            p_value = float(kruskal(*groups).pvalue)

        rows.append({
            "Variable": var,
            "Group 0": desc[0],
            "Group 1": desc[1],
            "Group 2": desc[2],
            "Test": test_name,
            "P_value": p_value,
        })

    out = pd.DataFrame(rows)
    return out


# =========================================================
# 3. Stratified train/test split and Table 2
# =========================================================

def standardized_mean_difference(x1, x2) -> float:
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    m1, m2 = np.mean(x1), np.mean(x2)
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    pooled = np.sqrt((s1**2 + s2**2) / 2)

    if pooled == 0:
        return 0.0

    return float((m1 - m2) / pooled)


def train_test_baseline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Table 2:
    - Descriptive mean ± SD
    - P value (t test if both Shapiro p>0.05, otherwise Mann-Whitney U)
    - Absolute SMD, with |SMD| < 0.1 commonly interpreted as good balance.
    """
    rows = []

    for var in X_train.columns:
        a = X_train[var].dropna().astype(float).values
        b = X_test[var].dropna().astype(float).values

        p1 = _safe_shapiro(a)
        p2 = _safe_shapiro(b)
        normal = np.isfinite(p1) and np.isfinite(p2) and p1 > 0.05 and p2 > 0.05

        if normal:
            p_value = float(ttest_ind(a, b, equal_var=False).pvalue)
            test_name = "Welch t-test"
        else:
            p_value = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
            test_name = "Mann-Whitney U"

        smd = standardized_mean_difference(a, b)

        rows.append({
            "Variable": var,
            "Train": f"{np.mean(a):.3f} ± {np.std(a, ddof=1):.3f}",
            "Test": f"{np.mean(b):.3f} ± {np.std(b, ddof=1):.3f}",
            "Test_name": test_name,
            "P_value": p_value,
            "SMD": smd,
            "Abs_SMD": abs(smd),
        })

    return pd.DataFrame(rows)


# =========================================================
# 4. Define 7 models and hyperparameter searches
# =========================================================

def build_searches():
    """
    Primary tuning target = macro one-vs-rest AUC.
    Secondary CV metrics are reported for model diagnostics.

    Class-imbalance strategy:
    - LR/SVC/DT/RF: class_weight='balanced'
    - XGBoost: balanced sample weights passed to fit()
    - LightGBM: class_weight='balanced'
    - MLP: RandomOverSampler inside CV pipeline, preventing leakage
    """

    cv = StratifiedKFold(
        n_splits=N_SPLITS_CV,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "macro_auc": "roc_auc_ovr",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "accuracy": "accuracy",
    }

    # 1. Softmax / multinomial logistic regression
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE
        ))
    ])

    lr_search = GridSearchCV(
        estimator=lr_pipe,
        param_grid={
            "model__C": np.logspace(-3, 3, 13)
        },
        scoring=scoring,
        refit="macro_auc",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )

    # 2. SVM / SVC
    svc_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    svc_search = RandomizedSearchCV(
        estimator=svc_pipe,
        param_distributions={
            "model__C": loguniform(1e-2, 1e2),
            "model__gamma": loguniform(1e-4, 1e0),
        },
        n_iter=N_ITER_RANDOM_SEARCH,
        scoring=scoring,
        refit="macro_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # 3. Decision Tree
    dt = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    dt_search = GridSearchCV(
        estimator=dt,
        param_grid={
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 3, 5, 10, 20],
            "ccp_alpha": [0.0, 0.001, 0.01],
        },
        scoring=scoring,
        refit="macro_auc",
        cv=cv,
        n_jobs=-1,
    )

    # 4. Random Forest
    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions={
            "n_estimators": randint(300, 1001),
            "max_depth": [None, 4, 6, 8, 10, 15],
            "min_samples_split": randint(2, 21),
            "min_samples_leaf": randint(1, 11),
            "max_features": ["sqrt", "log2", 0.5, 0.8],
        },
        n_iter=N_ITER_RANDOM_SEARCH,
        scoring=scoring,
        refit="macro_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # 5. XGBoost
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    xgb_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions={
            "n_estimators": randint(200, 1001),
            "max_depth": randint(2, 9),
            "learning_rate": loguniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "min_child_weight": randint(1, 11),
            "reg_alpha": loguniform(1e-4, 10),
            "reg_lambda": loguniform(1e-3, 30),
        },
        n_iter=N_ITER_RANDOM_SEARCH,
        scoring=scoring,
        refit="macro_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # 6. LightGBM
    lgbm = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )

    lgbm_search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions={
            "n_estimators": randint(200, 1001),
            "learning_rate": loguniform(0.01, 0.3),
            "num_leaves": randint(7, 64),
            "max_depth": [-1, 3, 5, 7, 10],
            "min_child_samples": randint(5, 51),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_alpha": loguniform(1e-4, 10),
            "reg_lambda": loguniform(1e-3, 30),
        },
        n_iter=N_ITER_RANDOM_SEARCH,
        scoring=scoring,
        refit="macro_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # 7. MLP
    # RandomOverSampler is inside the CV pipeline, so the test fold is never oversampled.
    mlp_pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("ros", RandomOverSampler(random_state=RANDOM_STATE)),
        ("model", MLPClassifier(
            max_iter=3000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=RANDOM_STATE
        ))
    ])

    mlp_search = RandomizedSearchCV(
        estimator=mlp_pipe,
        param_distributions={
            "model__hidden_layer_sizes": [
                (16,), (32,), (64,),
                (32, 16), (64, 32), (64, 32, 16)
            ],
            "model__activation": ["relu", "tanh"],
            "model__alpha": loguniform(1e-5, 1e-1),
            "model__learning_rate_init": loguniform(1e-4, 1e-2),
            "model__batch_size": [16, 32, 64],
        },
        n_iter=N_ITER_RANDOM_SEARCH,
        scoring=scoring,
        refit="macro_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return {
        "Softmax_LR": lr_search,
        "SVM": svc_search,
        "Decision_Tree": dt_search,
        "Random_Forest": rf_search,
        "XGBoost": xgb_search,
        "LightGBM": lgbm_search,
        "MLP": mlp_search,
    }


def tune_models(
    searches: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[Dict, pd.DataFrame]:

    fitted_models = {}
    cv_rows = []

    xgb_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    for name, search in searches.items():
        print("\n" + "=" * 70)
        print(f"TUNING: {name}")
        print("=" * 70)

        if name == "XGBoost":
            search.fit(X_train, y_train, sample_weight=xgb_weights)
        else:
            search.fit(X_train, y_train)

        fitted_models[name] = search.best_estimator_

        idx = search.best_index_
        cv_rows.append({
            "Model": name,
            "Best_CV_Macro_AUC": search.cv_results_["mean_test_macro_auc"][idx],
            "CV_Macro_F1": search.cv_results_["mean_test_macro_f1"][idx],
            "CV_Weighted_F1": search.cv_results_["mean_test_weighted_f1"][idx],
            "CV_Accuracy": search.cv_results_["mean_test_accuracy"][idx],
            "Best_Params": str(search.best_params_),
        })

        print("Best parameters:", search.best_params_)
        print(
            "Best CV macro-AUC:",
            round(search.cv_results_["mean_test_macro_auc"][idx], 4)
        )

    return fitted_models, pd.DataFrame(cv_rows)


# =========================================================
# 5. Core performance metrics
# =========================================================

def multiclass_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    total = cm.sum()

    sensitivities = []
    specificities = []

    for i in range(len(CLASSES)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - TP - FN - FP

        sens = TP / (TP + FN) if (TP + FN) else np.nan
        spec = TN / (TN + FP) if (TN + FP) else np.nan

        sensitivities.append(sens)
        specificities.append(spec)

    return np.array(sensitivities), np.array(specificities)


def class1_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_class1: np.ndarray
) -> Dict[str, float]:
    y_bin = (np.asarray(y_true) == 1).astype(int)
    pred_bin = (np.asarray(y_pred) == 1).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_bin, pred_bin, labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    return {
        "Class1_AUC": roc_auc_score(y_bin, y_prob_class1),
        "Class1_Sensitivity": sensitivity,
        "Class1_Specificity": specificity,
        "Class1_F1": f1_score(y_bin, pred_bin, zero_division=0),
        "Class1_PPV": ppv,
        "Class1_NPV": npv,
    }


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:

    sens, spec = multiclass_sensitivity_specificity(y_true, y_pred)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "Macro_AUC": roc_auc_score(
            y_true,
            y_prob,
            labels=CLASSES,
            multi_class="ovr",
            average="macro"
        ),
        "Macro_F1": f1_score(
            y_true, y_pred, labels=CLASSES,
            average="macro", zero_division=0
        ),
        "Weighted_F1": f1_score(
            y_true, y_pred, labels=CLASSES,
            average="weighted", zero_division=0
        ),
        "Macro_Sensitivity": np.nanmean(sens),
        "Macro_Specificity": np.nanmean(spec),
    }

    # Per-class sensitivity and specificity
    for i, c in enumerate(CLASSES):
        metrics[f"Class{c}_Sensitivity"] = sens[i]
        metrics[f"Class{c}_Specificity"] = spec[i]

    metrics.update(
        class1_binary_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob_class1=y_prob[:, 1]
        )
    )

    return metrics


def evaluate_test_set(
    models: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series
):
    predictions = {}
    point_rows = []

    for name, model in models.items():
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)

        predictions[name] = {
            "pred": np.asarray(pred),
            "prob": np.asarray(prob),
        }

        m = calculate_metrics(
            y_true=np.asarray(y_test),
            y_pred=np.asarray(pred),
            y_prob=np.asarray(prob)
        )
        m["Model"] = name
        point_rows.append(m)

    point_df = pd.DataFrame(point_rows)
    return predictions, point_df


# =========================================================
# 6. Stratified paired bootstrap: 1000 resamples
# =========================================================

def make_stratified_bootstrap_indices(
    y: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE
):
    """
    Same bootstrap indices are used for all models:
    this supports paired model comparison.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    class_indices = {
        c: np.where(y == c)[0]
        for c in CLASSES
    }

    all_boot_indices = []

    for _ in range(n_bootstrap):
        sampled = []
        for c in CLASSES:
            idx = class_indices[c]
            boot_c = rng.choice(idx, size=len(idx), replace=True)
            sampled.append(boot_c)

        boot_idx = np.concatenate(sampled)
        rng.shuffle(boot_idx)
        all_boot_indices.append(boot_idx)

    return all_boot_indices


def bootstrap_all_models(
    y_test: pd.Series,
    predictions: Dict,
    n_bootstrap: int = N_BOOTSTRAP
):
    y = np.asarray(y_test)
    boot_indices = make_stratified_bootstrap_indices(
        y, n_bootstrap=n_bootstrap
    )

    rows = []

    for b, idx in enumerate(boot_indices):
        y_b = y[idx]

        for name, out in predictions.items():
            pred_b = out["pred"][idx]
            prob_b = out["prob"][idx]

            m = calculate_metrics(y_b, pred_b, prob_b)

            for metric, value in m.items():
                rows.append({
                    "Bootstrap": b + 1,
                    "Model": name,
                    "Metric": metric,
                    "Value": value,
                })

    boot_df = pd.DataFrame(rows)
    return boot_df


def summarize_bootstrap(
    boot_df: pd.DataFrame,
    point_df: pd.DataFrame
) -> pd.DataFrame:

    long_point = point_df.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Point_Estimate"
    )

    # Keep numeric metrics only
    long_point["Point_Estimate"] = pd.to_numeric(
        long_point["Point_Estimate"],
        errors="coerce"
    )
    long_point = long_point.dropna(subset=["Point_Estimate"])

    summary = (
        boot_df
        .groupby(["Model", "Metric"])["Value"]
        .agg(
            Bootstrap_Mean="mean",
            Bootstrap_SD="std",
            Bootstrap_Median="median",
            CI_2_5=lambda x: np.nanquantile(x, 0.025),
            CI_97_5=lambda x: np.nanquantile(x, 0.975),
        )
        .reset_index()
    )

    summary = summary.merge(
        long_point,
        on=["Model", "Metric"],
        how="left"
    )

    summary["Bootstrap_Bias"] = (
        summary["Bootstrap_Mean"] - summary["Point_Estimate"]
    )

    return summary


def make_formatted_test_table(
    boot_summary: pd.DataFrame,
    selected_metrics=None
) -> pd.DataFrame:
    if selected_metrics is None:
        selected_metrics = [
            "Accuracy",
            "Kappa",
            "Macro_AUC",
            "Macro_F1",
            "Weighted_F1",
            "Macro_Sensitivity",
            "Macro_Specificity",
            "Class1_AUC",
            "Class1_Sensitivity",
            "Class1_Specificity",
            "Class1_F1",
            "Class1_PPV",
            "Class1_NPV",
        ]

    tmp = boot_summary[
        boot_summary["Metric"].isin(selected_metrics)
    ].copy()

    tmp["Estimate_95CI"] = tmp.apply(
        lambda r:
            f"{r['Point_Estimate']:.3f} "
            f"({r['CI_2_5']:.3f}–{r['CI_97_5']:.3f})",
        axis=1
    )

    table = tmp.pivot(
        index="Model",
        columns="Metric",
        values="Estimate_95CI"
    )

    # Preserve desired metric order
    table = table.reindex(columns=selected_metrics)
    return table.reset_index()


def paired_bootstrap_model_differences(
    boot_df: pd.DataFrame,
    reference_model: str,
    metrics=("Macro_AUC", "Macro_F1", "Class1_AUC", "Class1_Sensitivity", "Class1_F1")
) -> pd.DataFrame:
    """
    Optional table:
    paired bootstrap difference vs a prespecified reference model.
    95% CI excluding 0 suggests a stable difference in that metric.

    This is preferable to comparing overlapping CIs.
    """
    rows = []

    for metric in metrics:
        d = boot_df[boot_df["Metric"] == metric].pivot(
            index="Bootstrap",
            columns="Model",
            values="Value"
        )

        if reference_model not in d.columns:
            continue

        for model in d.columns:
            if model == reference_model:
                continue

            diff = d[model] - d[reference_model]

            rows.append({
                "Metric": metric,
                "Model": model,
                "Reference": reference_model,
                "Mean_Difference": diff.mean(),
                "CI_2_5": np.quantile(diff, 0.025),
                "CI_97_5": np.quantile(diff, 0.975),
                "Bootstrap_P_approx": 2 * min(
                    np.mean(diff <= 0),
                    np.mean(diff >= 0)
                ),
            })

    return pd.DataFrame(rows)


# =========================================================
# 7. ROC curves
# =========================================================

def plot_class1_roc(
    models: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path
):
    y_bin = (np.asarray(y_test) == 1).astype(int)

    plt.figure(figsize=(8, 7))

    for name, model in models.items():
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_bin, prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Class 1 vs Rest ROC curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_multiclass_roc_per_model(
    model_name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path
):
    y_true = np.asarray(y_test)
    y_bin = label_binarize(y_true, classes=CLASSES)
    prob = model.predict_proba(X_test)

    plt.figure(figsize=(8, 7))

    fpr_dict = {}
    tpr_dict = {}
    auc_dict = {}

    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], prob[:, i])
        fpr_dict[c] = fpr
        tpr_dict[c] = tpr
        auc_dict[c] = auc(fpr, tpr)

        plt.plot(
            fpr, tpr,
            label=f"Class {c} (AUC={auc_dict[c]:.3f})"
        )

    # Macro-average ROC
    all_fpr = np.unique(
        np.concatenate([fpr_dict[c] for c in CLASSES])
    )
    mean_tpr = np.zeros_like(all_fpr)

    for c in CLASSES:
        mean_tpr += np.interp(
            all_fpr,
            fpr_dict[c],
            tpr_dict[c]
        )

    mean_tpr /= len(CLASSES)
    macro_auc = auc(all_fpr, mean_tpr)

    plt.plot(
        all_fpr,
        mean_tpr,
        linestyle=":",
        linewidth=2,
        label=f"Macro-average (AUC={macro_auc:.3f})"
    )

    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"One-vs-Rest ROC: {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 8. Calibration curve + calibration intercept and slope
# =========================================================

def calibration_intercept_slope(
    y_binary: np.ndarray,
    prob: np.ndarray
) -> Tuple[float, float]:
    """
    Logistic recalibration:
        logit(P(Y=1)) = intercept + slope * logit(predicted probability)

    Ideal:
        intercept = 0
        slope = 1
    """
    y_binary = np.asarray(y_binary).astype(int)
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))

    X_cal = sm.add_constant(lp)

    try:
        fit = sm.GLM(
            y_binary,
            X_cal,
            family=sm.families.Binomial()
        ).fit()

        intercept = float(fit.params[0])
        slope = float(fit.params[1])
    except Exception:
        intercept = np.nan
        slope = np.nan

    return intercept, slope


def calibration_table(
    models: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    rows = []

    for name, model in models.items():
        prob = model.predict_proba(X_test)

        for i, c in enumerate(CLASSES):
            y_bin = (np.asarray(y_test) == c).astype(int)

            intercept, slope = calibration_intercept_slope(
                y_bin,
                prob[:, i]
            )

            brier = np.mean(
                (prob[:, i] - y_bin) ** 2
            )

            rows.append({
                "Model": name,
                "Class": int(c),
                "Calibration_Intercept": intercept,
                "Calibration_Slope": slope,
                "Brier_Score": brier,
            })

    return pd.DataFrame(rows)


def plot_class1_calibration(
    models: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path
):
    y_bin = (np.asarray(y_test) == 1).astype(int)

    plt.figure(figsize=(8, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal")

    for name, model in models.items():
        prob = model.predict_proba(X_test)[:, 1]

        frac_pos, mean_pred = calibration_curve(
            y_bin,
            prob,
            n_bins=10,
            strategy="quantile"
        )

        plt.plot(
            mean_pred,
            frac_pos,
            marker="o",
            label=name
        )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("Calibration curves: Class 1 vs Rest")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 9. Decision Curve Analysis for Class 1 vs Rest
# =========================================================

def net_benefit_binary(
    y_binary: np.ndarray,
    prob: np.ndarray,
    thresholds: np.ndarray
) -> np.ndarray:
    y_binary = np.asarray(y_binary).astype(int)
    prob = np.asarray(prob)
    n = len(y_binary)

    out = []

    for pt in thresholds:
        pred = (prob >= pt).astype(int)

        tp = np.sum((pred == 1) & (y_binary == 1))
        fp = np.sum((pred == 1) & (y_binary == 0))

        nb = (
            tp / n
            - fp / n * (pt / (1 - pt))
        )
        out.append(nb)

    return np.asarray(out)


def plot_class1_dca(
    models: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path
):
    y_bin = (np.asarray(y_test) == 1).astype(int)

    # Avoid thresholds extremely close to 1.
    thresholds = np.linspace(0.01, 0.80, 80)

    plt.figure(figsize=(9, 7))

    for name, model in models.items():
        prob = model.predict_proba(X_test)[:, 1]
        nb = net_benefit_binary(
            y_bin,
            prob,
            thresholds
        )
        plt.plot(
            thresholds,
            nb,
            label=name
        )

    prevalence = y_bin.mean()

    treat_all = (
        prevalence
        - (1 - prevalence)
        * thresholds / (1 - thresholds)
    )

    treat_none = np.zeros_like(thresholds)

    plt.plot(
        thresholds,
        treat_all,
        linestyle="--",
        label="Treat all"
    )
    plt.plot(
        thresholds,
        treat_none,
        linestyle=":",
        label="Treat none"
    )

    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision Curve Analysis: Class 1 vs Rest")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 10. Confusion matrix
# =========================================================

def plot_confusion_matrix_with_row_percent(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASSES
    )

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["0", "1", "2"])
    ax.set_yticklabels(["0", "1", "2"])

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix: {model_name}")

    for i in range(3):
        for j in range(3):
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_pct[i, j] * 100:.1f}%)",
                ha="center",
                va="center"
            )

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 11. SHAP for the final selected model, focusing on Class 1
# =========================================================

def shap_class1_summary_and_bar(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    background_n: int = 100,
    explain_n: int = 150,
):
    """
    Model-agnostic SHAP probability explanation.
    This works across pipelines and most classifiers with predict_proba.

    For publication:
    - Select the final model based on prespecified performance/calibration/DCA.
    - Explain the final model.
    - Focus on SHAP output for Class 1.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    bg_n = min(background_n, len(X_train))
    ex_n = min(explain_n, len(X_test))

    bg_idx = rng.choice(
        np.arange(len(X_train)),
        size=bg_n,
        replace=False
    )
    ex_idx = rng.choice(
        np.arange(len(X_test)),
        size=ex_n,
        replace=False
    )

    background = X_train.iloc[bg_idx].copy()
    X_explain = X_test.iloc[ex_idx].copy()

    masker = shap.maskers.Independent(background)

    explainer = shap.Explainer(
        model.predict_proba,
        masker=masker,
        algorithm="permutation",
        output_names=[
            "Class 0",
            "Class 1",
            "Class 2"
        ]
    )

    # For permutation SHAP, at least 2 * n_features + 1 evaluations are needed.
    max_evals = max(
        2 * X_explain.shape[1] + 1,
        31
    )

    shap_values = explainer(
        X_explain,
        max_evals=max_evals
    )

    values = shap_values.values

    if values.ndim != 3:
        raise ValueError(
            "Expected multiclass SHAP values with shape "
            "(n_samples, n_features, n_classes)."
        )

    class1_values = values[:, :, 1]

    # Beeswarm summary plot
    plt.figure()
    shap.summary_plot(
        class1_values,
        X_explain,
        show=False,
        max_display=15
    )
    plt.title(
        f"SHAP summary: {model_name}, Class 1"
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / f"SHAP_summary_{model_name}_Class1.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Mean absolute SHAP bar plot
    plt.figure()
    shap.summary_plot(
        class1_values,
        X_explain,
        plot_type="bar",
        show=False,
        max_display=15
    )
    plt.title(
        f"Mean |SHAP|: {model_name}, Class 1"
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / f"SHAP_bar_{model_name}_Class1.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Numerical SHAP importance table
    importance = pd.DataFrame({
        "Feature": X_explain.columns,
        "Mean_abs_SHAP_Class1": np.mean(
            np.abs(class1_values),
            axis=0
        )
    }).sort_values(
        "Mean_abs_SHAP_Class1",
        ascending=False
    )

    return importance


# =========================================================
# 12. Optional sensitivity analysis:
#     remove derived discrimination indices
# =========================================================

def get_raw_feature_sensitivity_set(
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Optional robustness analysis.
    shine, Mentzer and green appear to be derived discrimination indices.
    If these are mathematically derived from CBC variables, rerun models without
    them to assess whether conclusions depend on derived features.
    """
    derived = [
        c for c in ["shine", "Mentzer", "green"]
        if c in X.columns
    ]
    return X.drop(columns=derived)


# =========================================================
# 13. Main workflow
# =========================================================

def main():
    # ----------------------------
    # A. Load data
    # ----------------------------
    df, X, y = load_data(DATA_PATH)
    feature_cols = X.columns.tolist()

    # ----------------------------
    # B. Table 1: 3-class baseline
    # ----------------------------
    table1 = baseline_three_groups(
        df=df,
        feature_cols=feature_cols
    )
    table1.to_csv(
        OUTPUT_DIR / "Table1_three_group_baseline.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ----------------------------
    # C. Stratified split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Save row indices to guarantee reproducibility
    split_info = pd.DataFrame({
        "row_index": np.concatenate(
            [X_train.index.values, X_test.index.values]
        ),
        "set": (
            ["Train"] * len(X_train)
            + ["Test"] * len(X_test)
        )
    })
    split_info.to_csv(
        OUTPUT_DIR / "train_test_split_indices.csv",
        index=False
    )

    print("\nTrain distribution:")
    print(y_train.value_counts().sort_index())
    print("\nTest distribution:")
    print(y_test.value_counts().sort_index())

    # ----------------------------
    # D. Table 2: train/test baseline
    # ----------------------------
    table2 = train_test_baseline(
        X_train,
        X_test
    )
    table2.to_csv(
        OUTPUT_DIR / "Table2_train_test_baseline.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ----------------------------
    # E. Hyperparameter tuning
    # ----------------------------
    searches = build_searches()

    fitted_models, cv_table = tune_models(
        searches,
        X_train,
        y_train
    )

    cv_table.to_csv(
        OUTPUT_DIR / "CV_tuning_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ----------------------------
    # F. Independent test evaluation
    # ----------------------------
    predictions, point_df = evaluate_test_set(
        fitted_models,
        X_test,
        y_test
    )

    point_df.to_csv(
        OUTPUT_DIR / "Test_point_estimates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ----------------------------
    # G. Bootstrap 1000
    # ----------------------------
    boot_df = bootstrap_all_models(
        y_test=y_test,
        predictions=predictions,
        n_bootstrap=N_BOOTSTRAP
    )

    boot_df.to_csv(
        OUTPUT_DIR / "Bootstrap_1000_raw_long.csv",
        index=False,
        encoding="utf-8-sig"
    )

    boot_summary = summarize_bootstrap(
        boot_df,
        point_df
    )

    boot_summary.to_csv(
        OUTPUT_DIR / "Table4_bootstrap_1000_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Table 3: test metrics formatted as estimate (95% CI)
    table3 = make_formatted_test_table(
        boot_summary
    )

    table3.to_csv(
        OUTPUT_DIR / "Table3_test_7model_comparison_95CI.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ----------------------------
    # H. Optional paired bootstrap differences
    # Reference can be changed after prespecification.
    # ----------------------------
    paired_diff = paired_bootstrap_model_differences(
        boot_df,
        reference_model="XGBoost"
    )

    paired_diff.to_csv(
        OUTPUT_DIR / "Optional_paired_bootstrap_vs_XGBoost.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ----------------------------
    # I. Calibration
    # ----------------------------
    cal_table = calibration_table(
        fitted_models,
        X_test,
        y_test
    )

    cal_table.to_csv(
        OUTPUT_DIR / "Calibration_intercept_slope_brier.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ----------------------------
    # J. ROC figures
    # ----------------------------
    plot_class1_roc(
        fitted_models,
        X_test,
        y_test,
        FIG_DIR / "ROC_Class1_vs_Rest_7models.png"
    )

    for name, model in fitted_models.items():
        plot_multiclass_roc_per_model(
            model_name=name,
            model=model,
            X_test=X_test,
            y_test=y_test,
            save_path=FIG_DIR / f"ROC_multiclass_{name}.png"
        )

    # ----------------------------
    # K. Calibration figure
    # ----------------------------
    plot_class1_calibration(
        fitted_models,
        X_test,
        y_test,
        FIG_DIR / "Calibration_Class1_vs_Rest_7models.png"
    )

    # ----------------------------
    # L. DCA figure
    # ----------------------------
    plot_class1_dca(
        fitted_models,
        X_test,
        y_test,
        FIG_DIR / "DCA_Class1_vs_Rest_7models.png"
    )

    # ----------------------------
    # M. Confusion matrices
    # ----------------------------
    for name, out in predictions.items():
        plot_confusion_matrix_with_row_percent(
            model_name=name,
            y_true=np.asarray(y_test),
            y_pred=out["pred"],
            save_path=FIG_DIR / f"Confusion_matrix_{name}.png"
        )

    # ----------------------------
    # N. Select final model for SHAP
    # Here: highest original-test Class1_AUC as a placeholder selection rule.
    # For a manuscript, prespecify the final selection hierarchy.
    # ----------------------------
    shap_model_name = (
        point_df
        .sort_values(
            ["Class1_AUC", "Macro_AUC", "Class1_F1"],
            ascending=False
        )
        .iloc[0]["Model"]
    )

    print(
        f"\nSHAP model selected by placeholder rule: "
        f"{shap_model_name}"
    )

    shap_importance = shap_class1_summary_and_bar(
        model=fitted_models[shap_model_name],
        X_train=X_train,
        X_test=X_test,
        model_name=shap_model_name,
        output_dir=FIG_DIR,
        background_n=100,
        explain_n=150
    )

    shap_importance.to_csv(
        OUTPUT_DIR / f"SHAP_importance_{shap_model_name}_Class1.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ----------------------------
    # O. Export main tables into one Excel workbook
    # ----------------------------
    with pd.ExcelWriter(
        OUTPUT_DIR / "thalassemia_7model_main_tables.xlsx",
        engine="openpyxl"
    ) as writer:
        table1.to_excel(
            writer,
            sheet_name="Table1_3Groups",
            index=False
        )
        table2.to_excel(
            writer,
            sheet_name="Table2_TrainVsTest",
            index=False
        )
        cv_table.to_excel(
            writer,
            sheet_name="CV_Tuning",
            index=False
        )
        point_df.to_excel(
            writer,
            sheet_name="Test_Point",
            index=False
        )
        table3.to_excel(
            writer,
            sheet_name="Table3_Test_95CI",
            index=False
        )
        boot_summary.to_excel(
            writer,
            sheet_name="Table4_Bootstrap",
            index=False
        )
        cal_table.to_excel(
            writer,
            sheet_name="Calibration",
            index=False
        )
        paired_diff.to_excel(
            writer,
            sheet_name="Paired_Bootstrap",
            index=False
        )
        shap_importance.to_excel(
            writer,
            sheet_name="SHAP_Class1",
            index=False
        )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results directory: {OUTPUT_DIR.resolve()}")
    print(f"Figures directory: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()

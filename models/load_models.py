from pathlib import Path

import joblib

from config import settings


def load_trained_model(model_name, model_dir=None):
    """
    加载单个已训练并保存的模型 Pipeline（joblib 格式）。

    Parameters
    ----------
    model_name : str
        模型名称，须与训练时 ModelTrainer.save() 使用的名称一致
        （即 settings.MODEL_ORDER 中的名字，如 "XGBoost"、"RandomForest"）。
    model_dir : str or Path, optional
        模型所在目录，默认使用 settings.MODEL_DIR。

    Returns
    -------
    sklearn.pipeline.Pipeline
        训练好的完整 Pipeline（预处理 + 分类器），可直接 .predict() / .predict_proba()。

    Raises
    ------
    FileNotFoundError
        如果对应的 .joblib 文件不存在（说明该模型还没训练/保存过）。
    """
    model_dir = Path(model_dir) if model_dir is not None else settings.MODEL_DIR
    path = model_dir / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def load_all_trained_models(model_names=None, model_dir=None):
    """
    批量加载已训练并保存的模型。

    Parameters
    ----------
    model_names : list[str], optional
        默认加载 settings.MODEL_ORDER 中的全部 7 个模型；
        找不到对应 .joblib 文件的模型会被跳过并打印警告（不会中断整体加载）。
    model_dir : str or Path, optional
        默认使用 settings.MODEL_DIR。

    Returns
    -------
    dict[str, sklearn.pipeline.Pipeline]
        模型名 -> 已加载的 Pipeline。
    """
    model_names = model_names or settings.MODEL_ORDER
    models = {}
    for name in model_names:
        try:
            models[name] = load_trained_model(name, model_dir)
            print(f"[INFO] Loaded model: {name}")
        except FileNotFoundError:
            print(f"[WARN] Skipped {name}: joblib file not found")

    # 加载清理后的数据集
    [X_train, X_test, y_train, y_test, FEATURE_COLS] = joblib.load(settings.MODEL_DIR / 'dataset.joblib')
    print(f"[INFO] Loaded dataset: X_train, X_test, y_train, y_test, FEATURE_COLS")
    return [models, X_train, X_test, y_train, y_test, FEATURE_COLS]


if __name__ == '__main__':
    # 一次性加载全部已保存模型（不需要重新训练）
    loaded_models = load_all_trained_models()

    # 只加载单个模型
    xgb_pipeline = load_trained_model("XGBoost")

    # 对新数据 / 测试集做预测（需要先准备好与训练时列名一致的 DataFrame，如 X_test）
    # y_pred = xgb_pipeline.predict(X_test)
    # y_proba = xgb_pipeline.predict_proba(X_test)

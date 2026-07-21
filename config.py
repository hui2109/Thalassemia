# -*- coding: utf-8 -*-
"""
config.py
=========
全局配置模块。集中管理项目中用到的路径、随机种子、类别定义、特征列表等常量。
所有其他模块都应从这里导入配置，而不是在各处硬编码，方便客户后续统一修改。

使用方式：
    from config import Config
    cfg = Config()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class Config:
    """项目全局配置类。

    Attributes
    ----------
    project_root : Path
        项目根目录。
    raw_data_path : Path
        原始数据 csv 文件路径。
    output_dir : Path
        所有产出物（图表/表格/模型）的根输出目录。
    figure_dir / table_dir / model_dir : Path
        输出子目录。
    id_col : str
        样本编号列名，仅用于溯源，不参与建模。
    target_col : str
        金标准（结局）列名。
    class_labels : Dict[int, str]
        类别编码 -> 中文标签的映射。 0=对照组, 1=静止型地贫, 2=非静止型地贫。
    random_state : int
        全局随机种子，保证结果可复现。
    test_size : float
        测试集比例（7:3 划分中的 0.3）。
    cv_folds : int
        训练集内部交叉验证折数。
    n_bootstrap : int
        Bootstrap 重采样次数（用于指标 95% CI 及测试集模型对比）。
    ci_alpha : float
        置信区间显著性水平，0.05 对应 95% CI。
    n_jobs : int
        并行计算使用的核心数，-1 表示使用全部可用核心。
    """

    project_root: Path = Path(__file__).resolve().parent
    raw_data_path: Path = project_root / "data" / "thalassemia_raw.csv"
    output_dir: Path = project_root / "outputs"

    id_col: str = "no"
    target_col: str = "diagnosis"

    class_labels: Dict[int, str] = field(
        default_factory=lambda: {0: "对照组", 1: "静止型地贫", 2: "非静止型地贫"}
    )

    random_state: int = 42
    test_size: float = 0.3
    cv_folds: int = 5
    n_bootstrap: int = 1000
    ci_alpha: float = 0.05
    n_jobs: int = -1

    def __post_init__(self):
        self.figure_dir: Path = self.output_dir / "figures"
        self.table_dir: Path = self.output_dir / "tables"
        self.model_dir: Path = self.output_dir / "models"
        for d in (self.figure_dir, self.table_dir, self.model_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def class_order(self) -> List[int]:
        """返回按编码升序排列的类别列表，如 [0, 1, 2]。"""
        return sorted(self.class_labels.keys())

    def feature_cols(self, df_columns: List[str]) -> List[str]:
        """根据数据框的全部列名，推导出用于建模的特征列（排除编号列与结局列）。

        Parameters
        ----------
        df_columns : List[str]
            数据框的全部列名。

        Returns
        -------
        List[str]
            特征列名列表，保持原始顺序。
        """
        exclude = {self.id_col, self.target_col}
        return [c for c in df_columns if c not in exclude]

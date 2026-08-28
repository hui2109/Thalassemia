from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_thalassemia_gene_screening(
        input_xlsx: str | Path,
        output_csv: str | Path,
        sheet_name: str | int = 0,
) -> pd.DataFrame:
    """
    汇总 diagnosis=1 和 diagnosis=2 样本的地中海贫血基因筛查结果。

    功能：
        1. 读取 Excel 文件；
        2. 筛选 diagnosis 列值为 1 或 2 的样本；
        3. 按“地中海贫血基因筛查”中的具体型别进行统计；
        4. 分别统计 diagnosis=1 和 diagnosis=2 的数量；
        5. 计算每个型别的总数量；
        6. 将汇总结果保存为 CSV 文件。

    参数：
        input_xlsx : str | pathlib.Path
            输入 Excel 文件路径。

        output_csv : str | pathlib.Path
            输出 CSV 文件路径。

        sheet_name : str | int, optional
            Excel 工作表名称或索引。
            默认值为 0，即读取第一个工作表。

    返回：
        pandas.DataFrame
            汇总结果，包含以下列：
            - 地中海贫血基因筛查型别
            - diagnosis_1_数量
            - diagnosis_2_数量
            - 合计

    Raises:
        FileNotFoundError:
            输入 Excel 文件不存在。

        KeyError:
            Excel 中缺少 diagnosis 或“地中海贫血基因筛查”列。
    """

    # 转换为 Path 对象
    input_xlsx = Path(input_xlsx)
    output_csv = Path(output_csv)

    # ------------------------------
    # 1. 检查输入文件
    # ------------------------------
    if not input_xlsx.exists():
        raise FileNotFoundError(
            f"找不到输入文件：\n{input_xlsx}"
        )

    # ------------------------------
    # 2. 读取 Excel
    # ------------------------------
    df = pd.read_excel(
        input_xlsx,
        sheet_name=sheet_name,
        engine="openpyxl",
    )

    # 清理列名两侧空格
    df.columns = df.columns.astype(str).str.strip()

    diagnosis_col = "diagnosis"
    gene_col = "地中海贫血基因筛查"

    # ------------------------------
    # 3. 检查必要列
    # ------------------------------
    required_columns = [
        diagnosis_col,
        gene_col,
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Excel 中缺少以下必要列："
            + ", ".join(missing_columns)
            + "\n\n当前 Excel 列名为：\n"
            + "\n".join(df.columns)
        )

    # ------------------------------
    # 4. 清洗 diagnosis
    # ------------------------------
    # 非数字值自动转换为 NaN
    df[diagnosis_col] = pd.to_numeric(
        df[diagnosis_col],
        errors="coerce",
    )

    # 只保留 diagnosis=1 或 diagnosis=2
    filtered_df = df.loc[
        df[diagnosis_col].isin([1, 2]),
        [diagnosis_col, gene_col],
    ].copy()

    # 转为整数，避免结果出现 1.0、2.0
    filtered_df[diagnosis_col] = (
        filtered_df[diagnosis_col]
        .astype(int)
    )

    # ------------------------------
    # 5. 清洗基因筛查结果
    # ------------------------------
    # 缺失值统一标记
    filtered_df[gene_col] = (
        filtered_df[gene_col]
        .fillna("缺失/空白")
        .astype(str)
        .str.strip()
    )

    # 空字符串也标记为缺失
    filtered_df.loc[
        filtered_df[gene_col].eq(""),
        gene_col,
    ] = "缺失/空白"

    # ------------------------------
    # 6. 按基因型别和 diagnosis 统计
    # ------------------------------
    summary = pd.crosstab(
        index=filtered_df[gene_col],
        columns=filtered_df[diagnosis_col],
    )

    # 确保即使某一组不存在，也有对应列
    summary = summary.reindex(
        columns=[1, 2],
        fill_value=0,
    )

    # 重命名
    summary = summary.rename(
        columns={
            1: "diagnosis_1_数量",
            2: "diagnosis_2_数量",
        }
    )

    # index 转普通列
    summary = summary.reset_index()

    # 修改第一列名称
    summary = summary.rename(
        columns={
            gene_col: "地中海贫血基因筛查型别"
        }
    )

    # ------------------------------
    # 7. 计算合计
    # ------------------------------
    summary["合计"] = (
            summary["diagnosis_1_数量"]
            + summary["diagnosis_2_数量"]
    )

    # ------------------------------
    # 8. 按总数量降序排列
    # ------------------------------
    summary = summary.sort_values(
        by=[
            "合计",
            "diagnosis_2_数量",
            "diagnosis_1_数量",
            "地中海贫血基因筛查型别",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    # ------------------------------
    # 9. 创建输出目录
    # ------------------------------
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------
    # 10. 保存 CSV
    # ------------------------------
    # utf-8-sig 可以避免 Excel 打开中文 CSV 时乱码
    summary.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    return summary


def main() -> None:
    """
    主程序入口。

    根据当前项目目录自动确定输入、输出文件路径，
    调用 summarize_thalassemia_gene_screening() 完成统计。
    """

    # 当前文件：
    # Thalassemia/stats_analysis/gene_statistics.py
    current_file = Path(__file__).resolve()

    # 项目根目录：
    # Thalassemia/
    project_root = current_file.parent.parent

    # ==================================================
    # 输入文件
    # ==================================================
    #
    # 默认认为 Excel 放在：
    #
    # Thalassemia/data/07_模型原始数据.xlsx
    #
    input_path = (
            project_root / 'data'
            / "07_模型原始数据.xlsx"
    )

    # ==================================================
    # 输出文件
    # ==================================================
    #
    # 输出到：
    #
    # Thalassemia/outputs/tables/table6_gene_summary.csv
    #
    output_path = (
            project_root
            / "outputs"
            / "tables"
            / "table6_gene_summary.csv"
    )

    # 执行统计
    summary = summarize_thalassemia_gene_screening(
        input_xlsx=input_path,
        output_csv=output_path,
        sheet_name=0,
    )

    # ==================================================
    # 结果校验
    # ==================================================
    diagnosis_1_total = int(
        summary["diagnosis_1_数量"].sum()
    )

    diagnosis_2_total = int(
        summary["diagnosis_2_数量"].sum()
    )

    total = int(
        summary["合计"].sum()
    )

    type_count = len(summary)

    print("=" * 70)
    print("地中海贫血基因筛查结果汇总完成")
    print("=" * 70)

    print(f"\n输入文件：")
    print(input_path)

    print(f"\n输出文件：")
    print(output_path)

    print("\n统计结果：")
    print(
        f"diagnosis=1 样本数："
        f"{diagnosis_1_total}"
    )
    print(
        f"diagnosis=2 样本数："
        f"{diagnosis_2_total}"
    )
    print(
        f"总样本数："
        f"{total}"
    )
    print(
        f"基因筛查型别数："
        f"{type_count}"
    )

    print("\n汇总表：")
    print(
        summary.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("CSV 已保存。")
    print("=" * 70)


if __name__ == "__main__":
    main()

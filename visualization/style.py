# -*- coding: utf-8 -*-
"""
style.py
统一设置 matplotlib / seaborn 的全局绘图风格，保证所有图表字体、配色一致，
并解决中文标签乱码、负号显示异常的问题。

在任何绘图代码执行之前，只需在 main.py 中调用一次 apply_style() 即可全局生效。
"""

import matplotlib.pyplot as plt
import seaborn as sns


def apply_style():
    """
    应用统一绘图风格。图表内容已全部使用英文（符合 SCI 期刊投稿要求），
    因此字体统一使用 Arial（SCI 期刊图表最常用字体），不再依赖中文字体。
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 11
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.0)

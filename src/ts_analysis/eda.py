"""描述性统计与分布特征 (对应报告第 3 节).

包括：各价格/成交量变量的分位数描述统计，以及按十年分组的价格水平、
收益率均值、波动率与峰度对照。对应报告中的“价格与交易量变量描述性统计”
与“各时期价格与收益率描述性统计”两张表。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PRICE_COLS, DEFAULT_TARGET

# 报告的十年分段 (末段截止到样本终点)
DECADES = [
    (1980, 1989, "1980–1989"),
    (1990, 1999, "1990–1999"),
    (2000, 2009, "2000–2009"),
    (2010, 2019, "2010–2019"),
    (2020, 2025, "2020–2025"),
]

_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]
_QUANTILE_LABELS = ["1%分位", "10%分位", "中位数", "90%分位", "99%分位"]


def descriptive_stats(
    df: pd.DataFrame, cols: list[str] | None = None
) -> pd.DataFrame:
    """计算价格与成交量变量的分位数描述统计。

    Returns
    -------
    pandas.DataFrame
        行索引为统计量 (均值、标准差、最小值、分位数、最大值)，
        列为各变量，与报告表 2.1 的行列一致。
    """
    cols = cols or PRICE_COLS + ["Volume"]
    stats = {"均值": df[cols].mean(), "标准差": df[cols].std(), "最小值": df[cols].min()}

    quant = df[cols].quantile(_QUANTILES)
    for label, value in zip(_QUANTILE_LABELS, quant.itertuples(index=False)):
        stats[label] = pd.Series(value, index=cols)

    stats["最大值"] = df[cols].max()
    return pd.DataFrame(stats).T


def stats_by_decade(
    df: pd.DataFrame,
    price_col: str = DEFAULT_TARGET,
    return_col: str = "return_pct",
) -> pd.DataFrame:
    """按十年区间汇总价格水平与收益率特征。

    收益率口径与报告一致：使用简单百分比收益率 ``return_pct``，峰度取
    Fisher 超额峰度 (正态基准为 0)。

    Returns
    -------
    pandas.DataFrame
        列含时期、样本量、均价、价格范围、日均收益率(%)、收益率波动率(%)
        与收益率峰度，对应报告的“各时期价格与收益率描述性统计”表。
    """
    years = df.index.year
    rows = []
    for start, end, label in DECADES:
        mask = (years >= start) & (years <= end)
        block = df.loc[mask]
        ret_pct = block[return_col].dropna()
        price = block[price_col]
        if len(block) == 0:
            continue
        rows.append(
            {
                "时期": label,
                "样本量": int(mask.sum()),
                "均价（美元）": price.mean(),
                "价格范围": f"{price.min():.2f}–{price.max():.2f}",
                "日均收益率（%）": ret_pct.mean(),
                "收益率波动率（%）": ret_pct.std(),
                "收益率峰度": float(ret_pct.kurtosis()),  # Fisher 超额峰度
            }
        )
    return pd.DataFrame(rows)


def return_summary(returns: pd.Series) -> pd.Series:
    """单个收益率序列的常用矩：均值、标准差、偏度、超额峰度、皮尔逊峰度。"""
    r = returns.dropna()
    return pd.Series(
        {
            "样本量": len(r),
            "均值": r.mean(),
            "标准差": r.std(),
            "偏度": r.skew(),
            "超额峰度": r.kurtosis(),
            "峰度": r.kurtosis() + 3.0,
        }
    )

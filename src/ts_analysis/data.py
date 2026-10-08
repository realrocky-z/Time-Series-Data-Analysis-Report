"""数据载入与清洗 (对应报告第 1–2 节).

数据来源为 Kaggle 公开的 Apple 公司历史股票数据 (AAPL)，覆盖 1980-12-12 至
2025-01-03 共 11,107 个交易日。CSV 的第一列是日期但其表头为空，其余列为
Open / High / Low / Close / Adj Close / Volume。

本模块把这套清洗逻辑固化为可直接复用的函数：解析日期、去重、按日期排序、
建立时间索引，并派生对数价格与对数收益率 ``y_t = ∇ln(x_t)``。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# 默认数据路径：<repo>/data/apple_stock.csv
DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "apple_stock.csv"

PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close"]
DEFAULT_TARGET = "Adj Close"


def load_prices(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """载入原始股票数据并完成清洗。

    Parameters
    ----------
    path:
        CSV 文件路径，默认指向 ``data/apple_stock.csv``。

    Returns
    -------
    pandas.DataFrame
        以 ``DatetimeIndex`` 为索引的价格数据，列包含 ``Open``、``High``、
        ``Low``、``Close``、``Adj Close``、``Volume``。

    Notes
    -----
    清洗步骤：把无表头的第一列命名为 ``date``、解析为日期、去除重复交易日、
    按日期升序排列并设为索引。原始数据本身没有缺失值，这里仍做一次排序与
    去重，以保证在真实业务场景下换用其它数据源时行为一致。
    """
    df = pd.read_csv(path)
    # 第一列无表头，重复的列名会被 pandas 改写，因此按位置重命名。
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = (
        df.dropna(subset=["date"])
        .drop_duplicates(subset="date")
        .sort_values("date")
        .set_index("date")
    )
    return df


def add_returns(df: pd.DataFrame, price_col: str = DEFAULT_TARGET) -> pd.DataFrame:
    """在两列基础上派生对数价格、对数收益率与百分比收益率。

    Parameters
    ----------
    df:
        :func:`load_prices` 返回的价格数据。
    price_col:
        用于计算收益率的基准价格列，报告中使用经拆股与股息调整的
        ``Adj Close``。

    Returns
    -------
    pandas.DataFrame
        在原数据基础上增加 ``log_price``、``log_return``、``return_pct`` 三列。

    Notes
    -----
    ``log_return`` 即报告中的平稳序列 ``y_t = ∇ln(x_t)``。
    """
    out = df.copy()
    out["log_price"] = np.log(out[price_col])
    out["log_return"] = out["log_price"].diff()
    out["return_pct"] = out[price_col].pct_change() * 100.0
    return out


def load_dataset(path: str | Path = DEFAULT_CSV, price_col: str = DEFAULT_TARGET) -> pd.DataFrame:
    """一步完成载入与收益率派生，返回建模所需的完整数据框。"""
    return add_returns(load_prices(path), price_col=price_col)


@dataclass
class Split:
    """训练集 / 测试集的切片结果。

    Attributes
    ----------
    train, test:
        对应的数据框切片 (含价格与收益率列)。
    train_return, test_return:
        训练集与测试集的对数收益率 ``y_t``。
    test_price:
        测试集的价格序列，用于评估预测。
    last_log_price:
        训练集最后一个交易日的对数价格 ``ln(x_T)``，预测的起算点。
    """

    train: pd.DataFrame
    test: pd.DataFrame
    train_return: pd.Series
    test_return: pd.Series
    test_price: pd.Series
    last_log_price: float


def train_test_split(
    df: pd.DataFrame,
    test_size: int = 504,
    price_col: str = DEFAULT_TARGET,
    return_col: str = "log_return",
) -> Split:
    """按报告口径划分训练集与测试集。

    报告以最后 504 个交易日 (2023-01 至 2025-01) 作为测试集，之前的 10,602
    个收益率观测作为训练集。这里按样本量切分，与报告的固定窗口一致。
    """
    ret = df[return_col].dropna()
    n = len(ret)
    if test_size >= n:
        raise ValueError(f"test_size={test_size} 不小于可用观测数 {n}")
    split = n - test_size
    train_return = ret.iloc[:split]
    test_return = ret.iloc[split:]
    return Split(
        train=df.loc[: train_return.index[-1]],
        test=df.loc[test_return.index[0] :],
        train_return=train_return,
        test_return=test_return,
        test_price=df[price_col].loc[test_return.index],
        last_log_price=float(df["log_price"].loc[train_return.index[-1]]),
    )


def annualized_volatility(returns: pd.Series, periods: int = 252) -> float:
    """把日度收益率标准差折算为年化波动率。"""
    return float(returns.std() * np.sqrt(periods))

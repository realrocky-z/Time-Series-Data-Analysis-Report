"""残差诊断与稳健性检验 (对应报告第 5 节).

报告从两个维度检验 GARCH(1,1) 的充分性：标准化残差的白噪声性质
(标准化残差及其平方的 Ljung-Box 检验) 与残差分布诊断 (Q-Q 图、Jarque-Bera)。
随后做三类稳健性检验：子样本重估、条件分布替换 (t → GED)、以及用 EGARCH
检验杠杆效应。本模块把这些检验封装为函数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from . import garch


def residual_white_noise(
    std_residuals: pd.Series, lags=(5, 10, 15, 20)
) -> pd.DataFrame:
    """标准化残差及其平方的 Ljung-Box 检验。

    对应报告“标准化残差的 Ljung-Box 检验”表。若模型充分，标准化残差平方应
    不再具有显著自相关 (p 值大)，即 ARCH 效应已被吸收。
    """
    z = std_residuals.dropna()
    lb_z = acorr_ljungbox(z, lags=list(lags), return_df=True)
    lb_z2 = acorr_ljungbox(z**2, lags=list(lags), return_df=True)
    return pd.DataFrame(
        {
            "滞后阶数": list(lags),
            "残差统计量": lb_z["lb_stat"].values,
            "残差 p值": lb_z["lb_pvalue"].values,
            "残差平方统计量": lb_z2["lb_stat"].values,
            "残差平方 p值": lb_z2["lb_pvalue"].values,
        }
    ).set_index("滞后阶数")


def jarque_bera(series: pd.Series) -> dict:
    """Jarque-Bera 正态性检验，返回统计量与 p 值。"""
    stat, pvalue = stats.jarque_bera(series.dropna())[:2]
    return {"stat": float(stat), "pvalue": float(pvalue)}


def subsample_garch(
    returns: pd.Series, splits
) -> pd.DataFrame:
    """在不同子样本上分别估计 GARCH(1,1)，检验波动率持续性的稳健性。

    Parameters
    ----------
    splits:
        可迭代的 ``(name, start_date, end_date)``，闭区间由 ``pandas`` 切片
        实现。
    """
    rows = []
    for name, start, end in splits:
        sub = returns.loc[start:end].dropna()
        if len(sub) < 250:
            continue
        res = garch.fit_garch(sub)
        rows.append(
            {
                "子样本": name,
                "样本量": len(sub),
                "α1": res.params.get("alpha[1]", np.nan),
                "β1": res.params.get("beta[1]", np.nan),
                "α1+β1": garch.persistence(res),
            }
        )
    return pd.DataFrame(rows)


def compare_distributions(
    returns: pd.Series, dists=("t", "ged", "skewt")
) -> pd.DataFrame:
    """以不同条件分布拟合同一 GARCH(1,1)，比较 AIC。

    对应报告稳健性检验中“改用 GED 分布替代 t 分布”的敏感性分析。
    """
    rows = []
    for dist in dists:
        try:
            spec = dict(mean="Zero", vol="GARCH", p=1, q=1, dist=dist)
            res = garch.fit_spec(returns, spec)
        except Exception:
            continue
        rows.append({"分布": dist, "AIC": res.aic, "BIC": res.bic, "对数似然": res.loglikelihood})
    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


def egarch_leverage(returns: pd.Series) -> pd.Series:
    """拟合 EGARCH(1,1) 并返回参数，用于检验波动率的杠杆(不对称)效应。

    关注不对称项 ``gamma[1]`` 的显著性：若其不显著，说明正负冲击对波动的
    影响对称，即不存在显著的杠杆效应。
    """
    res = garch.fit_spec(
        returns, dict(mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="t")
    )
    return res.params

"""ARCH 效应检验与 ARMA-GARCH 建模 (对应报告第 4 节).

报告在发现 ARIMA 残差存在显著异方差后，先定量检验收益率的 ARCH 效应
(收益率平方的 Ljung-Box 检验与 ARCH-LM 检验)，再比较若干 ARMA-GARCH 候选
模型，最终选定 GARCH(1,1) 并使用 t 分布刻画厚尾。本模块把这些步骤封装成
函数，模型估计依赖 ``arch`` 包。

条件均值方程：r_t = μ + ε_t；条件方差方程：σ_t² = ω + α ε_{t-1}² + β σ_{t-1}²，
其中 ε_t = σ_t z_t，z_t ~ t_ν。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def _fit(returns: pd.Series, spec: dict):
    """拟合 ``arch`` 模型，屏蔽优化器与数据频率相关的告警。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return arch_model(returns.dropna(), **spec).fit(disp="off")

# 报告用到的候选设定，键为模型名，值为 arch_model 的关键字参数
CANDIDATE_SPECS = {
    "ARCH(1)": dict(mean="Zero", vol="GARCH", p=1, q=0, dist="t"),
    "GARCH(1,1)": dict(mean="Zero", vol="GARCH", p=1, q=1, dist="t"),
    "GARCH(2,1)": dict(mean="Zero", vol="GARCH", p=2, q=1, dist="t"),
    "AR(1)-GARCH(1,1)": dict(mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t"),
}


def ljung_box(series: pd.Series, lags=(5, 10, 15, 20)) -> pd.DataFrame:
    """对序列做 Ljung-Box 检验，返回统计量与 p 值表。"""
    res = acorr_ljungbox(series.dropna(), lags=list(lags), return_df=True)
    res.index.name = "lag"
    return res.rename(columns={"lb_stat": "统计量", "lb_pvalue": "p值"})


def arch_lm_test(returns: pd.Series, nlags: int = 12) -> dict:
    """Engle 的 ARCH-LM 检验 (拉格朗日乘子法)。

    原假设为“不存在 ARCH 效应”，返回 ``TR²`` 统计量及其 p 值。
    """
    stat, pvalue, f_stat, f_pvalue = het_arch(returns.dropna(), nlags=nlags)
    return {
        "TR2": float(stat),
        "pvalue": float(pvalue),
        "f_stat": float(f_stat),
        "f_pvalue": float(f_pvalue),
    }


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    mean: str = "Zero",
    dist: str = "t",
    lags=None,
):
    """拟合单个 GARCH(p, q) 模型，返回 ``arch`` 的结果对象。"""
    spec = dict(mean=mean, vol="GARCH", p=p, q=q, dist=dist)
    if lags is not None:
        spec["lags"] = lags
    return _fit(returns, spec)


def fit_spec(returns: pd.Series, spec: dict):
    """按 ``arch_model`` 关键字字典拟合模型，供 :func:`compare_models` 使用。"""
    return _fit(returns, spec)


def compare_models(returns: pd.Series, specs: dict | None = None) -> pd.DataFrame:
    """批量拟合候选模型并按 AIC 排序，输出信息准则比较表。

    对应报告的 ARMA-GARCH 模型信息准则比较表。注意 ``arch`` 包的条件均值
    类型不含 ARMA(1,4)，报告中的 ARMA(1,4)-GARCH(1,1) 由 R 的 rugarch 完成，
    故默认候选集不含该项。
    """
    specs = specs or CANDIDATE_SPECS
    rows = []
    for name, spec in specs.items():
        try:
            res = fit_spec(returns, spec)
        except Exception:
            continue
        rows.append(
            {
                "模型": name,
                "AIC": res.aic,
                "BIC": res.bic,
                "对数似然": res.loglikelihood,
                "fit": res,
            }
        )
    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


def conditional_volatility(result) -> pd.Series:
    """从拟合结果中取出条件标准差序列 σ_t。"""
    return result.conditional_volatility


def standardized_residuals(result) -> pd.Series:
    """标准化残差 z_t = ε_t / σ_t。"""
    return result.resid / result.conditional_volatility


def persistence(result) -> float:
    """波动率持久性 α + β (ARCH 项与 GARCH 项系数之和)。"""
    params = result.params
    return float(params.get("alpha[1]", 0.0) + params.get("beta[1]", 0.0))


def unconditional_volatility(result) -> float:
    """由 GARCH(1,1) 参数推出的无条件标准差 √(ω / (1-α-β))。"""
    params = result.params
    alpha = params.get("alpha[1]", 0.0)
    beta = params.get("beta[1]", 0.0)
    omega = params.get("omega", np.nan)
    return float(np.sqrt(omega / (1.0 - alpha - beta)))


def forecast_volatility(result, horizon: int) -> np.ndarray:
    """基于已拟合模型做多步条件波动率预测，返回预测标准差序列。"""
    fc = result.forecast(horizon=horizon, reindex=False)
    return np.sqrt(fc.variance.values[-1])


def evaluate_volatility_forecast(
    actual_returns: pd.Series,
    predicted_sigma: np.ndarray,
    constant_level: float | None = None,
) -> pd.DataFrame:
    """比较 GARCH 预测波动率与常数历史波动率两种基准。

    以实际收益率绝对值 ``|r_t|`` 作为波动率的代理，计算 RMSE 与 MAE，
    对应报告的样本外预测性能比较表。

    Parameters
    ----------
    constant_level:
        常数波动率基准的取值。为避免前视偏差，应传入**训练集**的平均绝对
        收益率 (报告取 1.95%)；若为 ``None`` 则退化为测试集的平均绝对收益率。
    """
    actual = np.abs(np.asarray(actual_returns.dropna(), dtype=float))
    pred = np.asarray(predicted_sigma, dtype=float)
    n = min(len(actual), len(pred))
    actual, pred = actual[:n], pred[:n]

    if constant_level is None:
        constant_level = float(np.mean(actual))
    const_pred = np.full_like(actual, constant_level)

    def _metrics(p):
        err = actual - p
        return np.sqrt(np.mean(err**2)), np.mean(np.abs(err))

    g_rmse, g_mae = _metrics(pred)
    c_rmse, c_mae = _metrics(const_pred)
    return pd.DataFrame(
        [
            {"模型": f"常数波动率(历史平均={constant_level*100:.2f}%)", "RMSE": c_rmse, "MAE": c_mae},
            {"模型": "GARCH(1,1)", "RMSE": g_rmse, "MAE": g_mae},
        ]
    )

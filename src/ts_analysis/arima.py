"""平稳性检验、ARIMA 建模与预测 (对应报告第 4 节).

报告的建模路径是标准的 Box–Jenkins 流程：先检验趋势与平稳性，通过变换与
差分得到平稳序列，再由 ACF/PACF 与信息准则识别 ARIMA 结构，最后做样本外
预测。报告最终选定的模型是价格序列上的 ``ARIMA(1, 4, 0)``，等价于对数收益率
序列 ``y_t = ∇ln(x_t)`` 上的 ``ARIMA(4, 0, 0)``，即一个 AR(4) 结构。

本模块提供：分层平稳性检验、候选模型网格搜索、模型拟合、价格水平预测与
多步预测误差评估。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class ARIMASelection:
    """网格搜索结果。

    Attributes
    ----------
    order:
        最优模型的阶数 ``(p, d, q)``。
    aic, bic:
        最优模型的信息准则值。
    table:
        全部收敛候选模型的 AIC 排序表。
    fit:
        最优模型的 ``statsmodels`` 拟合结果对象。
    """

    order: tuple[int, int, int]
    aic: float
    bic: float
    table: pd.DataFrame
    fit: object


def test_stationarity(series: pd.Series) -> dict:
    """对单条序列同时做 ADF 与 KPSS 检验。

    ADF 的原假设是“存在单位根 (非平稳)”，KPSS 的原假设是“平稳”，两者互为
    补充。这里沿用报告的口径：ADF 拒绝且 KPSS 不拒绝时判为平稳。
    """
    s = series.dropna()
    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(s, autolag="AIC")
    with warnings.catch_warnings():
        # KPSS 的 p 值可能落在查表范围之外，会抛 InterpolationWarning
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, _, _ = kpss(s, regression="c", nlags="auto")
    return {
        "adf_stat": adf_stat,
        "adf_pvalue": adf_p,
        "adf_crit_1pct": adf_crit["1%"],
        "adf_crit_5pct": adf_crit["5%"],
        "kpss_stat": kpss_stat,
        "kpss_pvalue": kpss_p,
        "stationary": bool(adf_p < 0.05 and kpss_p > 0.05),
    }


def stationarity_stages(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    """复现报告的数据变换三阶段平稳性对照表。

    阶段 A 为原始价格 ``x_t``，阶段 B 为对数价格 ``ln(x_t)``，阶段 C 为对数
    一阶差分 ``y_t = ∇ln(x_t)``。返回的表格直接对应报告中的分层平稳性检验表。
    """
    stages = {
        "A：原始": (f"{price_col}", df[price_col]),
        "B：对数变换后": ("ln(x_t)", np.log(df[price_col])),
        "C：对数一阶差分": ("y_t = ∇ln(x_t)", np.log(df[price_col]).diff()),
    }
    rows = []
    for stage, (name, series) in stages.items():
        res = test_stationarity(series)
        rows.append(
            {
                "检验阶段": stage,
                "序列": name,
                "ADF统计量": res["adf_stat"],
                "ADF p值": res["adf_pvalue"],
                "结论": "平稳" if res["stationary"] else "非平稳",
            }
        )
    return pd.DataFrame(rows)


def fit_arima(series: pd.Series, order: tuple[int, int, int] = (4, 0, 0), maxiter: int = 500):
    """拟合一个 ARIMA(p, d, q) 模型，返回 ``statsmodels`` 结果对象。"""
    with warnings.catch_warnings():
        # 屏蔽索引频率与个别阶数不收敛的告警，保持批量定阶过程输出干净
        warnings.simplefilter("ignore")
        model = ARIMA(series.dropna(), order=order)
        return model.fit(method_kwargs={"maxiter": maxiter})


def select_arima(
    series: pd.Series, max_order: int = 4, d: int = 0
) -> ARIMASelection:
    """在 ``p, q ∈ [0, max_order]`` 上网格搜索，按 AIC 排序返回最优模型。

    报告在 ``y_t`` 上遍历 ``p, q ∈ {0,…,4}`` 共 25 个候选 (剔除白噪声)，发现
    ACF 与 PACF 形态接近、弱相关，故用信息准则而非图形主观定阶。默认返回
    AIC 最小的模型；报告最终以“参数全部显著 + BIC 更小”为准选择了
    ``ARIMA(4, 0, 0)`` (见 :func:`fit_arima` 的默认阶数)。
    """
    candidates = [
        (p, d, q)
        for p in range(max_order + 1)
        for q in range(max_order + 1)
        if not (p == 0 and q == 0)
    ]
    rows = []
    for order in candidates:
        try:
            fit = fit_arima(series, order=order)
            rows.append({"order": order, "aic": fit.aic, "bic": fit.bic, "fit": fit})
        except Exception:  # 个别阶数可能不收敛，跳过即可
            continue
    if not rows:
        raise RuntimeError("所有候选 ARIMA 模型均未收敛")
    rows.sort(key=lambda r: r["aic"])
    table = pd.DataFrame(
        [{"order": r["order"], "aic": r["aic"], "bic": r["bic"]} for r in rows]
    ).reset_index(drop=True)
    best = rows[0]
    return ARIMASelection(best["order"], best["aic"], best["bic"], table, best["fit"])


def forecast_prices(
    fit, last_log_price: float, steps: int, alpha: float = 0.05
) -> pd.DataFrame:
    """把对数收益率的预测累计回价格水平，并给出预测区间。

    Parameters
    ----------
    fit:
        :func:`fit_arima` 的返回值。
    last_log_price:
        训练集最后一个交易日的对数价格 ``ln(x_T)``。
    steps:
        预测步数 (报告测试集为 504 个交易日)。
    alpha:
        显著性水平，默认 0.05 对应 95% 预测区间。

    Returns
    -------
    pandas.DataFrame
        含 ``forecast``、``lower``、``upper`` 三列 (价格水平)。
    """
    from scipy import stats

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = fit.get_forecast(steps=steps)
    mean_y = np.asarray(fc.predicted_mean)
    se_y = np.asarray(fc.se_mean) * stats.norm.ppf(1 - alpha / 2)
    cum = np.cumsum(mean_y)
    cum_se = np.cumsum(se_y)
    return pd.DataFrame(
        {
            "forecast": np.exp(last_log_price + cum),
            "lower": np.exp(last_log_price + cum - cum_se),
            "upper": np.exp(last_log_price + cum + cum_se),
        }
    )


def evaluate_forecast(
    actual_price: pd.Series, predicted_price: pd.Series
) -> dict:
    """计算价格预测的 RMSE、MAE 与 MAPE (百分比)。"""
    actual = np.asarray(actual_price, dtype=float)
    pred = np.asarray(predicted_price, dtype=float)
    err = actual - pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / actual)) * 100.0)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def horizon_rmse(
    fit, last_log_price: float, actual_price: pd.Series, horizons=(63, 126, 252, 504)
) -> pd.DataFrame:
    """报告口径的多步预测误差表 (63/126/252/504 天)。

    每一步都从训练集末端重新预测到该步长，再与对应长度的真实价格比较。
    """
    rows = []
    for h in horizons:
        if h > len(actual_price):
            continue
        fc = forecast_prices(fit, last_log_price, steps=h)
        metrics = evaluate_forecast(actual_price.iloc[:h], fc["forecast"])
        rows.append({"horizon": h, **metrics})
    return pd.DataFrame(rows)


def rolling_forecast_rmse(
    df: pd.DataFrame,
    windows,
    order: tuple[int, int, int] = (4, 0, 0),
    price_col: str = "Adj Close",
    return_col: str = "log_return",
) -> pd.DataFrame:
    """滚动窗口稳健性检验：在不同子样本上拟合并预测，比较 RMSE。

    Parameters
    ----------
    windows:
        可迭代的 ``(start, end, horizon, label)``，索引为 ``df`` 的行号。
    """
    rows = []
    for start, end, horizon, label in windows:
        if end + horizon > len(df):
            continue
        train = df[return_col].iloc[start:end].dropna()
        test_price = df[price_col].iloc[end : end + horizon]
        last_log_price = df["log_price"].iloc[end - 1]
        try:
            fit = fit_arima(train, order=order)
            fc = forecast_prices(fit, last_log_price, horizon)
            metrics = evaluate_forecast(test_price, fc["forecast"])
            rows.append(
                {
                    "label": label,
                    "start": df.index[start].date(),
                    "end": df.index[min(end + horizon - 1, len(df) - 1)].date(),
                    **metrics,
                }
            )
        except Exception as exc:
            rows.append({"label": label, "error": str(exc)})
    return pd.DataFrame(rows)

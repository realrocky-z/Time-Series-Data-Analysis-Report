"""ts_analysis — Apple (AAPL) 日度股票时间序列分析的可复用代码包.

模块与报告章节的对应关系:

- :mod:`ts_analysis.data`       —— 数据载入与清洗 (报告第 1–2 节)
- :mod:`ts_analysis.eda`        —— 描述性统计与分布特征 (报告第 3 节)
- :mod:`ts_analysis.arima`      —— 平稳性检验、ARIMA 建模与预测 (报告第 4 节)
- :mod:`ts_analysis.garch`      —— ARCH 效应检验、ARMA-GARCH 建模与样本外预测 (报告第 4 节)
- :mod:`ts_analysis.diagnostics`—— 残差诊断与稳健性检验 (报告第 5 节)

典型用法::

    from ts_analysis import data, arima, garch

    df = data.load_dataset()                 # 载入并计算对数收益率
    train, test = data.train_test_split(df)  # 按报告划分训练/测试集
    fit = arima.fit_arima(train["log_return"], order=(4, 0, 0))
    fc = arima.forecast_prices(fit, last_log_price=..., steps=504)
"""

from . import data, eda, arima, garch, diagnostics

__all__ = ["data", "eda", "arima", "garch", "diagnostics"]
__version__ = "1.0.0"

# Time Series Data Analysis Report

苹果公司（AAPL）日度股票时间序列分析。数据覆盖 1980-12-12 至 2025-01，约 11,000 个交易日。
本项目把完整分析流程整理为三部分：可复用的 Python 代码包、一个端到端的总览 notebook、以及最终研究报告。

## 目录结构

```
Time Series Data Analysis Report/
├── data/
│   └── apple_stock.csv          # 原始数据（无表头首列为日期）
├── src/ts_analysis/             # 可复用分析代码包
│   ├── data.py                  # 数据载入、清洗、派生变量、训练/测试切分
│   ├── eda.py                   # 描述性统计与分时期统计
│   ├── arima.py                 # 平稳性检验、ARIMA 定阶/拟合/预测、误差评估
│   ├── garch.py                 # ARCH 检验、ARMA-GARCH 拟合与波动率预测
│   └── diagnostics.py           # 残差诊断与稳健性检验
├── notebooks/
│   └── apple_analysis_overview.ipynb   # 端到端总览（按报告章节顺序）
├── report/
│   ├── report.tex / report.pdf  # 最终报告
│   ├── flowchart.png            # 分析流程
│   └── pic/                     # 报告插图
├── pyproject.toml               # 代码包配置
└── _build_notebook.py           # 生成并执行 notebook 的脚本
```

## 快速开始

安装代码包（建议在虚拟环境中）：

```bash
pip install -e .
```

然后在 `notebooks/` 目录下打开 `apple_analysis_overview.ipynb` 运行。notebook 第 1 个单元格会把 `../src` 加入 `sys.path`，无需额外配置即可 `import ts_analysis`。

也可以直接在脚本中调用：

```python
from ts_analysis import data, eda, arima, garch, diagnostics

df = data.load_dataset()
sp = data.train_test_split(df, test_size=504)
fit = arima.fit_arima(sp.train_return, order=(4, 0, 0))
```

## 分析流程（Box–Jenkins）

整体思路是先刻画条件均值，再刻画条件方差：

1. **数据清洗**：解析日期、去重排序、建立时间索引，派生对数价格与对数收益率 `y_t = ∇ln(x_t)`。
2. **描述性统计**：价格与收益率的分位数统计、分时期（十年）统计，观察尖峰厚尾。
3. **平稳性**：原始价格与对数价格均非平稳，对数一阶差分平稳，确定差分阶数 `d = 1`。
4. **条件均值**：在训练集上网格搜索 ARIMA 阶数，选定 `ARIMA(4,0,0)`（即价格上的 `ARIMA(1,4,0)`），但残差存在显著 ARCH 效应。
5. **条件方差**：改用 GARCH 类模型，以 AIC/对数似然比较候选，选定 `GARCH(1,1)` 并取 t 分布刻画厚尾。
6. **样本外预测**：训练集 1980-12 至 2022-12（10,602 个观测），测试集 2023-01 至 2025-01（504 个观测）。
7. **检验与稳健性**：标准化残差白噪声检验、子样本重估、条件分布替换、EGARCH 杠杆效应。

## notebook 与报告章节对照

| notebook 小节 | 报告章节 |
|---|---|
| 1. 数据载入与清洗 | 第 1–2 节 |
| 2. 描述性统计与分布 | 第 3 节 |
| 3. 平稳性检验与数据变换 | 第 4 节 |
| 4. ARIMA 建模与预测 | 第 4 节 |
| 5. ARIMA 残差诊断与异方差 | 第 4.1 节 |
| 6. 波动聚集与 ARCH 效应检验 | 第 4.2 节 |
| 7. ARMA-GARCH 模型建立 | 第 4.3 节 |
| 8. 样本外预测验证 | 第 4.4 节 |
| 9. 模型检验与稳健性 | 第 5 节 |

## 说明

报告中的部分数字来自 R 的 `rugarch` 包，本项目统一用 Python（`statsmodels` + `arch`）实现。
模型选择与结论一致，个别数值（如子样本 GARCH 参数、个别检验统计量）会与报告略有出入。
`_build_notebook.py` 用于在缺少 nbformat/nbconvert 的环境下手工构造并执行 notebook。

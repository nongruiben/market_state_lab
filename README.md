# Market State Lab

面向美股收盘后研究的市场风险状态、市场风格与新闻事件实验室。项目只读取公开数据和已授权的 IBKR 数据，不包含下单接口。

## 当前方法

- 风险状态语义为 `low_risk / mid_risk / high_risk`，不再把中间状态误称为 transition。
- 同一套因果 walk-forward 引擎比较风险分位基线、对角 GMM、仅前向滤波的对角 HMM。
- 成分排序使用软归属风险，空成分或有效样本不足时自动跳过该次模型。
- 模型权重只使用截至前一日的滚动 Brier 分数校准，且基线权重不低于 40%。
- 原始状态概率与 15 日半衰期的决策权重分别输出。
- 风格模块分别报告 Fama-French 与 ETF 代理，不再把 soft sign 称为概率。
- 新闻通过 TWS 只读 sidecar 获取，经严格时间窗、去重、事件聚类和结构化 LLM 校验后作为 challenger overlay；默认不进入状态模型。

## 安装

在 PyCharm 中打开本目录，选择 Python 3.11-3.13，然后在 Terminal 运行：

```powershell
.\bootstrap.ps1
.\.venv\Scripts\python.exe scripts\doctor.py
```

需要运行现有 TWS 新闻 sidecar 时：

```powershell
.\bootstrap.ps1 -WithNews
```

## 运行

正常日更依次尝试 Stooq、低频串行且缓存 24 小时的 Yahoo Chart，最后才以 Nasdaq 网页接口作逐标的兜底。公开端点的成功与失败都会保留在 manifest：

```powershell
.\.venv\Scripts\python.exe scripts\run_daily.py
```

使用固定脱敏样本验证完整流程，不访问网络：

```powershell
.\.venv\Scripts\market-state.exe run --offline
```

显式增加 IBKR 延迟快照：

```powershell
.\.venv\Scripts\python.exe scripts\run_daily.py --with-ibkr
```

TWS 必须保持 `Read-Only API`。项目不会自动连接 IBKR；历史行情还要求 `IBKR_ALLOW_HISTORICAL=1` 以及账户已有数据权限。

## 新闻流程

先处理已有的 TWS/Web JSON；有 `DEEPSEEK_API_KEY` 时生成结构化事件：

```powershell
.\.venv\Scripts\market-state.exe news
```

先显式运行只读 TWS sidecar，再处理数据：

```powershell
.\.venv\Scripts\market-state.exe news --fetch
```

只检查时间窗、去重、聚类和数据质量，不调用 LLM：

```powershell
.\.venv\Scripts\market-state.exe news --no-llm
```

新闻输出包括 `data/processed/news_events.parquet`、`news_daily_features.parquet`、`reports/news_quality.json` 和 `news_llm_metadata.json`。模型、prompt 版本、hash、重试和 token usage 均保留。原始新闻与 LLM 原始响应被 `.gitignore` 排除，避免把授权正文或敏感输出上传 GitHub。

## 历史数据口径

默认 FRED 为 latest vintage，当前判断可用，但历史行不允许宣称为严格样本外。JSON/HTML 会显示：

- `history_is_latest_vintage`
- `historical_backtest_eligible`
- `information_date`
- `run_date`

启用 ALFRED initial release：

1. 设置环境变量 `FRED_API_KEY`。
2. 将 `data.fred.vintage_mode` 改为 `point_in_time`。
3. 严格研究时将 `data.strict_history` 改为 `true`，并禁用仍会修订的 French 历史输入。

ALFRED 数据按首次公开可得日索引，不再额外套固定发布滞后。每次运行还会按 XNYS 交易日检查 VIX、SPY 和其他数据源的新鲜度；必需源失败时流水线直接失败，非必需源会从本次模型特征中移除。

## 主要输出

- `reports/market_state_dashboard.html`：综合报告。
- `reports/latest_market_state.json`：状态概率、决策权重、信息日与模型权重。
- `reports/market_state_history.parquet`：walk-forward 历史。
- `reports/model_comparison.csv`：Brier、翻转率和平均持续时间。
- `reports/decision_value_comparison.csv`：相对纯波动率控制的校准和回撤对照。
- `reports/model_refit_diagnostics.csv`：每次 GMM/HMM 重估结果及空成分保护记录。
- `reports/latest_style_state.csv`：FF/ETF 分列风格分数和一致性。
- `reports/data_manifest.csv`：抓取状态、交易日年龄、阈值与模型可用性。

所有输出都是研究证据，不是自动交易指令。

## 测试与复现

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src tests scripts
```

测试包含整条 `build_features -> fit_market_state` 的前缀不变性、合成体制恢复、HMM 前向过滤、备用数据源健康判断、新闻时间窗和证据 ID 守卫。GitHub Actions 还会执行一次固定 fixture 的完整离线报告。

本项目采用 MIT License。

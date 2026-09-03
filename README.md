# Market State Lab

面向美股收盘后研究的市场风险状态、市场风格与新闻事件实验室。项目只读取公开数据和已授权的 IBKR 数据，不包含下单接口。

## 当前方法

- 风险状态语义为 `low_risk / mid_risk / high_risk`，不再把中间状态误称为 transition。
- 同一套因果 walk-forward 引擎比较风险分位基线、对角 GMM、仅前向滤波的对角 HMM。
- 成分排序使用软归属风险，空成分或有效样本不足时自动跳过该次模型。
- 模型权重只使用截至前一日的滚动 Brier 分数校准，且基线权重不低于 40%。
- 风格模块分别报告 Fama-French 与 ETF 代理，不再把 soft sign 称为概率。
- 新闻通过 TWS 只读 sidecar 获取，经严格时间窗、去重、事件聚类和结构化 LLM 校验后作为 challenger overlay；默认不进入状态模型。

### 两个 Brier，不能混用

`self_consistency_brier` 的目标由 `risk_percentile` 在 0.38 / 0.62 硬切得到，而基线又是同一个分位在同样节点上的 sigmoid——拿它给基线打分是恒等式，不是证据。真正的准确度是 `forward_brier`：目标为未来 20 个交易日已实现波动率的三分位，三分位阈值只用在 t 时刻已经完全实现的历史，因此标注本身不含前视。

`model_comparison.csv` 两个指标并列输出，并且**永远带一行 `climatology`**（滚动基率，无技能参照）。技能差 `forward_brier_vs_climatology` 只在「该模型与 climatology 都有输出」的公共交集上配对计算，避免拿不同样本期互比。跑不赢 climatology 就是没有技能。

### 校准层

原始状态的命中率高于 climatology，但 Brier 低于它——信息是有的，坏的是过度自信。因此在集成之上加了一层 walk-forward 温度缩放：每次重估用滚动窗口内**已结算**的前瞻目标拟合一个标量温度，再样本外应用。温度缩放是单调的，只改变置信度、不会改变哪个状态排第一。

原始概率 `p_*`、校准后概率 `calibrated_p_*` 与 15 日半衰期的决策权重 `decision_p_*` 分别输出；`decision_*` 默认由校准后概率生成（`models.market_state.calibration.apply_to_decision`）。`latest_market_state.json` 的 `decision_weight_source` 和 `decision_half_life_days` 说明该按哪个数字行动——决策权重按设计滞后状态约 15 个交易日。

### risk_score 的口径是固定的

`risk_score` 是 `features.required_risk_blocks` 里那几个 block 的均值，**永远是同一组**。此前它对「当天恰好可得的 block」取均值，于是在 HYG 历史开始前是 4-block 量、之后是 5-block 量，而滚动分位又把这两者放在同一个窗口里排名。现在缺任何一个必需 block 就直接输出 NaN，并同时给出 `risk_score_block_count`。`risk_credit` 暂时不在必需列表内，原因见下节。

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

### 覆盖率检查（新鲜 ≠ 完整）

只查新鲜度看不见「按时到达但历史被截断」。FRED 的公开 graph CSV 对受 ICE 许可限制的 `BAMLH0A0HYM2` / `BAMLC0A0CM` 只返回约 3 年滚动窗口（对 `T10Y2Y` 则返回 1976 年至今），任何 `cosd` / `coed` 组合都无法解除。结果是 `hy_oas` 每天都显示 `cache_fresh`，却缺了 89% 的历史。

因此 `source_health` 增加了 `min_history_years`、`min_rows` 与 `min_observation_density`（行数 ÷ 跨度内的工作日数，周频序列请保持为 0）。不满足即判 `health_status: truncated` 且 `model_eligible: False`，manifest 里可见。离线 fixture 的跨度是刻意设短的，因此豁免深度与密度检查。

要真正拿回这两条序列的完整历史，需要一个免费的 FRED API key：设置 `FRED_API_KEY` 后，`data.fred.prefer_api` 会走官方 observations 端点（失败时回退 graph CSV，provider 字段会标明）。没有 key 时它们会以 `truncated` 明确失败，而不是无声地从模型里消失。

每次运行还会输出 `reports/feature_coverage.csv`：逐特征的非空覆盖率。`required: true` 的特征低于阈值会让运行直接失败，`required: false` 的仍会标 `below_threshold`。

## 主要输出

- `reports/market_state_dashboard.html`：综合报告。
- `reports/latest_market_state.json`：原始概率、校准后概率、决策权重与其来源、模型权重、`risk_score_block_count`，以及 `forward_skill`（是否跑赢 climatology）。
- `reports/market_state_history.parquet`：walk-forward 历史，含 `calibrated_p_*` 与 `calibration_temperature`。
- `reports/model_comparison.csv`：`self_consistency_brier` 与 `forward_brier` 并列，含 climatology 行与配对技能差。
- `reports/feature_coverage.csv`：逐特征覆盖率、阈值与是否达标。
- `reports/decision_value_comparison.csv`：相对纯波动率控制的校准和回撤对照。
- `reports/model_refit_diagnostics.csv`：每次 GMM/HMM 重估结果及空成分保护记录。
- `reports/latest_style_state.csv`：FF/ETF 分列风格分数和一致性。
- `reports/data_manifest.csv`：抓取状态、交易日年龄、历史深度与密度、阈值与模型可用性。

所有输出都是研究证据，不是自动交易指令。

## 测试与复现

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src tests scripts
```

测试包含整条 `build_features -> fit_market_state` 的前缀不变性（现已覆盖 `calibrated_p_*` 与 `calibration_temperature`，因为校准层是最容易重新引入前视的地方）、合成体制恢复、HMM 前向过滤、备用数据源健康判断、新闻时间窗和证据 ID 守卫。

另外还钉住了几条口径不变量：`model_comparison.csv` 必须同时给出两个 Brier 且必须带 climatology 行、技能差必须配对计算、`risk_score` 只在必需 block 齐全时才有值、温度缩放不得改变 argmax。GitHub Actions 还会执行一次固定 fixture 的完整离线报告。

本项目采用 MIT License。

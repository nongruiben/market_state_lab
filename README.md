# Market State Lab

Standalone US market-state and market-style research project. It is designed for
one reliable update after the US close, with optional read-only delayed snapshots
from a live IBKR TWS session.

## Safety contract

- The IBKR adapter contains quote, contract, current-time, and historical-data
  reads only. It has no order-placement or order-cancellation calls.
- TWS must keep `Read-Only API` enabled.
- IBKR is never connected automatically. Add `--with-ibkr` explicitly.
- Historical bars are disabled unless `IBKR_ALLOW_HISTORICAL=1`, and still work
  only when the logged-in username already has the required entitlement.
- MegaDB is disabled and is not needed by the pipeline.

## Data route

- Kenneth French daily factors: market, size, value, quality/profitability,
  investment, momentum, and reversal research history.
- FRED: credit spreads, yield curve, and financial-conditions series.
- Cboe: official VIX daily history.
- OFR: official daily Financial Stress Index.
- Nasdaq: best-effort ETF proxy history from Nasdaq's public website endpoint.
  Requests are serial, jittered, cached for 24 hours, and never the only source
  of the core market-risk signal.
- IBKR: optional delayed snapshots and optional entitled historical bars.

The downloaded dataset is small: a few dozen time series rather than a full stock
database. Raw downloads are cached under `data/raw` and derived files under
`data/processed`.

## PyCharm setup

1. Open this directory as a PyCharm project.
2. Select Python 3.11-3.13 as the interpreter.
3. Run `bootstrap.ps1` in the PyCharm terminal.
4. Run the `scripts/doctor.py` file.
5. Run `scripts/run_daily.py` after the US close.

PowerShell equivalents:

```powershell
.\bootstrap.ps1
.\.venv\Scripts\python.exe scripts\doctor.py
.\.venv\Scripts\python.exe scripts\run_daily.py
```

To add the optional IBKR delayed snapshot, first start TWS, enable socket clients,
keep `Read-Only API` checked, and run:

```powershell
.\.venv\Scripts\python.exe scripts\run_daily.py --with-ibkr
```

TWS live defaults to port `7496`; paper TWS defaults to `7497`. This project uses
client ID `91` to reduce collisions with other API tools.

## Outputs

- `reports/latest_market_state.json`: current calm/transition/stress probabilities.
- `reports/latest_style_state.csv`: current style dimensions and confidence.
- `reports/market_state_history.parquet`: walk-forward out-of-sample state history.
- `reports/style_history.parquet`: style score and probability history.
- `reports/data_manifest.csv`: source, freshness, coverage, and cache status.
- `reports/market_state_dashboard.html`: consolidated offline-friendly report.
- `reports/ibkr_snapshot.csv`: optional delayed IBKR snapshot.

## Research interpretation

Market-state probabilities are an ensemble of a robust observable risk score and
an expanding-window Gaussian-mixture regime model. Model states are ordered by the
training sample's risk level, and historical probabilities are generated only from
past data with quarterly refits.

Style probabilities combine Fama-French factor returns with ETF relative-return
proxies when those proxies are available. Each horizon is normalized using trailing
data only. The output is evidence for exposure decisions, not an automatic trade.

Latest-vintage FRED values are suitable for today's dashboard but not automatically
safe for historical backtests. Strict macro backtests should be rebuilt from ALFRED
vintages using a personal `FRED_API_KEY`.

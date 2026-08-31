from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "fixtures" / "market_daily.csv"


def main() -> None:
    rng = np.random.default_rng(20260828)
    dates = pd.bdate_range("2021-01-04", "2026-08-27")
    block = max(1, len(dates) // 7)
    regime = np.resize(np.repeat([0, 0, 1, 2, 1, 0, 2], block), len(dates))
    volatility = np.choose(regime, [0.007, 0.013, 0.024])
    market_return = np.choose(regime, [0.0005, 0.0001, -0.0006]) + rng.normal(0, volatility)
    base = 100 * np.exp(np.cumsum(market_return))
    frame = pd.DataFrame({"date": dates, "spy": base})
    for name, tilt in {
        "qqq": 0.00008,
        "rsp": -0.00002,
        "iwm": 0.00003,
        "iwf": 0.00006,
        "iwd": -0.00004,
        "mtum": 0.00005,
        "qual": 0.00003,
        "usmv": -0.00002,
        "hyg": -0.00005,
        "lqd": 0.00001,
        "tlt": 0.0,
        "gld": 0.0,
    }.items():
        frame[name] = base * np.exp(np.cumsum(tilt + rng.normal(0, 0.002, len(dates))))
    frame["vix_close"] = 13 + regime * 10 + rng.normal(0, 1, len(dates))
    frame["hy_oas"] = 3 + regime * 1.8 + rng.normal(0, 0.15, len(dates))
    frame["ig_oas"] = 0.9 + regime * 0.45 + rng.normal(0, 0.04, len(dates))
    frame["yield_curve_10y2y"] = 1.0 - regime * 0.4
    frame["yield_curve_10y3m"] = 1.2 - regime * 0.45
    frame["financial_conditions"] = -0.5 + regime * 0.7
    frame["fed_stress"] = -0.7 + regime * 1.0
    frame["ofr_fsi"] = -1.0 + regime * 1.4 + rng.normal(0, 0.08, len(dates))
    frame["mkt_rf"] = market_return - 0.00004
    frame["rf"] = 0.00004
    for name, scale in {
        "smb": 0.006,
        "hml": 0.005,
        "rmw": 0.004,
        "cma": 0.003,
        "momentum": 0.006,
        "short_reversal": 0.004,
        "long_reversal": 0.003,
    }.items():
        frame[name] = rng.normal(0, scale, len(dates))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False, float_format="%.8f")
    print(f"Wrote {len(frame)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()

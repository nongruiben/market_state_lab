from __future__ import annotations

from typing import Any

import pandas as pd

from market_state_lab.config import project_path
from market_state_lab.data.public import PublicDataBundle


def load_offline_fixture(config: dict[str, Any]) -> PublicDataBundle:
    path = project_path(config, "data", "raw", "fixtures", "market_daily.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Offline fixture is missing: {path}. Run scripts/generate_offline_fixture.py."
        )
    raw = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    macro_columns = (
        "hy_oas",
        "ig_oas",
        "yield_curve_10y2y",
        "yield_curve_10y3m",
        "financial_conditions",
        "fed_stress",
    )
    french_columns = (
        "mkt_rf",
        "smb",
        "hml",
        "rmw",
        "cma",
        "rf",
        "momentum",
        "short_reversal",
        "long_reversal",
    )
    etf_columns = tuple(config["data"]["nasdaq"]["symbols"])
    latest = raw.index.max().date().isoformat()
    earliest = raw.index.min().date().isoformat()
    rows = len(raw)
    datasets = [
        *(str(name) for name in macro_columns),
        "vix",
        "financial_stress_index",
        "ff5",
        "momentum",
        "short_reversal",
        "long_reversal",
        *(str(name) for name in etf_columns),
    ]
    manifest = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "provider": "offline synthetic fixture",
                "status": "fixture",
                "rows": rows,
                "earliest_date": earliest,
                "latest_date": latest,
                "vintage_mode": "synthetic",
                "error": "",
            }
            for dataset in datasets
        ]
    )
    return PublicDataBundle(
        macro=raw[list(macro_columns)],
        vix=raw[["vix_close"]],
        ofr=raw[["ofr_fsi"]],
        french=raw[list(french_columns)],
        etf_close=raw[[column for column in etf_columns if column in raw]],
        manifest=manifest,
        point_in_time_status="synthetic_fixture",
    )

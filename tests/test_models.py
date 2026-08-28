from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_state_lab.config import load_config
from market_state_lab.data.public import PublicDataBundle
from market_state_lab.features import build_features
from market_state_lab.models import fit_market_state, fit_style


def _synthetic_bundle() -> tuple[PublicDataBundle, pd.DatetimeIndex]:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-02", periods=1100, freq="B")
    regime = np.repeat([0, 1, 2, 1, 0], 220)
    volatility = np.choose(regime, [0.007, 0.012, 0.025])
    drift = np.choose(regime, [0.0006, 0.0001, -0.0008])
    market_return = drift + rng.normal(0, volatility)
    rf = np.full(len(dates), 0.00004)
    french = pd.DataFrame(
        {
            "mkt_rf": market_return - rf,
            "smb": 0.0001 + rng.normal(0, 0.006, len(dates)),
            "hml": np.where(regime == 2, 0.0005, -0.0001) + rng.normal(0, 0.005, len(dates)),
            "rmw": np.where(regime == 2, 0.0004, 0.0001) + rng.normal(0, 0.004, len(dates)),
            "cma": rng.normal(0, 0.003, len(dates)),
            "rf": rf,
            "momentum": np.where(regime == 0, 0.0004, -0.0001) + rng.normal(0, 0.006, len(dates)),
            "short_reversal": rng.normal(0, 0.004, len(dates)),
            "long_reversal": rng.normal(0, 0.003, len(dates)),
        },
        index=dates,
    )
    macro = pd.DataFrame(
        {
            "hy_oas": 3.0 + regime * 2.0 + rng.normal(0, 0.2, len(dates)),
            "ig_oas": 0.9 + regime * 0.5 + rng.normal(0, 0.05, len(dates)),
            "yield_curve_10y2y": 1.0 - regime * 0.4,
            "yield_curve_10y3m": 1.2 - regime * 0.5,
            "financial_conditions": -0.5 + regime * 0.8,
            "fed_stress": -0.7 + regime * 1.1,
        },
        index=dates,
    )
    vix = pd.DataFrame({"vix_close": 13 + regime * 12 + rng.normal(0, 1, len(dates))}, index=dates)
    ofr = pd.DataFrame({"ofr_fsi": -1 + regime * 1.5 + rng.normal(0, 0.1, len(dates))}, index=dates)
    base_prices = 100 * np.exp(np.cumsum(market_return))
    etf_close = pd.DataFrame(index=dates)
    for name, tilt in {
        "spy": 0.0,
        "qqq": 0.0001,
        "rsp": -0.00005,
        "iwm": 0.00005,
        "iwf": 0.0001,
        "iwd": -0.0001,
        "mtum": 0.00008,
        "qual": 0.00004,
        "usmv": -0.00002,
        "hyg": -0.0001,
        "lqd": 0.00002,
        "tlt": 0.0,
        "gld": 0.0,
    }.items():
        noise = rng.normal(0, 0.002, len(dates))
        etf_close[name] = base_prices * np.exp(np.cumsum(tilt + noise))
    bundle = PublicDataBundle(macro, vix, ofr, french, etf_close, pd.DataFrame())
    return bundle, dates


def test_feature_lag_and_walk_forward_models() -> None:
    bundle, dates = _synthetic_bundle()
    config = load_config()
    config["project"]["start_date"] = str(dates[0].date())
    config["features"]["rolling_normalization_window"] = 252
    config["models"]["market_state"]["minimum_train_observations"] = 400
    config["models"]["market_state"]["refit_every_observations"] = 126
    features = build_features(bundle, config)

    assert np.isnan(features.source_panel.loc[dates[0], "ff_mkt_rf"])
    assert features.source_panel.loc[dates[1], "ff_mkt_rf"] == pytest.approx(bundle.french.loc[dates[0], "mkt_rf"])

    state = fit_market_state(features.market, config)
    probabilities = state.history[["p_calm", "p_transition", "p_stress"]].dropna()
    assert not probabilities.empty
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert state.latest["market_state"] in {"calm", "transition", "stress"}
    assert state.feature_columns

    style = fit_style(features.style_returns, config)
    assert not style.latest.empty
    assert style.latest["favored_probability"].between(0.5, 1.0).all()


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from market_state_lab.data.public import PublicDataBundle


@dataclass
class FeatureSet:
    market: pd.DataFrame
    style_returns: pd.DataFrame
    source_panel: pd.DataFrame


def _calendar(
    bundle: PublicDataBundle,
    start: pd.Timestamp,
    as_of: pd.Timestamp,
) -> pd.DatetimeIndex:
    indexes = [
        frame.index
        for frame in (bundle.macro, bundle.vix, bundle.ofr, bundle.french, bundle.etf_close)
        if not frame.empty
    ]
    if not indexes:
        raise ValueError("No public data was downloaded")
    first_observation = min(index.min() for index in indexes)
    return pd.date_range(max(start, first_observation), as_of.normalize(), freq="B")


def _available_panel(
    frame: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    lag: int,
    prefix: str,
    forward_fill_limit: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=calendar)
    result = frame.reindex(calendar).ffill(limit=forward_fill_limit).shift(lag)
    result.columns = [f"{prefix}{column}" for column in result.columns]
    return result


def _rolling_compound(returns: pd.Series, window: int) -> pd.Series:
    safe = returns.clip(lower=-0.999999)
    return np.expm1(np.log1p(safe).rolling(window, min_periods=window).sum())


def build_features(
    bundle: PublicDataBundle,
    config: dict[str, Any],
    as_of: str | pd.Timestamp | None = None,
) -> FeatureSet:
    start = pd.Timestamp(config["project"]["start_date"])
    lags = config["data"]["publication_lags_business_days"]
    runtime_as_of = as_of or config.get("_runtime", {}).get("market_session")
    end = pd.Timestamp(runtime_as_of) if runtime_as_of else pd.Timestamp.today().normalize()
    calendar = _calendar(bundle, start, end)
    fred_lag_daily = int(lags["fred_daily"])
    fred_lag_weekly = int(lags["fred_weekly"])
    if str(config["data"].get("fred", {}).get("vintage_mode", "latest")) == "point_in_time":
        # ALFRED rows are already indexed by their first public availability date.
        fred_lag_daily = 0
        fred_lag_weekly = 0

    macro_daily_columns = [
        column
        for column in bundle.macro.columns
        if column not in {"financial_conditions", "fed_stress"}
    ]
    macro_weekly_columns = [
        column
        for column in bundle.macro.columns
        if column in {"financial_conditions", "fed_stress"}
    ]
    panels = [
        _available_panel(
            bundle.macro[macro_daily_columns] if macro_daily_columns else pd.DataFrame(),
            calendar,
            fred_lag_daily,
            "macro_",
            5,
        ),
        _available_panel(
            bundle.macro[macro_weekly_columns] if macro_weekly_columns else pd.DataFrame(),
            calendar,
            fred_lag_weekly,
            "macro_",
            10,
        ),
        _available_panel(bundle.vix, calendar, int(lags["cboe_daily"]), "", 3),
        # OFR/French freshness is governed by the observed last-successful date
        # in the manifest. Their latest-vintage history is never declared causal.
        _available_panel(bundle.ofr, calendar, 0, "", 5),
        _available_panel(bundle.french, calendar, 0, "ff_", 3),
    ]
    etf_close = _available_panel(
        bundle.etf_close,
        calendar,
        int(lags["price_proxy_daily"]),
        "px_",
        3,
    )
    panels.append(etf_close)
    source = pd.concat(panels, axis=1).sort_index()

    etf_returns = etf_close.pct_change(fill_method=None).rename(
        columns=lambda column: column.replace("px_", "ret_", 1)
    )
    source = pd.concat([source, etf_returns], axis=1)
    allow_french_market = bool(
        config["features"].get("use_revision_prone_french_history", False)
    )
    if "ret_spy" in source:
        market_return = source["ret_spy"]
    elif allow_french_market and "ff_mkt_rf" in source and "ff_rf" in source:
        french_market_return = source["ff_mkt_rf"] + source["ff_rf"]
        market_return = french_market_return
    else:
        raise ValueError(
            "SPY return is unavailable and revision-prone French market history is disabled"
        )

    feature_config = config["features"]
    market = pd.DataFrame(index=calendar)
    market["market_return"] = market_return
    for window in feature_config["volatility_windows"]:
        market[f"volatility_{window}"] = market_return.rolling(window).std() * np.sqrt(252)
    for window in feature_config["momentum_windows"]:
        market[f"momentum_{window}"] = _rolling_compound(market_return, int(window))
    wealth = (1.0 + market_return.fillna(0)).cumprod()
    market["drawdown_252"] = wealth / wealth.rolling(252, min_periods=60).max() - 1.0
    # Semi-deviation about zero. The earlier form masked positive days to NaN and
    # then demanded 30 surviving observations, so the feature only existed in the
    # ~19% of windows that happened to hold enough down days; it silently vanished
    # from the volatility risk block everywhere else.
    downside = market_return.clip(upper=0.0)
    market["downside_volatility_60"] = (
        downside.pow(2).rolling(60, min_periods=30).mean().pow(0.5) * np.sqrt(252)
    )
    market["return_skew_60"] = market_return.rolling(60, min_periods=30).skew()

    passthrough = [
        "vix_close",
        # Carried so decision_value can report a Sharpe excess of cash rather than
        # excess of zero. Not a regime input: see PREFERRED_MODEL_COLUMNS.
        "ff_rf",
        "macro_hy_oas",
        "macro_ig_oas",
        "macro_yield_curve_10y2y",
        "macro_yield_curve_10y3m",
        "macro_financial_conditions",
        "macro_fed_stress",
    ]
    passthrough.extend(column for column in source.columns if column.startswith("ofr_"))
    for column in dict.fromkeys(passthrough):
        if column in source:
            market[column] = source[column]
    if "vix_close" in market:
        market["vix_change_21"] = market["vix_close"].pct_change(21, fill_method=None)
    # Term structure, not levels. VIX above VIX3M is backwardation - the market
    # pricing near-term stress above longer-dated - and that is the classic
    # regime signal; the raw levels would just restate vix_close.
    if {"vix_close", "vix3m"}.issubset(source.columns):
        market["vix_term_slope"] = source["vix_close"] / source["vix3m"].replace(0, np.nan)
    if {"vix9d", "vix_close"}.issubset(source.columns):
        market["vix_short_slope"] = source["vix9d"] / source["vix_close"].replace(0, np.nan)
    for column in ("skew", "vvix"):
        if column in source:
            market[column] = source[column]
    if {"ret_hyg", "ret_lqd"}.issubset(source.columns):
        market["credit_risk_return_21"] = _rolling_compound(
            source["ret_hyg"] - source["ret_lqd"], 21
        )
    proxy_return_columns = [
        column
        for column in ("ret_spy", "ret_qqq", "ret_rsp", "ret_iwm", "ret_hyg", "ret_lqd")
        if column in source
    ]
    if proxy_return_columns:
        market["proxy_breadth"] = (source[proxy_return_columns] > 0).mean(axis=1)

    style = pd.DataFrame(index=calendar)
    if "ff_smb" in source:
        style["size"] = source["ff_smb"]
    if {"ret_iwm", "ret_spy"}.issubset(source.columns):
        style["size_etf"] = source["ret_iwm"] - source["ret_spy"]
    if "ff_hml" in source:
        style["value"] = source["ff_hml"]
    if {"ret_iwd", "ret_iwf"}.issubset(source.columns):
        style["value_etf"] = source["ret_iwd"] - source["ret_iwf"]
    if "ff_rmw" in source:
        style["quality"] = source["ff_rmw"]
    if {"ret_qual", "ret_spy"}.issubset(source.columns):
        style["quality_etf"] = source["ret_qual"] - source["ret_spy"]
    if "ff_cma" in source:
        style["conservative_investment"] = source["ff_cma"]
    if "ff_momentum" in source:
        style["momentum"] = source["ff_momentum"]
    if {"ret_mtum", "ret_spy"}.issubset(source.columns):
        style["momentum_etf"] = source["ret_mtum"] - source["ret_spy"]
    reversal_columns = [column for column in ("ff_short_reversal", "ff_long_reversal") if column in source]
    if reversal_columns:
        style["reversal"] = source[reversal_columns].mean(axis=1)
    if {"ret_usmv", "ret_spy"}.issubset(source.columns):
        style["defensive_etf"] = source["ret_usmv"] - source["ret_spy"]

    return FeatureSet(market=market, style_returns=style, source_panel=source)

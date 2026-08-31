from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score


def _future_variance(returns: pd.Series, horizon: int) -> pd.Series:
    return returns.pow(2).shift(-1).rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))


def _qlike(realized: pd.Series, forecast: pd.Series) -> float:
    variance = realized.clip(lower=1e-10)
    prediction = forecast.clip(lower=1e-10)
    return float((np.log(prediction) + variance / prediction).mean())


def evaluate_news_forward(
    news_daily: pd.DataFrame,
    market_returns: pd.Series,
    horizons: tuple[int, ...] = (5, 20),
    minimum_train: int = 126,
) -> pd.DataFrame:
    """Evaluate news as a causal challenger; future outcomes are used only for scoring."""
    signals = [
        column
        for column in (
            "news_stress",
            "news_uncertainty",
            "news_transition_alert",
            "monetary_hawkishness",
            "growth_slowdown",
            "credit_stress",
            "liquidity_stress",
            "geopolitical_supply_risk",
        )
        if column in news_daily
    ]
    if len(news_daily) < minimum_train or not signals:
        return pd.DataFrame()
    returns = market_returns.dropna().sort_index()
    panel = news_daily[signals].join(returns.rename("return"), how="inner").sort_index()
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        panel["target"] = _future_variance(returns, horizon).reindex(panel.index)
        panel["baseline"] = (
            returns.pow(2).rolling(60, min_periods=40).mean().mul(horizon).reindex(panel.index)
        )
        predictions = pd.Series(index=panel.index, dtype=float)
        feature_columns = ["baseline", *signals]
        for position in range(minimum_train, len(panel)):
            # A target ending within the last horizon days was not yet observable.
            train = panel.iloc[: max(0, position - horizon)].dropna(
                subset=[*feature_columns, "target"]
            )
            current = panel.iloc[[position]].dropna(subset=feature_columns)
            if len(train) < minimum_train or current.empty:
                continue
            model = Ridge(alpha=1.0)
            model.fit(train[feature_columns], np.log(train["target"].clip(lower=1e-10)))
            predictions.iloc[position] = float(np.exp(model.predict(current[feature_columns])[0]))
        valid = panel["target"].notna() & panel["baseline"].notna() & predictions.notna()
        if not valid.any():
            continue
        threshold = panel["target"].rolling(756, min_periods=126).quantile(0.90).shift(horizon)
        stress_label = panel["target"].gt(threshold)
        alert_valid = valid & panel["news_transition_alert"].notna() & threshold.notna()
        rows.append(
            {
                "horizon_days": horizon,
                "observations": int(valid.sum()),
                "baseline_qlike": _qlike(panel.loc[valid, "target"], panel.loc[valid, "baseline"]),
                "news_augmented_qlike": _qlike(panel.loc[valid, "target"], predictions.loc[valid]),
                "qlike_improvement": _qlike(panel.loc[valid, "target"], panel.loc[valid, "baseline"])
                - _qlike(panel.loc[valid, "target"], predictions.loc[valid]),
                "stress_event_auprc": (
                    float(
                        average_precision_score(
                            stress_label.loc[alert_valid],
                            panel.loc[alert_valid, "news_transition_alert"],
                        )
                    )
                    if stress_label.loc[alert_valid].nunique() > 1
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)

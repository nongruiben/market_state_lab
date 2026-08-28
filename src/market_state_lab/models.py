from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Some restricted Windows environments cannot query physical cores via WMIC.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


STATE_NAMES = ("calm", "transition", "stress")


@dataclass
class MarketStateResult:
    history: pd.DataFrame
    latest: dict[str, Any]
    feature_columns: list[str]


@dataclass
class StyleResult:
    history: pd.DataFrame
    latest: pd.DataFrame


def _trailing_z(series: pd.Series, window: int) -> pd.Series:
    minimum = max(60, window // 4)
    mean = series.rolling(window, min_periods=minimum).mean()
    std = series.rolling(window, min_periods=minimum).std().replace(0, np.nan)
    return ((series - mean) / std).clip(-4, 4)


def _sigmoid(value: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -20, 20)))


def _risk_score(features: pd.DataFrame, window: int) -> pd.Series:
    candidates: list[pd.Series] = []
    positive = [
        "volatility_20",
        "volatility_60",
        "downside_volatility_60",
        "vix_close",
        "vix_change_21",
        "macro_hy_oas",
        "macro_ig_oas",
        "macro_financial_conditions",
        "macro_fed_stress",
    ]
    positive.extend(column for column in features if column.startswith("ofr_") and "fsi" in column)
    negative = ["momentum_63", "momentum_252", "drawdown_252", "credit_risk_return_21", "proxy_breadth"]
    for column in dict.fromkeys(positive):
        if column in features:
            candidates.append(_trailing_z(features[column], window).rename(column))
    for column in negative:
        if column in features:
            candidates.append((-_trailing_z(features[column], window)).rename(column))
    if not candidates:
        raise ValueError("No risk features are available")
    return pd.concat(candidates, axis=1).mean(axis=1, skipna=True).rename("risk_score")


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    minimum = max(252, window // 3)

    def last_rank(values: np.ndarray) -> float:
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            return np.nan
        return float((valid <= valid[-1]).mean())

    return series.rolling(window, min_periods=minimum).apply(last_rank, raw=True)


def _baseline_probabilities(percentile: pd.Series) -> pd.DataFrame:
    calm = _sigmoid((0.38 - percentile) / 0.10)
    stress = _sigmoid((percentile - 0.62) / 0.10)
    transition = np.maximum(0.05, 1.0 - np.maximum(calm, stress))
    probabilities = pd.DataFrame(
        {"p_calm": calm, "p_transition": transition, "p_stress": stress},
        index=percentile.index,
    )
    return probabilities.div(probabilities.sum(axis=1), axis=0)


def _model_columns(features: pd.DataFrame, minimum_train: int) -> list[str]:
    preferred = [
        "volatility_20",
        "volatility_60",
        "downside_volatility_60",
        "momentum_63",
        "momentum_252",
        "drawdown_252",
        "vix_close",
        "macro_hy_oas",
        "macro_ig_oas",
        "macro_yield_curve_10y2y",
        "macro_financial_conditions",
        "credit_risk_return_21",
        "proxy_breadth",
    ]
    preferred.extend(column for column in features if column.startswith("ofr_") and "fsi" in column)
    initial_window = features.iloc[: min(len(features), minimum_train)]
    threshold = max(120, minimum_train // 2)
    return [
        column
        for column in dict.fromkeys(preferred)
        if column in features and initial_window[column].count() >= threshold
    ]


def fit_market_state(features: pd.DataFrame, config: dict[str, Any]) -> MarketStateResult:
    settings = config["models"]["market_state"]
    normalization_window = int(config["features"]["rolling_normalization_window"])
    minimum_train = int(settings["minimum_train_observations"])
    refit_every = int(settings["refit_every_observations"])
    gmm_weight = float(settings["gmm_weight"])
    random_seed = int(config["project"]["random_seed"])

    risk_score = _risk_score(features, normalization_window)
    risk_percentile = _rolling_percentile(risk_score, normalization_window)
    baseline = _baseline_probabilities(risk_percentile)
    columns = _model_columns(features, minimum_train)
    if len(columns) < 3:
        raise ValueError(f"Only {len(columns)} usable regime features; at least 3 are required")

    values = features[columns].copy()
    gmm_probabilities = pd.DataFrame(index=features.index, columns=[f"p_{name}" for name in STATE_NAMES], dtype=float)
    valid_positions = np.flatnonzero(values.notna().sum(axis=1).to_numpy() >= max(2, len(columns) // 3))
    if len(valid_positions) <= minimum_train:
        raise ValueError(f"Only {len(valid_positions)} usable observations; need more than {minimum_train}")

    for start_offset in range(minimum_train, len(valid_positions), refit_every):
        train_positions = valid_positions[:start_offset]
        test_positions = valid_positions[start_offset : start_offset + refit_every]
        if len(test_positions) == 0:
            break
        train = values.iloc[train_positions]
        test = values.iloc[test_positions]
        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler(quantile_range=(10, 90))
        train_array = scaler.fit_transform(imputer.fit_transform(train))
        test_array = scaler.transform(imputer.transform(test))
        model = GaussianMixture(
            n_components=int(settings["n_states"]),
            covariance_type="full",
            reg_covar=1e-5,
            n_init=int(settings.get("n_init", 3)),
            random_state=random_seed,
        )
        model.fit(train_array)
        train_labels = model.predict(train_array)
        train_risk = risk_score.iloc[train_positions].to_numpy()
        component_risk = {
            component: float(np.nanmean(train_risk[train_labels == component]))
            for component in range(model.n_components)
        }
        ordered_components = sorted(component_risk, key=component_risk.get)
        raw_probabilities = model.predict_proba(test_array)
        ordered = np.zeros_like(raw_probabilities)
        for state_index, component in enumerate(ordered_components):
            ordered[:, state_index] = raw_probabilities[:, component]
        gmm_probabilities.iloc[test_positions] = ordered

    combined = baseline.copy()
    available = gmm_probabilities.notna().all(axis=1)
    combined.loc[available] = (
        gmm_weight * gmm_probabilities.loc[available]
        + (1.0 - gmm_weight) * baseline.loc[available]
    )
    combined = combined.ewm(alpha=float(settings["persistence_alpha"]), adjust=False).mean()
    combined = combined.div(combined.sum(axis=1), axis=0)
    history = pd.concat([features, risk_score, risk_percentile.rename("risk_percentile"), combined], axis=1)
    probability_columns = ["p_calm", "p_transition", "p_stress"]
    complete = history[probability_columns].notna().all(axis=1)
    history["market_state"] = pd.Series(index=history.index, dtype="object")
    history.loc[complete, "market_state"] = (
        history.loc[complete, probability_columns]
        .idxmax(axis=1)
        .str.replace("p_", "", regex=False)
    )
    history["state_confidence"] = history[probability_columns].max(axis=1)
    latest_row = history.dropna(subset=probability_columns).iloc[-1]
    latest = {
        "as_of": latest_row.name.date().isoformat(),
        "market_state": latest_row["market_state"],
        "confidence": float(latest_row["state_confidence"]),
        "probabilities": {name: float(latest_row[f"p_{name}"]) for name in STATE_NAMES},
        "risk_score": float(latest_row["risk_score"]),
        "risk_percentile": float(latest_row["risk_percentile"]),
        "method": "walk-forward GMM plus observable-risk ensemble",
    }
    return MarketStateResult(history, latest, columns)


def _annualized_score(returns: pd.Series, window: int) -> pd.Series:
    mean = returns.rolling(window, min_periods=max(10, window // 2)).mean()
    volatility = returns.rolling(window, min_periods=max(10, window // 2)).std().replace(0, np.nan)
    return mean / volatility * np.sqrt(252)


def fit_style(style_returns: pd.DataFrame, config: dict[str, Any]) -> StyleResult:
    if style_returns.empty:
        raise ValueError("No style returns are available")
    windows = [int(value) for value in config["features"]["style_windows"]]
    weights = np.array([0.50, 0.30, 0.20][: len(windows)], dtype=float)
    weights = weights / weights.sum()
    normalization_window = int(config["features"]["rolling_normalization_window"])
    raw_scores = pd.DataFrame(index=style_returns.index)
    for column in style_returns:
        horizon_scores = [_annualized_score(style_returns[column], window) for window in windows]
        raw_scores[column] = sum(weight * score for weight, score in zip(weights, horizon_scores))

    def evidence(base: str) -> pd.Series:
        columns = [column for column in raw_scores if column == base or column.startswith(f"{base}_")]
        if not columns:
            return pd.Series(index=raw_scores.index, dtype=float)
        normalized = pd.concat(
            [_trailing_z(raw_scores[column], normalization_window) for column in columns], axis=1
        )
        return normalized.mean(axis=1, skipna=True)

    scores = pd.DataFrame(index=style_returns.index)
    for dimension in ("size", "value", "quality", "conservative_investment", "momentum", "reversal", "defensive"):
        score = evidence(dimension)
        if score.notna().any():
            scores[f"score_{dimension}"] = score

    temperature = float(config["models"]["style"]["score_temperature"])
    history = scores.copy()
    pairs = {
        "size": ("small", "large"),
        "value": ("value", "growth"),
        "quality": ("quality", "speculative"),
        "conservative_investment": ("conservative", "aggressive"),
        "momentum": ("momentum", "anti_momentum"),
        "reversal": ("reversal", "trend_persistence"),
        "defensive": ("defensive", "cyclical"),
    }
    latest_rows = []
    for dimension, (positive_name, negative_name) in pairs.items():
        score_column = f"score_{dimension}"
        if score_column not in scores:
            continue
        positive = _sigmoid(scores[score_column] / temperature)
        history[f"p_{positive_name}"] = positive
        history[f"p_{negative_name}"] = 1.0 - positive
        valid = history.dropna(subset=[score_column, f"p_{positive_name}"])
        if valid.empty:
            continue
        row = valid.iloc[-1]
        positive_probability = float(row[f"p_{positive_name}"])
        favored = positive_name if positive_probability >= 0.5 else negative_name
        as_of = valid.index[-1]
        latest_available = history.index.max()
        age_business_days = int(np.busday_count(as_of.date(), latest_available.date()))
        latest_rows.append(
            {
                "as_of": as_of.date().isoformat(),
                "age_business_days": age_business_days,
                "data_status": "fresh" if age_business_days <= 5 else "stale",
                "dimension": dimension,
                "favored_style": favored,
                "favored_probability": max(positive_probability, 1.0 - positive_probability),
                "score": float(row[score_column]),
                "positive_style": positive_name,
                "positive_probability": positive_probability,
                "negative_style": negative_name,
                "negative_probability": 1.0 - positive_probability,
            }
        )
    return StyleResult(history=history, latest=pd.DataFrame(latest_rows))

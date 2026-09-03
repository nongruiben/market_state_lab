from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

STATE_NAMES = ("low_risk", "mid_risk", "high_risk")
PROBABILITY_COLUMNS = [f"p_{name}" for name in STATE_NAMES]


@dataclass
class MarketStateResult:
    history: pd.DataFrame
    latest: dict[str, Any]
    feature_columns: list[str]
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    decision_value: pd.DataFrame = field(default_factory=pd.DataFrame)


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


def _block_score(
    features: pd.DataFrame,
    positive: list[str],
    negative: list[str],
    window: int,
) -> pd.Series:
    values: list[pd.Series] = []
    for column in positive:
        if column in features:
            values.append(_trailing_z(features[column], window))
    for column in negative:
        if column in features:
            values.append(-_trailing_z(features[column], window))
    if not values:
        return pd.Series(index=features.index, dtype=float)
    return pd.concat(values, axis=1).mean(axis=1, skipna=True)


def _risk_blocks(features: pd.DataFrame, window: int) -> pd.DataFrame:
    ofr = [column for column in features if column.startswith("ofr_") and "fsi" in column]
    blocks = {
        "risk_volatility": _block_score(
            features,
            ["volatility_20", "volatility_60", "downside_volatility_60", "vix_close", "vix_change_21"],
            [],
            window,
        ),
        "risk_credit": _block_score(
            features,
            ["macro_hy_oas", "macro_ig_oas"],
            ["credit_risk_return_21"],
            window,
        ),
        "risk_financial_conditions": _block_score(
            features,
            ["macro_financial_conditions", "macro_fed_stress", *ofr],
            [],
            window,
        ),
        "risk_trend_breadth": _block_score(
            features,
            [],
            ["momentum_63", "momentum_252", "drawdown_252", "proxy_breadth"],
            window,
        ),
        "risk_rates": _block_score(
            features,
            [],
            ["macro_yield_curve_10y2y", "macro_yield_curve_10y3m"],
            window,
        ),
    }
    return pd.DataFrame(blocks, index=features.index)


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    minimum = max(126, window // 3)

    def last_rank(values: np.ndarray) -> float:
        valid = values[np.isfinite(values)]
        if len(valid) == 0 or not np.isfinite(values[-1]):
            return np.nan
        return float((valid <= values[-1]).mean())

    return series.rolling(window, min_periods=minimum).apply(last_rank, raw=True)


def _baseline_probabilities(percentile: pd.Series) -> pd.DataFrame:
    low = _sigmoid((0.38 - percentile) / 0.10)
    high = _sigmoid((percentile - 0.62) / 0.10)
    mid = np.maximum(0.05, 1.0 - np.maximum(low, high))
    probabilities = pd.DataFrame(
        {"p_low_risk": low, "p_mid_risk": mid, "p_high_risk": high},
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


def _soft_component_order(
    responsibilities: np.ndarray,
    risk: np.ndarray,
    minimum_effective: float,
) -> tuple[list[int] | None, np.ndarray]:
    effective = responsibilities.sum(axis=0)
    weighted_risk = np.full(responsibilities.shape[1], np.nan)
    valid_risk = np.isfinite(risk)
    for component in range(responsibilities.shape[1]):
        weights = responsibilities[valid_risk, component]
        denominator = weights.sum()
        if denominator > 0:
            weighted_risk[component] = float(np.dot(weights, risk[valid_risk]) / denominator)
    valid = (effective >= minimum_effective) & np.isfinite(weighted_risk)
    if valid.sum() != responsibilities.shape[1]:
        return None, effective
    return list(np.argsort(weighted_risk)), effective


def _forward_filter(model: GaussianHMM, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_emission = model._compute_log_likelihood(values)
    log_transition = np.log(np.clip(model.transmat_, 1e-300, None))
    log_filtered = np.empty_like(log_emission)
    switch = np.full(len(values), np.nan)
    current = np.log(np.clip(model.startprob_, 1e-300, None)) + log_emission[0]
    current -= logsumexp(current)
    log_filtered[0] = current
    for index in range(1, len(values)):
        joint = current[:, None] + log_transition + log_emission[index][None, :]
        normalizer = logsumexp(joint)
        switch[index] = 1.0 - float(np.exp(np.diag(joint) - normalizer).sum())
        current = logsumexp(joint, axis=0)
        current -= logsumexp(current)
        log_filtered[index] = current
    return np.exp(log_filtered), switch


def _fit_walk_forward_models(
    values: pd.DataFrame,
    risk_score: pd.Series,
    settings: dict[str, Any],
    random_seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.Series, pd.DataFrame]:
    outputs = {
        "gmm": pd.DataFrame(index=values.index, columns=PROBABILITY_COLUMNS, dtype=float),
        "hmm": pd.DataFrame(index=values.index, columns=PROBABILITY_COLUMNS, dtype=float),
    }
    switch_probability = pd.Series(index=values.index, dtype=float, name="switch_probability")
    diagnostics: list[dict[str, Any]] = []
    minimum_train = int(settings["minimum_train_observations"])
    maximum_train = int(settings.get("maximum_train_observations", 0))
    refit_every = int(settings["refit_every_observations"])
    minimum_effective = float(settings.get("minimum_component_effective_observations", 25))
    valid_positions = np.flatnonzero(
        values.notna().sum(axis=1).to_numpy() >= max(2, len(values.columns) // 3)
    )
    if len(valid_positions) <= minimum_train:
        raise ValueError(f"Only {len(valid_positions)} usable observations; need more than {minimum_train}")

    for start_offset in range(minimum_train, len(valid_positions), refit_every):
        train_positions = valid_positions[:start_offset]
        if maximum_train > 0:
            train_positions = train_positions[-maximum_train:]
        test_positions = valid_positions[start_offset : start_offset + refit_every]
        if len(test_positions) == 0:
            break
        train = values.iloc[train_positions]
        test = values.iloc[test_positions]
        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler(quantile_range=(10, 90))
        train_array = scaler.fit_transform(imputer.fit_transform(train))
        test_array = scaler.transform(imputer.transform(test))
        train_risk = risk_score.iloc[train_positions].to_numpy()
        common = {
            "train_end": str(train.index[-1].date()),
            "test_start": str(test.index[0].date()),
            "test_end": str(test.index[-1].date()),
            "train_rows": len(train),
            "test_rows": len(test),
        }

        gmm_initializer: GaussianMixture | None = None
        try:
            gmm = GaussianMixture(
                n_components=int(settings["n_states"]),
                covariance_type=str(settings.get("covariance_type", "diag")),
                reg_covar=float(settings.get("reg_covar", 1e-4)),
                n_init=int(settings.get("n_init", 3)),
                random_state=random_seed,
            )
            gmm.fit(train_array)
            gmm_initializer = gmm
            order, effective = _soft_component_order(
                gmm.predict_proba(train_array), train_risk, minimum_effective
            )
            if order is None:
                raise ValueError(f"underfilled component effective counts={effective.round(1).tolist()}")
            outputs["gmm"].iloc[test_positions] = gmm.predict_proba(test_array)[:, order]
            diagnostics.append({**common, "model": "gmm", "status": "ok", "detail": ""})
        except Exception as exc:
            diagnostics.append({**common, "model": "gmm", "status": "skipped", "detail": str(exc)})

        if bool(settings.get("hmm", {}).get("enabled", True)):
            try:
                n_states = int(settings["n_states"])
                transition_prior = float(settings.get("hmm", {}).get("transition_prior", 12.0))
                prior = np.ones((n_states, n_states))
                prior[np.diag_indices(n_states)] = transition_prior
                hmm = GaussianHMM(
                    n_components=n_states,
                    covariance_type=str(settings.get("covariance_type", "diag")),
                    min_covar=float(settings.get("reg_covar", 1e-4)),
                    n_iter=int(settings.get("hmm", {}).get("maximum_iterations", 75)),
                    tol=1e-3,
                    transmat_prior=prior,
                    random_state=random_seed,
                )
                if gmm_initializer is not None:
                    self_probability = 0.96
                    hmm.init_params = ""
                    hmm.startprob_ = np.clip(gmm_initializer.weights_, 1e-6, None)
                    hmm.startprob_ /= hmm.startprob_.sum()
                    hmm.transmat_ = np.full(
                        (n_states, n_states), (1.0 - self_probability) / (n_states - 1)
                    )
                    np.fill_diagonal(hmm.transmat_, self_probability)
                    hmm.means_ = gmm_initializer.means_.copy()
                    hmm.covars_ = np.maximum(
                        gmm_initializer.covariances_, float(settings.get("reg_covar", 1e-4))
                    )
                hmm_logger = logging.getLogger("hmmlearn.base")
                previous_level = hmm_logger.level
                try:
                    hmm_logger.setLevel(logging.ERROR)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", category=RuntimeWarning, module=r"hmmlearn\..*")
                        hmm.fit(train_array)
                finally:
                    hmm_logger.setLevel(previous_level)
                history = list(hmm.monitor_.history)
                if len(history) >= 2 and history[-1] - history[-2] < -0.01:
                    raise ValueError(f"HMM likelihood decreased by {history[-1] - history[-2]:.6f}")
                if not all(
                    np.isfinite(parameter).all()
                    for parameter in (hmm.startprob_, hmm.transmat_, hmm.means_, hmm.covars_)
                ):
                    raise ValueError("HMM produced non-finite parameters")
                filtered, switches = _forward_filter(hmm, np.vstack([train_array, test_array]))
                order, effective = _soft_component_order(
                    filtered[: len(train_array)], train_risk, minimum_effective
                )
                if order is None:
                    raise ValueError(f"underfilled state effective counts={effective.round(1).tolist()}")
                outputs["hmm"].iloc[test_positions] = filtered[len(train_array) :, order]
                switch_probability.iloc[test_positions] = switches[len(train_array) :]
                diagnostics.append({**common, "model": "hmm", "status": "ok", "detail": ""})
            except Exception as exc:
                diagnostics.append({**common, "model": "hmm", "status": "skipped", "detail": str(exc)})

    return outputs, switch_probability, pd.DataFrame(diagnostics)


def _observable_target(percentile: pd.Series) -> pd.DataFrame:
    """Self-consistency target.

    This is a deterministic re-encoding of ``risk_percentile`` at the same 0.38 /
    0.62 knots the baseline sigmoid uses, so scoring the baseline against it is an
    identity, not evidence. It is kept because the dynamic ensemble weights are
    defined on it, but it must never be reported as accuracy - use
    ``_forward_volatility_target`` for that.
    """
    target = pd.DataFrame(np.nan, index=percentile.index, columns=PROBABILITY_COLUMNS)
    valid = percentile.notna()
    target.loc[valid, :] = 0.0
    target.loc[valid & percentile.lt(0.38), "p_low_risk"] = 1.0
    target.loc[valid & percentile.between(0.38, 0.62, inclusive="both"), "p_mid_risk"] = 1.0
    target.loc[valid & percentile.gt(0.62), "p_high_risk"] = 1.0
    return target


def _forward_volatility_target(
    market_return: pd.Series,
    horizon: int,
    window: int,
    minimum: int,
) -> pd.DataFrame:
    """External scoring target: the tercile of realised volatility over the NEXT
    ``horizon`` sessions.

    The outcome is future by construction - that is the point, and it is used for
    scoring only, never as a model input. The tercile thresholds, however, are
    built solely from outcomes that had already fully realised by ``t``, so the
    labelling itself carries no lookahead.
    """
    forward = (
        market_return.shift(-1).rolling(horizon, min_periods=horizon).std().shift(-(horizon - 1))
        * np.sqrt(252)
    )
    settled = forward.shift(horizon)
    low = settled.rolling(window, min_periods=minimum).quantile(1.0 / 3.0)
    high = settled.rolling(window, min_periods=minimum).quantile(2.0 / 3.0)
    target = pd.DataFrame(np.nan, index=market_return.index, columns=PROBABILITY_COLUMNS)
    valid = forward.notna() & low.notna() & high.notna()
    target.loc[valid, :] = 0.0
    target.loc[valid & forward.le(low), "p_low_risk"] = 1.0
    target.loc[valid & forward.gt(low) & forward.le(high), "p_mid_risk"] = 1.0
    target.loc[valid & forward.gt(high), "p_high_risk"] = 1.0
    return target


def _climatology(target: pd.DataFrame, horizon: int, window: int, minimum: int) -> pd.DataFrame:
    """No-skill reference forecast: the trailing base rate of the target.

    Only outcomes settled by ``t`` contribute, so this is a forecast a person could
    actually have made. Any model that cannot beat it has no demonstrated skill.
    """
    settled = target.shift(horizon)
    rate = settled.rolling(window, min_periods=minimum).mean()
    total = rate.sum(axis=1).replace(0.0, np.nan)
    return rate.div(total, axis=0)


def _apply_temperature(log_probabilities: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    scaled = log_probabilities / temperature[:, None]
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _best_temperature(
    log_probabilities: np.ndarray,
    target: np.ndarray,
    grid: np.ndarray,
) -> float:
    best_temperature = 1.0
    best_loss = np.inf
    for candidate in grid:
        probabilities = _apply_temperature(
            log_probabilities, np.full(len(log_probabilities), float(candidate))
        )
        loss = float(((probabilities - target) ** 2).sum(axis=1).mean())
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(candidate)
    return best_temperature


def _walk_forward_temperature(
    probabilities: pd.DataFrame,
    target: pd.DataFrame,
    horizon: int,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """Temperature scaling refit walk-forward against the external target.

    The diagnosis is overconfidence, not misinformation: the raw states beat the
    climatology hit rate while losing to it on Brier. One scalar per refit is the
    least overfit-prone correction for exactly that, and it cannot reorder the
    states - only soften or sharpen them.
    """
    window = int(settings.get("window", 1260))
    minimum = int(settings.get("min_observations", 252))
    refit_every = int(settings.get("refit_every_observations", 126))
    grid = np.geomspace(0.5, 12.0, 28)

    values = probabilities.to_numpy(dtype=float)
    outcomes = target.to_numpy(dtype=float)
    usable = np.isfinite(values).all(axis=1) & np.isfinite(outcomes).all(axis=1)
    log_probabilities = np.full_like(values, np.nan)
    finite = np.isfinite(values).all(axis=1)
    log_probabilities[finite] = np.log(np.clip(values[finite], 1e-12, None))

    total = len(probabilities)
    temperature = np.ones(total)
    current = 1.0
    for start in range(0, total, refit_every):
        # A target dated d is only settled at d + horizon, so nothing inside the
        # trailing horizon may inform the temperature used from `start` onward.
        cutoff = start - horizon
        if cutoff > 0:
            lower = max(0, cutoff - window)
            rows = np.flatnonzero(usable[lower:cutoff]) + lower
            if len(rows) >= minimum:
                current = _best_temperature(log_probabilities[rows], outcomes[rows], grid)
        temperature[start : start + refit_every] = current

    calibrated = pd.DataFrame(np.nan, index=probabilities.index, columns=PROBABILITY_COLUMNS)
    if finite.any():
        calibrated.loc[finite, :] = _apply_temperature(
            log_probabilities[finite], temperature[finite]
        )
    return calibrated, pd.Series(temperature, index=probabilities.index, name="calibration_temperature")


def _dynamic_ensemble(
    probabilities: dict[str, pd.DataFrame],
    percentile: pd.Series,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configured = {name: float(value) for name, value in settings["model_weights"].items()}
    window = int(settings.get("dynamic_weight_window", 252))
    minimum = int(settings.get("dynamic_weight_min_observations", 126))
    baseline_floor = float(settings.get("baseline_weight_floor", 0.4))
    target = _observable_target(percentile)
    losses: dict[str, pd.Series] = {}
    for name, frame in probabilities.items():
        daily = ((frame - target) ** 2).sum(axis=1)
        losses[name] = daily.shift(1).rolling(window, min_periods=minimum).mean()

    weights = pd.DataFrame(0.0, index=percentile.index, columns=list(probabilities))
    ensemble = pd.DataFrame(np.nan, index=percentile.index, columns=PROBABILITY_COLUMNS)
    for date in percentile.index:
        available = {name: frame.loc[date].notna().all() for name, frame in probabilities.items()}
        raw: dict[str, float] = {}
        for name in probabilities:
            if not available[name]:
                raw[name] = 0.0
                continue
            loss = losses[name].loc[date]
            multiplier = np.exp(-3.0 * float(loss)) if np.isfinite(loss) else 1.0
            raw[name] = configured.get(name, 0.0) * multiplier
        total = sum(raw.values())
        if total <= 0:
            continue
        normalized = {name: value / total for name, value in raw.items()}
        if available.get("baseline", False) and normalized["baseline"] < baseline_floor:
            remaining = 1.0 - baseline_floor
            other_total = sum(value for name, value in normalized.items() if name != "baseline")
            normalized["baseline"] = baseline_floor
            for name in normalized:
                if name != "baseline":
                    normalized[name] = remaining * normalized[name] / other_total if other_total else 0.0
        weights.loc[date] = normalized
        combined = sum(
            normalized[name] * probabilities[name].loc[date]
            for name in probabilities
            if normalized[name] > 0
        )
        ensemble.loc[date] = combined / combined.sum()
    return ensemble, weights.add_prefix("weight_")


def _score_against(frame: pd.DataFrame, target: pd.DataFrame) -> tuple[float, float, int]:
    valid = frame.notna().all(axis=1) & target.notna().all(axis=1)
    if not valid.any():
        return np.nan, np.nan, 0
    brier = float(((frame.loc[valid] - target.loc[valid]) ** 2).sum(axis=1).mean())
    hit = float((frame.loc[valid].idxmax(axis=1) == target.loc[valid].idxmax(axis=1)).mean())
    return brier, hit, int(valid.sum())


def _model_comparison(
    probabilities: dict[str, pd.DataFrame],
    self_target: pd.DataFrame,
    forward_target: pd.DataFrame,
    climatology: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Report both metrics side by side, with the no-skill row always present.

    ``self_consistency_brier`` says whether a model reproduces the hand-written
    percentile rule; ``forward_brier`` says whether it forecasts anything. They are
    not interchangeable and the first one is roughly six times more flattering, so
    neither is ever published alone.
    """
    scored = dict(probabilities)
    if climatology is not None:
        scored["climatology"] = climatology
    rows: list[dict[str, Any]] = []
    for name, frame in scored.items():
        valid = frame.notna().all(axis=1) & self_target.notna().all(axis=1)
        self_brier, _, self_observations = _score_against(frame, self_target)
        forward_brier, forward_hit, forward_observations = _score_against(frame, forward_target)
        if self_observations == 0 and forward_observations == 0:
            continue
        # The learned models only start after minimum_train, so each model covers a
        # different span. Comparing raw Brier across those spans would compare
        # different markets, not different models: the skill delta is therefore
        # always measured on the days where this model and climatology both spoke.
        paired_brier = np.nan
        paired_reference = np.nan
        paired_observations = 0
        if climatology is not None:
            paired = (
                frame.notna().all(axis=1)
                & climatology.notna().all(axis=1)
                & forward_target.notna().all(axis=1)
            )
            if paired.any():
                paired_observations = int(paired.sum())
                paired_brier = float(
                    ((frame.loc[paired] - forward_target.loc[paired]) ** 2).sum(axis=1).mean()
                )
                paired_reference = float(
                    ((climatology.loc[paired] - forward_target.loc[paired]) ** 2)
                    .sum(axis=1)
                    .mean()
                )
        labelled = frame.dropna(how="any")
        if labelled.empty:
            flip_rate = np.nan
            mean_duration = np.nan
        else:
            labels = labelled.idxmax(axis=1)
            changes = labels.ne(labels.shift()).iloc[1:]
            durations = labels.groupby(labels.ne(labels.shift()).cumsum()).size()
            flip_rate = float(changes.mean()) if len(changes) else np.nan
            mean_duration = float(durations.mean())
        rows.append(
            {
                "model": name,
                "observations": int(valid.sum()),
                "self_consistency_brier": self_brier,
                "forward_brier": forward_brier,
                "forward_hit_rate": forward_hit,
                "forward_observations": forward_observations,
                "paired_observations": paired_observations,
                "paired_forward_brier": paired_brier,
                "paired_climatology_brier": paired_reference,
                "forward_brier_vs_climatology": paired_brier - paired_reference,
                "beats_climatology": bool(paired_brier < paired_reference)
                if np.isfinite(paired_brier) and np.isfinite(paired_reference)
                else False,
                "flip_rate": flip_rate,
                "mean_duration_days": mean_duration,
            }
        )
    comparison = pd.DataFrame(rows)
    if not comparison.empty and "model" in comparison:
        comparison.loc[comparison["model"].eq("climatology"), "beats_climatology"] = False
    return comparison


def _decision_value(
    features: pd.DataFrame,
    probabilities: dict[str, pd.DataFrame],
    target_volatility: float,
) -> pd.DataFrame:
    if "market_return" not in features or "volatility_20" not in features:
        return pd.DataFrame()
    vol_exposure = (target_volatility / features["volatility_20"]).clip(0.0, 1.5)
    common = features[["market_return", "volatility_20"]].notna().all(axis=1)
    for frame in probabilities.values():
        common &= frame.notna().all(axis=1)
    rows: list[dict[str, Any]] = []
    candidates: dict[str, pd.Series] = {"vol_only": vol_exposure}
    for name, frame in probabilities.items():
        candidates[name] = vol_exposure * (1.0 - 0.45 * frame["p_high_risk"])
    applied_candidates = {name: exposure.shift(1) for name, exposure in candidates.items()}
    for applied in applied_candidates.values():
        common &= applied.notna()
    for name, applied in applied_candidates.items():
        strategy_return = (applied * features["market_return"]).loc[common]
        if len(strategy_return) < 126:
            continue
        wealth = (1.0 + strategy_return).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        rolling_vol = strategy_return.rolling(20, min_periods=20).std() * np.sqrt(252)
        annual_vol = float(strategy_return.std() * np.sqrt(252))
        annual_return = float((wealth.iloc[-1] ** (252.0 / len(strategy_return))) - 1.0)
        rows.append(
            {
                "method": name,
                "observations": len(strategy_return),
                "annualized_return": annual_return,
                "annualized_volatility": annual_vol,
                "volatility_target_mae": float((rolling_vol - target_volatility).abs().mean()),
                "maximum_drawdown": float(drawdown.min()),
                "sharpe_zero_rate": annual_return / annual_vol if annual_vol > 0 else np.nan,
                "daily_turnover": float(applied.loc[common].diff().abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def fit_market_state(features: pd.DataFrame, config: dict[str, Any]) -> MarketStateResult:
    settings = config["models"]["market_state"]
    normalization_window = int(config["features"]["rolling_normalization_window"])
    minimum_train = int(settings["minimum_train_observations"])
    random_seed = int(config["project"]["random_seed"])

    blocks = _risk_blocks(features, normalization_window)
    # A mean over "whatever blocks happen to exist today" is not one quantity: it
    # was a 4-block mean before HYG history begins and a 5-block mean after, and
    # _rolling_percentile then ranked today's value against a window built from the
    # other composition. Fix the composition instead, and let a missing required
    # block produce NaN rather than a quietly different score.
    configured_blocks = config["features"].get("required_risk_blocks") or list(blocks.columns)
    required_blocks = [str(name) for name in configured_blocks]
    unknown = [name for name in required_blocks if name not in blocks.columns]
    if unknown:
        raise ValueError(f"Unknown required risk blocks: {unknown}")
    required_present = blocks[required_blocks].notna()
    risk_score = blocks[required_blocks].mean(axis=1).where(required_present.all(axis=1))
    risk_score.name = "risk_score"
    risk_score_block_count = required_present.sum(axis=1).rename("risk_score_block_count")
    risk_blocks_available = blocks.notna().sum(axis=1).rename("risk_blocks_available")
    risk_percentile = _rolling_percentile(risk_score, normalization_window)
    baseline = _baseline_probabilities(risk_percentile)
    columns = _model_columns(features, minimum_train)
    if len(columns) < 3:
        raise ValueError(f"Only {len(columns)} usable regime features; at least 3 are required")

    learned, switch_probability, diagnostics = _fit_walk_forward_models(
        features[columns].copy(), risk_score, settings, random_seed
    )
    probabilities = {"baseline": baseline, **learned}
    ensemble, model_weights = _dynamic_ensemble(probabilities, risk_percentile, settings)

    evaluation = settings.get("evaluation", {}) or {}
    forward_horizon = int(evaluation.get("forward_horizon_days", 20))
    forward_window = int(evaluation.get("forward_window", 756))
    forward_minimum = int(evaluation.get("forward_min_observations", 252))
    forward_target = _forward_volatility_target(
        features["market_return"], forward_horizon, forward_window, forward_minimum
    )
    climatology = _climatology(forward_target, forward_horizon, forward_window, forward_minimum)

    calibration = settings.get("calibration", {}) or {}
    if bool(calibration.get("enabled", True)):
        calibrated, temperature = _walk_forward_temperature(
            ensemble, forward_target, forward_horizon, calibration
        )
    else:
        calibrated = ensemble.copy()
        temperature = pd.Series(
            1.0, index=ensemble.index, name="calibration_temperature", dtype=float
        )
    apply_to_decision = bool(calibration.get("apply_to_decision", True))

    half_life = float(settings.get("decision_half_life_days", 15))
    decision_source = calibrated if apply_to_decision else ensemble
    decision = decision_source.ewm(halflife=half_life, adjust=False).mean()
    decision = decision.div(decision.sum(axis=1), axis=0)

    history = pd.concat(
        [
            features,
            blocks,
            risk_score,
            risk_score_block_count,
            risk_blocks_available,
            risk_percentile.rename("risk_percentile"),
            baseline.add_prefix("baseline_"),
            learned["gmm"].add_prefix("gmm_"),
            learned["hmm"].add_prefix("hmm_"),
            model_weights,
            ensemble,
            calibrated.add_prefix("calibrated_"),
            temperature,
            decision.add_prefix("decision_"),
            switch_probability,
        ],
        axis=1,
    )
    complete = history[PROBABILITY_COLUMNS].notna().all(axis=1)
    history["market_state"] = pd.Series(index=history.index, dtype="object")
    history.loc[complete, "market_state"] = (
        history.loc[complete, PROBABILITY_COLUMNS].idxmax(axis=1).str.replace("p_", "", regex=False)
    )
    history["state_confidence"] = history[PROBABILITY_COLUMNS].max(axis=1)
    entropy = -(history[PROBABILITY_COLUMNS] * np.log(history[PROBABILITY_COLUMNS].clip(lower=1e-12))).sum(axis=1)
    history["state_entropy"] = entropy / np.log(len(STATE_NAMES))
    history["state_change_score"] = 1.0 - (
        history[PROBABILITY_COLUMNS] * history[PROBABILITY_COLUMNS].shift(1)
    ).sum(axis=1)

    valid_history = history.dropna(subset=PROBABILITY_COLUMNS)
    if valid_history.empty:
        raise ValueError("No walk-forward state probabilities were produced")
    latest_row = valid_history.iloc[-1]
    latest_date = valid_history.index[-1]
    weight_values = {
        name.replace("weight_", ""): float(latest_row[name])
        for name in model_weights.columns
        if np.isfinite(latest_row[name])
    }
    latest = {
        "as_of": latest_date.date().isoformat(),
        "market_state": str(latest_row["market_state"]),
        "confidence": float(latest_row["state_confidence"]),
        "probabilities": {name: float(latest_row[f"p_{name}"]) for name in STATE_NAMES},
        "calibrated_probabilities": {
            name: float(latest_row[f"calibrated_p_{name}"]) for name in STATE_NAMES
        },
        "calibration_temperature": float(latest_row["calibration_temperature"]),
        "decision_weights": {name: float(latest_row[f"decision_p_{name}"]) for name in STATE_NAMES},
        "decision_weight_source": "calibrated" if apply_to_decision else "ensemble",
        "decision_half_life_days": half_life,
        "risk_score": float(latest_row["risk_score"]),
        "risk_score_block_count": int(latest_row["risk_score_block_count"]),
        "required_risk_blocks": required_blocks,
        "risk_percentile": float(latest_row["risk_percentile"]),
        "switch_probability": (
            float(latest_row["switch_probability"])
            if np.isfinite(latest_row["switch_probability"])
            else None
        ),
        "state_entropy": float(latest_row["state_entropy"]),
        "model_weights": weight_values,
        "method": "causal baseline + diagonal GMM + forward-filtered diagonal HMM",
    }
    self_target = _observable_target(risk_percentile)
    evaluated_probabilities = {
        **probabilities,
        "ensemble": ensemble,
        "ensemble_calibrated": calibrated,
    }
    comparison = _model_comparison(
        evaluated_probabilities, self_target, forward_target, climatology
    )
    if not comparison.empty and "beats_climatology" in comparison:
        headline = comparison.loc[comparison["model"].eq("ensemble_calibrated")]
        reference = comparison.loc[comparison["model"].eq("climatology")]
        if not headline.empty and not reference.empty:
            latest["forward_skill"] = {
                "target": f"tercile of realised volatility over the next {forward_horizon} sessions",
                "forward_brier": float(headline["paired_forward_brier"].iloc[0]),
                "climatology_brier": float(headline["paired_climatology_brier"].iloc[0]),
                "beats_climatology": bool(headline["beats_climatology"].iloc[0]),
                "forward_hit_rate": float(headline["forward_hit_rate"].iloc[0]),
                "climatology_hit_rate": float(reference["forward_hit_rate"].iloc[0]),
                "observations": int(headline["paired_observations"].iloc[0]),
                "note": "scored on the days this model and climatology both covered",
            }
    decision_value = _decision_value(
        features,
        {"baseline": baseline, "ensemble": ensemble, "ensemble_calibrated": calibrated},
        float(settings.get("evaluation_target_volatility", 0.10)),
    )
    return MarketStateResult(history, latest, columns, diagnostics, comparison, decision_value)


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
    raw = pd.DataFrame(index=style_returns.index)
    for column in style_returns:
        horizon_scores = [_annualized_score(style_returns[column], window) for window in windows]
        raw[column] = sum(weight * score for weight, score in zip(weights, horizon_scores))

    dimensions = (
        "size", "value", "quality", "conservative_investment", "momentum", "reversal", "defensive"
    )
    history = pd.DataFrame(index=style_returns.index)
    latest_rows: list[dict[str, Any]] = []
    temperature = float(config["models"]["style"]["score_temperature"])
    pairs = {
        "size": ("small", "large"),
        "value": ("value", "growth"),
        "quality": ("quality", "speculative"),
        "conservative_investment": ("conservative", "aggressive"),
        "momentum": ("momentum", "anti_momentum"),
        "reversal": ("reversal", "trend_persistence"),
        "defensive": ("defensive", "cyclical"),
    }
    for dimension in dimensions:
        ff_column = dimension if dimension in raw else None
        etf_name = f"{dimension}_etf"
        etf_column = etf_name if etf_name in raw else None
        source_scores: list[pd.Series] = []
        if ff_column:
            history[f"ff_score_{dimension}"] = _trailing_z(raw[ff_column], normalization_window)
            source_scores.append(history[f"ff_score_{dimension}"])
        if etf_column:
            history[f"etf_score_{dimension}"] = _trailing_z(raw[etf_column], normalization_window)
            source_scores.append(history[f"etf_score_{dimension}"])
        if not source_scores:
            continue
        history[f"score_{dimension}"] = pd.concat(source_scores, axis=1).mean(axis=1, skipna=True)
        history[f"soft_sign_{dimension}"] = _sigmoid(history[f"score_{dimension}"] / temperature)
        if len(source_scores) == 2:
            both_available = source_scores[0].notna() & source_scores[1].notna()
            history[f"agreement_{dimension}"] = (
                (np.sign(source_scores[0]) == np.sign(source_scores[1]))
                .astype(float)
                .where(both_available)
            )

        positive_name, negative_name = pairs[dimension]
        valid = history.dropna(subset=[f"score_{dimension}", f"soft_sign_{dimension}"])
        if valid.empty:
            continue
        row = valid.iloc[-1]
        as_of = valid.index[-1]
        soft_sign = float(row[f"soft_sign_{dimension}"])
        favored = positive_name if soft_sign >= 0.5 else negative_name
        age_business_days = int(np.busday_count(as_of.date(), history.index.max().date()))
        ff_score = row.get(f"ff_score_{dimension}", np.nan)
        etf_score = row.get(f"etf_score_{dimension}", np.nan)
        agreement = row.get(f"agreement_{dimension}", np.nan)
        ff_available = bool(np.isfinite(ff_score))
        etf_available = bool(np.isfinite(etf_score))
        latest_rows.append(
            {
                "as_of": as_of.date().isoformat(),
                "age_business_days": age_business_days,
                "data_status": "fresh" if age_business_days <= 5 else "stale",
                "dimension": dimension,
                "favored_style": favored,
                "favored_strength": max(soft_sign, 1.0 - soft_sign),
                "soft_sign": soft_sign,
                "score": float(row[f"score_{dimension}"]),
                "ff_score": float(ff_score) if ff_available else np.nan,
                "etf_score": float(etf_score) if etf_available else np.nan,
                "source_mode": (
                    "ff+etf"
                    if ff_available and etf_available
                    else ("ff" if ff_available else "etf")
                ),
                "source_agreement": bool(agreement) if np.isfinite(agreement) else None,
            }
        )
    return StyleResult(history=history, latest=pd.DataFrame(latest_rows))

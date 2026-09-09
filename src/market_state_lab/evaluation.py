"""The benchmark ledger: what any future rule has to beat, and how it is scored.

`research.py` names four benchmarks a candidate must be pre-registered against.
This computes them. Naming a bar that nothing can measure is how a bar quietly
becomes decorative, and the predecessor's whole failure was a comparison
problem rather than a coding one.

The four, from section 12.3:

    no_new_defense                      full exposure, always
    fixed_low_exposure                  a constant reduced position
    trailing_volatility_target          the one mechanical rule that survived
    fixed_long_term_trend_rule          in above the long average, out below

Three disciplines are baked in because each one previously flattered a result:

*The exposure-matched control.* A rule that de-risks also de-leverages, and
de-leveraging alone lowers drawdown. Comparing against full exposure therefore
measures the leverage change, not the signal - which is exactly how this project
once reported a drawdown edge that was not there. The matched control holds the
same average position with no signal in it, and the scaling that matches it is
computed from an expanding window so it cannot see the future.

*Signals execute after they are known.* A close-based signal trades at the next
available point. Assuming a fill at the close that produced the signal is a free
day of hindsight in every observation.

*Drawdown is a path.* It is not a mean, and resampling that breaks the path
destroys the quantity being measured, so the bootstrap here draws fixed-length
circular blocks and keeps candidates paired across draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# The names research.py pre-registers. Keep them identical: a benchmark that is
# named one thing in the registry and another in the ledger is not the same bar.
NO_NEW_DEFENSE = "no_new_defense"
FIXED_LOW_EXPOSURE = "fixed_low_exposure"
VOLATILITY_TARGET = "trailing_or_ewma_volatility_target"
# The name the registry pre-registers covers the family; the ledger computes
# both members so the cheaper EWMA variant is not a promise with no number.
EWMA_TARGET = "ewma_volatility_target"
TREND_RULE = "fixed_long_term_trend_rule"
BENCHMARKS = (NO_NEW_DEFENSE, FIXED_LOW_EXPOSURE, VOLATILITY_TARGET, EWMA_TARGET, TREND_RULE)


@dataclass(frozen=True)
class LedgerSettings:
    """Costs and conventions, fixed before anything is scored."""

    transaction_cost_bps: float = 2.0
    cash_rate_annual: float = 0.0
    fixed_low_exposure: float = 0.6
    volatility_target_annual: float = 0.10
    volatility_window: int = 20
    trend_window: int = 200
    exposure_cap: float = 1.0


def benchmark_exposures(
    prices: pd.Series,
    settings: LedgerSettings = LedgerSettings(),
) -> pd.DataFrame:
    """The four reference exposures, each one causal.

    Every series here is shifted so the position on a day is decided by data
    from before it. Without that, the volatility target reads today's volatility
    to size today's position and the whole comparison is unearned.
    """
    returns = prices.pct_change()
    realised = returns.rolling(settings.volatility_window).std() * np.sqrt(TRADING_DAYS)
    target = (settings.volatility_target_annual / realised).clip(upper=settings.exposure_cap)
    ewma_vol = returns.ewm(alpha=0.06, min_periods=settings.volatility_window).std() * np.sqrt(
        TRADING_DAYS
    )
    ewma = (settings.volatility_target_annual / ewma_vol).clip(upper=settings.exposure_cap)
    above = (prices > prices.rolling(settings.trend_window).mean()).astype(float)
    return pd.DataFrame(
        {
            NO_NEW_DEFENSE: pd.Series(1.0, index=prices.index),
            FIXED_LOW_EXPOSURE: pd.Series(settings.fixed_low_exposure, index=prices.index),
            # Shifted: the position is set by yesterday's information and held
            # through today, which is the only version anyone could have traded.
            VOLATILITY_TARGET: target.shift(1),
            EWMA_TARGET: ewma.shift(1),
            TREND_RULE: above.shift(1),
        }
    )


def ledger_returns(
    prices: pd.Series,
    exposures: pd.DataFrame,
    settings: LedgerSettings = LedgerSettings(),
) -> pd.DataFrame:
    """Net returns for each exposure path, with costs and cash included.

    Cash is not optional. A strategy that sits 40% in cash earns something on
    it, and omitting that makes every de-risked benchmark look worse than it was
    - which flatters whatever is being compared against them.
    """
    market = prices.pct_change()
    daily_cash = settings.cash_rate_annual / TRADING_DAYS
    out: dict[str, pd.Series] = {}
    for name in exposures.columns:
        weight = exposures[name].reindex(market.index).clip(0.0, settings.exposure_cap)
        turnover = weight.diff().abs().fillna(0.0)
        cost = turnover * settings.transaction_cost_bps / 10_000.0
        out[name] = weight * market + (1.0 - weight) * daily_cash - cost
    return pd.DataFrame(out).dropna(how="all")


def matched_control(
    candidate: pd.Series,
    baseline_exposure: pd.Series,
    candidate_exposure: pd.Series,
    minimum: int = TRADING_DAYS,
) -> pd.Series:
    """The baseline rescaled to the candidate's average position, causally.

    This is the control that took the drawdown edge away. A candidate holding
    less than the baseline will show a smaller drawdown for that reason alone,
    so the comparison has to be against the same average exposure carrying no
    signal. The ratio expands rather than using the full sample, because a
    full-sample ratio knows how much the candidate will hold in years it has not
    reached yet.
    """
    common = candidate_exposure.index.intersection(baseline_exposure.index)
    ratio = (
        candidate_exposure.reindex(common).expanding(min_periods=minimum).mean()
        / baseline_exposure.reindex(common).expanding(min_periods=minimum).mean()
    )
    scale = ratio.ffill().bfill().clip(lower=0.0, upper=2.0)
    return (candidate.reindex(common) * 0.0).add(
        baseline_exposure.reindex(common) * scale, fill_value=0.0
    )


def drawdown_depth(returns: np.ndarray) -> float:
    """Maximum peak-to-trough depth of the compounded path."""
    if returns.size == 0 or not np.isfinite(returns).all():
        return float("nan")
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def block_indices(total: int, block: int, generator: np.random.Generator) -> np.ndarray:
    """Fixed-length circular blocks, which is what keeps a path a path.

    Resampling single days destroys the ordering that drawdown is made of, so
    the answer would be about a different quantity than the one reported.
    """
    block = max(1, min(block, total))
    starts = generator.integers(0, total, size=int(np.ceil(total / block)))
    picks = np.concatenate([(np.arange(start, start + block) % total) for start in starts])
    return picks[:total]


def paired_bootstrap(
    left: pd.Series,
    right: pd.Series,
    statistic,
    draws: int = 1000,
    block: int = 60,
    seed: int = 0,
) -> dict[str, Any]:
    """A confidence interval on the difference, with both series drawn together.

    Paired: the same block indices are applied to both, so the interval is about
    the difference between two strategies over one history rather than about two
    independent histories that never coexisted.
    """
    common = left.dropna().index.intersection(right.dropna().index)
    a, b = left.loc[common].to_numpy(), right.loc[common].to_numpy()
    if len(common) < block:
        return {"observations": len(common), "difference": None, "low": None, "high": None,
                "significant": False, "note": "fewer observations than one block"}
    generator = np.random.default_rng(seed)
    differences = np.array([
        statistic(a[idx]) - statistic(b[idx])
        for idx in (block_indices(len(common), block, generator) for _ in range(draws))
    ])
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "observations": len(common),
        "difference": float(statistic(a) - statistic(b)),
        "low": float(low),
        "high": float(high),
        # Two-sided: an interval that excludes zero on either side is a finding,
        # and one that spans it is insufficient evidence rather than equivalence.
        "significant": bool(low > 0 or high < 0),
        "block_length": block,
        "draws": draws,
    }


def forward_drawdown_event(
    prices: pd.Series,
    horizon: int = 20,
    threshold: float = 0.05,
) -> pd.Series:
    """The research event: did price fall this far within the horizon, from here.

    Settled forward and therefore unavailable at the time it describes, which is
    the point - it is the label a rule would have to have anticipated, not an
    input a rule may use.
    """
    forward_low = prices.shift(-1).rolling(horizon, min_periods=horizon).min().shift(-(horizon - 1))
    return (forward_low / prices - 1.0 <= -abs(threshold)).astype(float).where(forward_low.notna())


def compare_against_benchmarks(
    prices: pd.Series,
    candidate_exposure: pd.Series | None = None,
    settings: LedgerSettings = LedgerSettings(),
    seed: int = 0,
) -> pd.DataFrame:
    """Every benchmark on one common day set, plus the matched control.

    One day set for all of them: scoring each on whatever days it happens to
    have makes the columns incomparable, and comparing incomparable columns is
    how three headline claims survived here for months.
    """
    exposures = benchmark_exposures(prices, settings)
    if candidate_exposure is not None:
        exposures = exposures.assign(candidate=candidate_exposure)
    returns = ledger_returns(prices, exposures, settings)
    common = returns.dropna().index
    rows: list[dict[str, Any]] = []
    reference = returns.loc[common, VOLATILITY_TARGET]

    for name in returns.columns:
        series = returns.loc[common, name]
        against_matched = None
        if name not in (NO_NEW_DEFENSE, VOLATILITY_TARGET):
            control = matched_control(
                series, exposures[VOLATILITY_TARGET], exposures[name]
            ).reindex(common).dropna()
            if not control.empty:
                controlled = ledger_returns(
                    prices, pd.DataFrame({"m": control}), settings
                )["m"].reindex(common)
                against_matched = paired_bootstrap(
                    series, controlled, drawdown_depth, seed=seed
                )
        rows.append({
            "strategy": name,
            "days": len(common),
            "annual_return": float(series.mean() * TRADING_DAYS),
            "annual_volatility": float(series.std() * np.sqrt(TRADING_DAYS)),
            "max_drawdown": drawdown_depth(series.to_numpy()),
            "average_exposure": float(exposures[name].reindex(common).mean()),
            "drawdown_vs_volatility_target": (
                None if name == VOLATILITY_TARGET
                else paired_bootstrap(series, reference, drawdown_depth, seed=seed)
            ),
            # The comparison that matters: same average position, no signal.
            "drawdown_vs_matched_control": against_matched,
        })
    return pd.DataFrame(rows)


def conditional_frequencies(
    prices: pd.Series,
    minimum: int = TRADING_DAYS,
) -> dict[str, Any]:
    """Causal conditional frequencies of the research event, around today's state.

    Descriptive, not predictive: every number is a frequency over settled
    history computed from expanding ranks, and none of it says what happens
    next. It answers the operational question a bare percentile cannot - when
    the market looked like this before, how often did a 5% drop arrive within
    20 sessions?

    The two conditions are the ones the replay machinery already validated as
    measurable: whether price sits below its 200-session average, and how high
    realised 20-session volatility ranks against its own history. Both are
    computed causally - the rank of a day uses only days before it, and the
    event window is entirely future, so the pairing is a pairing a person at
    the time could have made.
    """
    from market_state_lab.market_evidence import causal_percentile

    event = forward_drawdown_event(prices)
    returns = prices.pct_change()
    vol20 = returns.rolling(20).std() * np.sqrt(TRADING_DAYS)
    rank = causal_percentile(vol20, minimum)
    below = (prices <= prices.rolling(200).mean()).shift(1)
    frame = pd.concat(
        [event.rename("event"), rank.rename("vol_rank"), vol20.rename("vol"), below.rename("below")],
        axis=1,
    ).dropna()
    if frame.empty or len(frame) < minimum:
        return {}
    # A Series rather than a bare array: masks are combined and applied to
    # frame, and an index that does not match frame's raises unalignable.
    post13 = pd.Series(frame.index >= pd.Timestamp("2013-07-01"), index=frame.index)
    current = frame.iloc[-1]
    current_rank = float(current["vol_rank"])

    def rates(mask: pd.Series) -> dict[str, Any]:
        whole, recent = frame.loc[mask], frame.loc[mask & post13]
        return {
            "sessions": int(len(whole)),
            "event_frequency": float(whole["event"].mean()) if len(whole) else None,
            "event_frequency_post_2013": float(recent["event"].mean()) if len(recent) else None,
        }

    return {
        "event": "drawdown_5pct_20_sessions",
        "sessions": int(len(frame)),
        "base_rate": rates(pd.Series(True, index=frame.index)),
        "above_200d_ma": rates(~frame["below"].astype(bool)),
        "below_200d_ma": rates(frame["below"].astype(bool)),
        "vol_at_least_current": rates(frame["vol_rank"] >= current_rank),
        "current": {
            "vol_rank": current_rank,
            "realised_vol_20d": float(current["vol"]),
            "below_200d_ma": bool(current["below"]),
            "as_of": str(frame.index[-1].date()),
        },
        "note": "historical frequencies over settled windows, not probabilities",
    }


def build_episodes(event: pd.Series, gap_sessions: int = 5) -> pd.DataFrame:
    """Group overlapping event windows into episodes, per 12.2.

    A 20-session window means one crisis spans many overlapping event days, and
    scoring each of them as a separate detection inflates the recall of any rule
    that was right once. Windows closer than `gap_sessions` merge into one
    episode; the count of episodes, not of event days, is the denominator.
    """
    # Only the days the event fired on. dropna() alone keeps every non-NaN row,
    # zeros included, which turns the whole series into one giant episode.
    flagged = event.loc[event.eq(1)]
    if flagged.empty:
        return pd.DataFrame(columns=["episode", "start", "end", "event_days"])
    positions = event.index.get_indexer(flagged.index)
    episodes: list[dict[str, Any]] = []
    start = positions[0]
    previous = positions[0]
    for position in positions[1:]:
        if position - previous > gap_sessions:
            episodes.append(
                {
                    "episode": len(episodes) + 1,
                    "start": event.index[start],
                    "end": event.index[previous],
                    "event_days": int(previous - start + 1),
                }
            )
            start = position
        previous = position
    episodes.append(
        {
            "episode": len(episodes) + 1,
            "start": event.index[start],
            "end": event.index[previous],
            "event_days": int(previous - start + 1),
        }
    )
    return pd.DataFrame(episodes)


def replay_metrics(
    alerts: pd.Series,
    event: pd.Series,
    episodes: pd.DataFrame,
) -> dict[str, Any]:
    """How a binary rule did against the research event, at episode level.

    An episode is detected when the rule alerts on at least one session inside
    its window. Daily precision is reported alongside because the two measures
    fail differently: a rule that is right once per crisis has high episode
    recall and terrible daily precision, and both numbers are what they are.
    """
    both = pd.concat([alerts.rename("alert"), event.rename("event")], axis=1).dropna()
    if both.empty or episodes.empty:
        return {
            "sessions": 0,
            "alert_sessions": 0,
            "alerts_per_year": np.nan,
            "episodes": 0,
            "episodes_detected": 0,
            "episode_recall": np.nan,
            "false_alert_sessions": 0,
            "daily_precision": np.nan,
        }
    hit = both["alert"].eq(1) & both["event"].eq(1)
    detected = 0
    for episode in episodes.to_dict("records"):
        window = both.index[
            (both.index >= episode["start"]) & (both.index <= episode["end"])
        ]
        if (both.loc[window, "alert"] == 1).any():
            detected += 1
    alerts_total = int(both["alert"].sum())
    return {
        "sessions": len(both),
        "alert_sessions": alerts_total,
        "alerts_per_year": float(alerts_total / len(both) * TRADING_DAYS),
        "episodes": len(episodes),
        "episodes_detected": detected,
        "episode_recall": float(detected / len(episodes)),
        "false_alert_sessions": int((both["alert"].eq(1) & ~both["event"].astype(bool)).sum()),
        "daily_precision": float(hit.sum() / alerts_total) if alerts_total else np.nan,
    }


def current_target_exposures(
    prices: pd.Series,
    settings: LedgerSettings = LedgerSettings(),
) -> dict[str, Any]:
    """The reference exposure the only measured mechanism implies today.

    This is the one number in the project with a measured history behind it:
    the volatility target never predicts anything and its 26-year drawdown is
    on record, including the part worth reading - its worst case is the
    2000-2002 slow bear, not 2008. It is a yardstick, not advice.
    """
    exposures = benchmark_exposures(prices, settings)
    if exposures.empty:
        return {}
    last = exposures.dropna(how="all").iloc[-1]
    return {
        "trailing_exposure": float(last[VOLATILITY_TARGET]) if np.isfinite(last[VOLATILITY_TARGET]) else None,
        "ewma_exposure": float(last[EWMA_TARGET]) if np.isfinite(last[EWMA_TARGET]) else None,
        "trend_exposure": float(last[TREND_RULE]) if np.isfinite(last[TREND_RULE]) else None,
        "target_volatility_annual": settings.volatility_target_annual,
        "exposure_cap": settings.exposure_cap,
        "measured_history": (
            "26-year ledger: 6.4% annual return, -34.4% maximum drawdown at "
            "0.72 average exposure; the worst case was the 2000-2002 slow bear "
            "(0.95 exposure at the September 2000 peak), not 2008"
        ),
        "note": "a yardstick, not advice",
    }


__all__ = [
    "BENCHMARKS",
    "EWMA_TARGET",
    "LedgerSettings",
    "TREND_RULE",
    "VOLATILITY_TARGET",
    "benchmark_exposures",
    "block_indices",
    "build_episodes",
    "compare_against_benchmarks",
    "conditional_frequencies",
    "current_target_exposures",
    "drawdown_depth",
    "forward_drawdown_event",
    "ledger_returns",
    "matched_control",
    "paired_bootstrap",
    "replay_metrics",
]

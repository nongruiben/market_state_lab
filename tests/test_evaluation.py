"""The benchmark ledger. research.py names these bars; this is what measures
them, and each discipline here previously flattered a result."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_state_lab.evaluation import (
    BENCHMARKS,
    FIXED_LOW_EXPOSURE,
    NO_NEW_DEFENSE,
    TREND_RULE,
    VOLATILITY_TARGET,
    LedgerSettings,
    benchmark_exposures,
    block_indices,
    compare_against_benchmarks,
    drawdown_depth,
    forward_drawdown_event,
    ledger_returns,
    matched_control,
    paired_bootstrap,
)
from market_state_lab.research import REQUIRED_BENCHMARKS

SESSIONS = 1500


def _prices(seed: int = 11, vol: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-02", periods=SESSIONS)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0003, vol, SESSIONS))), index=index)


# ---------------------------------------------------------------------------
# The bars research.py names must be the bars this computes
# ---------------------------------------------------------------------------


def test_the_ledger_computes_exactly_the_benchmarks_the_registry_requires() -> None:
    # A bar that is named in one place and measured in another under a different
    # name is not the same bar.
    assert set(BENCHMARKS) == set(REQUIRED_BENCHMARKS)
    assert set(benchmark_exposures(_prices()).columns) == set(REQUIRED_BENCHMARKS)


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


def test_every_exposure_is_decided_before_the_day_it_applies_to() -> None:
    prices = _prices()
    exposures = benchmark_exposures(prices)
    # Appending a future observation must not change any earlier position.
    extended = pd.concat([prices, pd.Series([prices.iloc[-1] * 0.5],
                                            index=[prices.index[-1] + pd.Timedelta(days=1)])])
    later = benchmark_exposures(extended).reindex(prices.index)
    for column in (VOLATILITY_TARGET, TREND_RULE):
        pd.testing.assert_series_equal(
            exposures[column].dropna(), later[column].dropna(), check_names=False
        )


def test_the_volatility_target_sizes_from_yesterdays_volatility() -> None:
    prices = _prices()
    exposures = benchmark_exposures(prices)
    settings = LedgerSettings()
    realised = prices.pct_change().rolling(settings.volatility_window).std() * np.sqrt(252)
    unshifted = (settings.volatility_target_annual / realised).clip(upper=1.0)
    # Reading today's volatility to size today's position is a free day of
    # hindsight in every observation.
    assert exposures[VOLATILITY_TARGET].iloc[-1] == pytest.approx(unshifted.iloc[-2])


def test_the_forward_event_is_settled_after_the_day_it_labels() -> None:
    prices = _prices()
    event = forward_drawdown_event(prices, horizon=20, threshold=0.05)
    # The last horizon of days cannot be labelled yet, and pretending otherwise
    # is how a label becomes an input.
    assert event.iloc[-1] != event.iloc[-1] or pd.isna(event.iloc[-1])
    assert event.dropna().isin([0.0, 1.0]).all()


# ---------------------------------------------------------------------------
# The exposure-matched control
# ---------------------------------------------------------------------------


def test_a_lower_exposure_alone_lowers_drawdown() -> None:
    # The fact the matched control exists for: this needs no signal at all.
    prices = _prices()
    returns = ledger_returns(prices, benchmark_exposures(prices))
    full = drawdown_depth(returns[NO_NEW_DEFENSE].dropna().to_numpy())
    low = drawdown_depth(returns[FIXED_LOW_EXPOSURE].dropna().to_numpy())
    assert low > full  # less negative


def test_the_matched_control_holds_the_candidates_average_position() -> None:
    prices = _prices()
    exposures = benchmark_exposures(prices)
    control = matched_control(
        pd.Series(0.0, index=prices.index),
        exposures[VOLATILITY_TARGET],
        exposures[FIXED_LOW_EXPOSURE],
    ).dropna()
    candidate_average = exposures[FIXED_LOW_EXPOSURE].reindex(control.index).mean()
    assert control.mean() == pytest.approx(candidate_average, rel=0.25)


def test_the_matching_ratio_cannot_see_the_future() -> None:
    prices = _prices()
    exposures = benchmark_exposures(prices)
    args = (pd.Series(0.0, index=prices.index), exposures[VOLATILITY_TARGET])
    full = matched_control(*args, exposures[FIXED_LOW_EXPOSURE])
    half = matched_control(
        args[0].iloc[:800], args[1].iloc[:800], exposures[FIXED_LOW_EXPOSURE].iloc[:800]
    )
    # A full-sample ratio would know how much the candidate holds in years it
    # has not reached.
    pd.testing.assert_series_equal(
        full.iloc[:800].dropna().iloc[-50:], half.dropna().iloc[-50:], check_names=False
    )


# ---------------------------------------------------------------------------
# Costs and cash
# ---------------------------------------------------------------------------


def test_cash_is_not_omitted_from_a_de_risked_benchmark() -> None:
    prices = _prices()
    exposures = benchmark_exposures(prices)
    without = ledger_returns(prices, exposures, LedgerSettings(cash_rate_annual=0.0))
    with_cash = ledger_returns(prices, exposures, LedgerSettings(cash_rate_annual=0.04))
    # Omitting it makes every de-risked benchmark look worse than it was, which
    # flatters whatever is compared against them.
    assert with_cash[FIXED_LOW_EXPOSURE].mean() > without[FIXED_LOW_EXPOSURE].mean()
    # Full exposure holds no cash, so it is untouched.
    assert with_cash[NO_NEW_DEFENSE].mean() == pytest.approx(without[NO_NEW_DEFENSE].mean())


def test_turnover_is_charged() -> None:
    prices = _prices()
    exposures = benchmark_exposures(prices)
    free = ledger_returns(prices, exposures, LedgerSettings(transaction_cost_bps=0.0))
    charged = ledger_returns(prices, exposures, LedgerSettings(transaction_cost_bps=50.0))
    assert charged[VOLATILITY_TARGET].sum() < free[VOLATILITY_TARGET].sum()
    # A constant weight never trades, so it pays nothing.
    assert charged[FIXED_LOW_EXPOSURE].sum() == pytest.approx(free[FIXED_LOW_EXPOSURE].sum())


# ---------------------------------------------------------------------------
# The bootstrap keeps the path intact
# ---------------------------------------------------------------------------


def test_the_bootstrap_draws_whole_blocks() -> None:
    picks = block_indices(100, 10, np.random.default_rng(0))
    assert len(picks) == 100
    runs = [picks[i + 1] - picks[i] for i in range(len(picks) - 1)]
    # Consecutive within a block; resampling single days would destroy the
    # ordering that a drawdown is made of.
    assert sum(1 for r in runs if r == 1) >= 80


def test_blocks_wrap_rather_than_running_off_the_end() -> None:
    picks = block_indices(20, 8, np.random.default_rng(3))
    assert picks.max() < 20 and picks.min() >= 0


def test_an_interval_spanning_zero_is_not_significant() -> None:
    same = pd.Series(np.random.default_rng(1).normal(0, 0.01, 800))
    result = paired_bootstrap(same, same.copy(), drawdown_depth, draws=200, seed=2)
    assert result["difference"] == pytest.approx(0.0, abs=1e-12)
    assert not result["significant"]


def test_too_little_history_reports_that_rather_than_a_number() -> None:
    short = pd.Series([0.01, -0.01, 0.02])
    result = paired_bootstrap(short, short.copy(), drawdown_depth, block=60)
    assert result["difference"] is None
    assert "fewer observations than one block" in result["note"]


# ---------------------------------------------------------------------------
# The comparison table
# ---------------------------------------------------------------------------


def test_every_strategy_is_scored_on_one_common_day_set() -> None:
    table = compare_against_benchmarks(_prices())
    # Scoring each on whatever days it happens to have makes the columns
    # incomparable, and comparing incomparable columns is how three headline
    # claims survived here for months.
    assert table["days"].nunique() == 1
    assert set(table["strategy"]) == set(BENCHMARKS)


def test_the_matched_control_column_is_present_for_the_de_risked_rules() -> None:
    table = compare_against_benchmarks(_prices()).set_index("strategy")
    assert table.loc[FIXED_LOW_EXPOSURE, "drawdown_vs_matched_control"] is not None
    assert table.loc[TREND_RULE, "drawdown_vs_matched_control"] is not None


def test_a_candidate_can_be_added_and_is_scored_the_same_way() -> None:
    prices = _prices()
    candidate = pd.Series(0.5, index=prices.index)
    table = compare_against_benchmarks(prices, candidate_exposure=candidate)
    assert "candidate" in set(table["strategy"])
    assert table["days"].nunique() == 1

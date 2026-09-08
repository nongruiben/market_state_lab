"""Every number below is hand-computable, because a payoff calculator whose
arithmetic is only checked against itself is worth nothing."""

from __future__ import annotations

import pytest

from market_state_lab.scenarios import (
    PutQuote,
    ReferenceExposure,
    protective_put_scenarios,
    size_protection,
    summarise_candidate,
)


def _exposure() -> ReferenceExposure:
    # 100,000 / 500 = exactly 200 shares, so the arithmetic stays checkable.
    return ReferenceExposure(symbol="SPY", spot=500.0, notional_usd=100_000.0)


def _put(strike: float = 475.0, ask: float = 8.0, bid: float = 7.6) -> PutQuote:
    return PutQuote(
        symbol="SPY", expiry="2026-12-18", strike=strike, bid=bid, ask=ask, multiplier=100
    )


def test_sizing_reports_the_rounding_rather_than_hiding_it() -> None:
    sizing = size_protection(_exposure(), _put())
    # 200 shares / 100 per contract = 2.0 exactly.
    assert sizing["reference_shares"] == pytest.approx(200.0)
    assert sizing["contracts_exact"] == pytest.approx(2.0)
    assert sizing["contracts"] == 2
    assert sizing["coverage_ratio"] == pytest.approx(1.0)
    assert sizing["uncovered_shares"] == pytest.approx(0.0)


def test_partial_contract_leaves_visible_uncovered_shares() -> None:
    # 100,000 / 640 = 156.25 shares -> 1.5625 contracts -> 1 whole contract.
    sizing = size_protection(ReferenceExposure("QQQ", 640.0), _put())
    assert sizing["contracts"] == 1
    assert sizing["shares_covered"] == 100
    assert sizing["uncovered_shares"] == pytest.approx(56.25)
    assert sizing["coverage_ratio"] == pytest.approx(100 / 156.25)


def test_expiry_payoff_matches_hand_calculation() -> None:
    frame = protective_put_scenarios(_exposure(), _put(), fee_per_contract=0.65)
    # 2 contracts x 100 x $8.00 ask = $1,600 premium, plus $1.30 fees = $1,601.30.
    assert frame["premium_and_fees"].iloc[0] == pytest.approx(1601.30)

    worst = frame.loc[frame["move"] == -0.20].iloc[0]
    # Terminal 500 x 0.80 = 400. Stock: 200 x 400 = 80,000.
    # Put: 2 x 100 x max(475 - 400, 0) = 15,000. Hedged: 80,000 + 15,000 - 1,601.30.
    assert worst["terminal_price"] == pytest.approx(400.0)
    assert worst["unhedged_value"] == pytest.approx(80_000.0)
    assert worst["put_payoff"] == pytest.approx(15_000.0)
    assert worst["hedged_value"] == pytest.approx(93_398.70)
    assert worst["hedged_pnl"] == pytest.approx(-6_601.30)
    assert worst["unhedged_pnl"] == pytest.approx(-20_000.0)
    assert worst["protection_vs_unhedged"] == pytest.approx(13_398.70)


def test_above_the_strike_the_hedge_costs_exactly_its_premium() -> None:
    frame = protective_put_scenarios(_exposure(), _put(), fee_per_contract=0.65)
    for move in (0.0, 0.05, 0.10):
        row = frame.loc[frame["move"] == move].iloc[0]
        assert row["put_payoff"] == pytest.approx(0.0)
        assert row["protection_vs_unhedged"] == pytest.approx(-1601.30)


def test_shallow_move_inside_the_deductible_is_unprotected() -> None:
    # A 475 strike on a 500 spot only starts paying below -5%.
    frame = protective_put_scenarios(_exposure(), _put(strike=475.0))
    row = frame.loc[frame["move"] == -0.05].iloc[0]
    assert row["terminal_price"] == pytest.approx(475.0)
    assert row["put_payoff"] == pytest.approx(0.0)


def test_cost_uses_the_ask_never_the_mid() -> None:
    at_ask = protective_put_scenarios(_exposure(), _put(ask=8.0, bid=7.6), fee_per_contract=0.0)
    assert at_ask["premium_and_fees"].iloc[0] == pytest.approx(1600.0)
    # The mid would have been 7.80 -> 1,560; charging it would flatter the hedge.
    assert at_ask["premium_and_fees"].iloc[0] != pytest.approx(1560.0)


def test_a_quote_with_no_ask_cannot_be_priced() -> None:
    with pytest.raises(ValueError, match="positive ask"):
        PutQuote(symbol="SPY", expiry="2026-12-18", strike=475.0, bid=7.6, ask=float("nan"))


def test_summary_keeps_the_tradeoff_visible_and_scores_nothing() -> None:
    deep = summarise_candidate(protective_put_scenarios(_exposure(), _put(strike=450.0, ask=4.0)))
    near = summarise_candidate(protective_put_scenarios(_exposure(), _put(strike=490.0, ask=12.0)))
    # The cheaper strike protects less in the crash and costs less when flat.
    assert deep["cost_usd"] < near["cost_usd"]
    assert deep["protection_at_worst_move"] < near["protection_at_worst_move"]
    assert deep["pnl_if_flat"] > near["pnl_if_flat"]
    # No composite score exists to rank them by.
    assert "score" not in deep and "rank" not in deep

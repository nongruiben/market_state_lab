"""A screen has to reject for a stated reason and stay silent on what it does
not know, and the two controls have to sit in the same table as the puts - a
list of six puts with no "do nothing" row reads as "pick one of these"."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_state_lab.defense_tools import (
    DEGRADED,
    QUARANTINED,
    UNAVAILABLE,
    VALID,
    ScreenLimits,
    screen_candidates,
    screened,
    shortlist,
)
from market_state_lab.scenarios import (
    PutQuote,
    ReferenceExposure,
    no_protection_scenarios,
    protective_put_scenarios,
    reduced_exposure_scenarios,
    summarise_candidate,
    summarise_control,
)

LIMITS = ScreenLimits()
EXPOSURE = ReferenceExposure("SPY", spot=770.19, notional_usd=100_000.0)


def _row(**over) -> dict:
    row = {
        "symbol": "SPY",
        "expiry": "20261120",
        "strike": 730.0,
        "status": "priced",
        "bid": 7.65,
        "ask": 7.68,
        "relative_spread": 0.0039,
        "open_interest": 9695.0,
        "quote_age_seconds": None,
    }
    row.update(over)
    return row


def _screen(rows: list[dict], **kw) -> pd.DataFrame:
    kw.setdefault("quote_basis", "frozen_last_session")
    kw.setdefault("market_open", False)
    return screen_candidates(pd.DataFrame(rows), LIMITS, **kw)


def test_a_frozen_book_after_the_bell_is_the_whole_market_not_a_defect() -> None:
    out = _screen([_row()])
    assert out.loc[0, "quote_qualification"] == VALID
    assert out.loc[0, "screened"]


def test_the_same_frozen_book_while_the_session_trades_is_degraded() -> None:
    out = _screen([_row()], market_open=True)
    assert out.loc[0, "quote_qualification"] == DEGRADED
    # Degraded still describes a real market, so it is not thrown away.
    assert out.loc[0, "screened"]
    assert "frozen book while the session is trading" in out.loc[0, "screen_notes"]


def test_a_live_quote_past_the_age_limit_is_degraded_not_rejected() -> None:
    fresh = _screen([_row(quote_age_seconds=4.0)], quote_basis="tick_timestamp")
    stale = _screen([_row(quote_age_seconds=300.0)], quote_basis="tick_timestamp")
    assert fresh.loc[0, "quote_qualification"] == VALID
    assert stale.loc[0, "quote_qualification"] == DEGRADED
    assert "over the 60s limit" in stale.loc[0, "screen_notes"]


def test_a_wide_spread_is_quarantined_with_the_number_that_failed() -> None:
    out = _screen([_row(relative_spread=0.32)])
    assert out.loc[0, "quote_qualification"] == QUARANTINED
    assert not out.loc[0, "screened"]
    assert "relative spread 32.0% over 15%" in out.loc[0, "screen_failures"]


def test_thin_open_interest_is_quarantined_but_absent_open_interest_is_not() -> None:
    thin = _screen([_row(open_interest=12.0)])
    assert thin.loc[0, "quote_qualification"] == QUARANTINED
    assert "open interest 12 under 100" in thin.loc[0, "screen_failures"]

    # The tick arrives late or never. Dropping a liquid contract over a missing
    # tick is the same error as admitting an illiquid one over a zero.
    missing = _screen([_row(open_interest=None)])
    assert missing.loc[0, "quote_qualification"] == DEGRADED
    assert missing.loc[0, "screened"]
    assert "absent is not zero" in missing.loc[0, "screen_notes"]

    empty = _screen([_row(open_interest=0.0)])
    assert empty.loc[0, "quote_qualification"] == QUARANTINED


def test_an_unpriced_row_is_unavailable_and_never_screens_through() -> None:
    out = _screen([_row(status="no_ask", ask=None, relative_spread=np.nan)])
    assert out.loc[0, "quote_qualification"] == UNAVAILABLE
    assert not out.loc[0, "screened"]
    assert screened(out).empty


def test_the_do_nothing_control_is_exactly_the_unhedged_exposure() -> None:
    frame = no_protection_scenarios(EXPOSURE)
    assert (frame["hedged_pnl"] == frame["unhedged_pnl"]).all()
    assert (frame["premium_and_fees"] == 0.0).all()
    summary = summarise_control(frame)
    assert summary["cost_usd"] == 0.0
    assert summary["label"] == "no new protection"


def test_the_de_risk_control_keeps_its_share_of_both_directions() -> None:
    frame = reduced_exposure_scenarios(EXPOSURE, reduce_to=0.8, exit_cost_bps=2.0)
    worst = frame.loc[frame["move"].eq(-0.20)].iloc[0]
    best = frame.loc[frame["move"].eq(0.10)].iloc[0]
    # 80% of a -20% move is -16%, less the 4 dollars the sale cost.
    assert worst["hedged_pnl"] == pytest.approx(-16_004.0, abs=1.0)
    # And the same 20% that was sold does not participate in the rally.
    assert best["hedged_pnl"] == pytest.approx(7_996.0, abs=1.0)
    assert worst["protection_vs_unhedged"] == pytest.approx(3_996.0, abs=1.0)


def test_selling_costs_something_so_the_control_is_not_free() -> None:
    free = reduced_exposure_scenarios(EXPOSURE, reduce_to=0.8, exit_cost_bps=0.0)
    charged = reduced_exposure_scenarios(EXPOSURE, reduce_to=0.8, exit_cost_bps=2.0)
    assert summarise_control(charged)["cost_usd"] == pytest.approx(4.0)
    assert summarise_control(free)["cost_usd"] == 0.0


def test_a_control_and_a_candidate_carry_the_same_keys() -> None:
    put = PutQuote("SPY", "20261120", 730.0, bid=7.65, ask=7.68)
    candidate = summarise_candidate(protective_put_scenarios(EXPOSURE, put))
    control = summarise_control(no_protection_scenarios(EXPOSURE))
    # One table, read straight down, or the comparison is not a comparison.
    assert set(candidate) == set(control)


def _summaries() -> pd.DataFrame:
    rows = []
    for bucket, expiry in (("30-60d", "20261009"), ("60-90d", "20261120")):
        for depth, cost in ((-0.02, 0.0065), (-0.05, 0.0031), (-0.10, 0.0012)):
            rows.append(
                {
                    "label": f"put {depth} {expiry}",
                    "bucket": bucket,
                    "target_moneyness": depth,
                    "expiry": expiry,
                    "cost_pct_of_notional": cost,
                }
            )
    return pd.DataFrame(rows)


def test_the_short_list_is_three_depths_from_one_horizon() -> None:
    out = shortlist(_summaries(), LIMITS, max_candidates=3, horizon="60-90d")
    picked = out.loc[out["shortlisted"]]
    assert len(picked) == 3
    assert set(picked["bucket"]) == {"60-90d"}
    assert sorted(picked["target_moneyness"]) == [-0.10, -0.05, -0.02]
    # Depth and term are two questions; three rows cannot answer both.
    assert out.attrs["shortlist_horizon"] == "60-90d"


def test_nothing_is_ranked_within_the_short_list() -> None:
    out = shortlist(_summaries(), LIMITS, horizon="30-60d")
    assert "rank" not in out.columns and "score" not in out.columns
    assert set(out.loc[out["shortlisted"], "shortlist_reason"]) == {
        "-2% protection depth in 30-60d",
        "-5% protection depth in 30-60d",
        "-10% protection depth in 30-60d",
    }


def test_cost_over_the_limit_is_dropped_with_the_number_that_failed() -> None:
    summaries = _summaries()
    summaries.loc[summaries["target_moneyness"].eq(-0.02), "cost_pct_of_notional"] = 0.09
    out = shortlist(summaries, LIMITS, horizon="60-90d")
    dear = out.loc[out["target_moneyness"].eq(-0.02) & out["bucket"].eq("60-90d")].iloc[0]
    assert not dear["shortlisted"]
    assert "cost 9.00% over the 5% limit" in dear["shortlist_reason"]
    # The affordable depths still fill the remaining slots.
    assert out["shortlisted"].sum() == 2


def test_an_empty_candidate_set_still_returns_a_table() -> None:
    assert shortlist(pd.DataFrame()).empty
    assert screen_candidates(pd.DataFrame()).empty

"""Chain selection is where the wrong contract universe gets in, so the hazard
the live probe actually found is pinned here."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from market_state_lab.defense_tools import (
    DEFAULT_BUCKETS,
    ExpiryBucket,
    nearest_strike,
    plan_candidates,
    quotable,
    select_chain,
    select_expiries,
)

AS_OF = date(2026, 9, 8)


def _chain_params() -> pd.DataFrame:
    """Shaped like the real SPY response: many exchange rows, two trading classes,
    and the decoy listed first."""
    wide_expiries = ["20260911", "20261009", "20261120", "20270115"]
    wide_strikes = [float(s) for s in range(600, 900, 5)]
    rows = [
        {
            "exchange": "BOX",
            "trading_class": "2SPY",
            "multiplier": "100",
            "expirations": ["20260911", "20260916"],
            "strikes": [770.0, 775.0],
            "expiry_count": 2,
            "strike_count": 2,
        },
        {
            "exchange": "SMART",
            "trading_class": "SPY",
            "multiplier": "100",
            "expirations": wide_expiries,
            "strikes": wide_strikes,
            "expiry_count": len(wide_expiries),
            "strike_count": len(wide_strikes),
        },
        {
            "exchange": "PHLX",
            "trading_class": "SPY",
            "multiplier": "100",
            "expirations": wide_expiries[:2],
            "strikes": wide_strikes[:10],
            "expiry_count": 2,
            "strike_count": 10,
        },
    ]
    return pd.DataFrame(rows)


def test_select_chain_ignores_the_lookalike_trading_class() -> None:
    # The decoy is row 0 and would win an iloc[0]; it is a different universe.
    chain = select_chain(_chain_params(), "SPY")
    assert chain["trading_class"] == "SPY"
    assert chain["strike_count"] == 60
    assert chain["exchange"] == "SMART"


def test_select_chain_refuses_rather_than_guessing_when_nothing_matches() -> None:
    only_decoy = _chain_params().iloc[[0]]
    with pytest.raises(LookupError, match="matching trading class"):
        select_chain(only_decoy, "SPY")


def test_expiry_buckets_take_the_nearest_inside_each_window() -> None:
    chain = select_chain(_chain_params(), "SPY")
    picked = select_expiries(list(chain["expirations"]), AS_OF).set_index("bucket")
    # 20261009 is 31 days out -> 30-60d; 20261120 is 73 -> 60-90d.
    assert picked.loc["30-60d", "expiry"] == "20261009"
    assert picked.loc["30-60d", "days_to_expiry"] == 31
    assert picked.loc["60-90d", "expiry"] == "20261120"
    assert picked.loc["60-90d", "days_to_expiry"] == 73


def test_an_empty_bucket_says_so_instead_of_borrowing_a_neighbour() -> None:
    picked = select_expiries(["20260911"], AS_OF).set_index("bucket")
    assert picked.loc["30-60d", "status"] == "no_listed_expiry"
    assert picked.loc["30-60d", "expiry"] is None
    assert picked.loc["60-90d", "status"] == "no_listed_expiry"


def test_nearest_strike_breaks_ties_downward() -> None:
    # 767.5 is equidistant from 765 and 770; the lower strike is the cheaper one.
    assert nearest_strike([765.0, 770.0], 767.5) == 765.0
    assert nearest_strike([765.0, 770.0], 769.0) == 770.0
    assert nearest_strike([], 700.0) is None


def test_candidate_grid_covers_every_bucket_and_moneyness() -> None:
    chain = select_chain(_chain_params(), "SPY")
    plan = plan_candidates("SPY", chain, spot=770.0, as_of=AS_OF)
    assert len(plan) == len(DEFAULT_BUCKETS) * 3
    ok = quotable(plan)
    assert set(ok["bucket"]) == {"30-60d", "60-90d"}
    # -5% of 770 is 731.5; the 5-point grid puts the nearest listed strike at 730.
    near = ok.loc[ok["target_moneyness"].eq(-0.05)].iloc[0]
    assert near["strike"] == 730.0
    assert near["right"] == "P"
    assert near["actual_moneyness"] == pytest.approx(730.0 / 770.0 - 1.0)


def test_a_thin_chain_marks_the_strike_as_off_target_rather_than_pretending() -> None:
    sparse = pd.DataFrame(
        [
            {
                "exchange": "SMART",
                "trading_class": "SPY",
                "multiplier": "100",
                "expirations": ["20261009"],
                "strikes": [500.0, 900.0],
                "expiry_count": 1,
                "strike_count": 2,
            }
        ]
    )
    plan = plan_candidates("SPY", select_chain(sparse, "SPY"), spot=770.0, as_of=AS_OF)
    # Nothing sits near -5% of 770, so no row may claim to be that protection.
    assert (plan["status"] == "strike_far_from_target").any()
    assert quotable(plan).empty


def test_buckets_are_a_comparison_range_not_a_ranking() -> None:
    chain = select_chain(_chain_params(), "SPY")
    plan = plan_candidates(
        "SPY", chain, spot=770.0, as_of=AS_OF, buckets=(ExpiryBucket("wide", 20, 200),)
    )
    # No preference column exists to sort a "best" expiry by.
    assert "rank" not in plan.columns and "score" not in plan.columns
    assert set(plan["bucket"]) == {"wide"}

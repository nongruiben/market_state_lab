"""Parity is the one quote test that needs no model and no second data source,
so it has to actually bite - and has to stay quiet inside the quoted spread."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_state_lab.quote_checks import parity_verdict, put_call_parity_check

DAYS = {"20261009": 31, "20261120": 73}


def _chain(
    expiry: str,
    strikes: list[float],
    forward: float,
    rate: float,
    put_spread: float = 0.04,
    call_spread: float = 0.04,
    put_nudge: dict[float, float] | None = None,
) -> pd.DataFrame:
    """A coherent chain, built from parity so it is right by construction.

    Put prices come from a crude convex curve and the calls are derived through
    parity, which is the point: the test data satisfies the identity exactly, so
    anything the check flags came from the nudge, not from the fixture.
    """
    discount = float(np.exp(-rate * DAYS[expiry] / 365.0))
    rows = []
    for strike in strikes:
        put = max(0.5, 0.06 * (strike - (forward - 90.0)))
        # The call is derived from the *un-nudged* put, so the nudge lands on one
        # leg only. Nudging first and deriving after would preserve parity exactly
        # and the fixture would quietly stop testing anything.
        call = put + discount * (forward - strike)
        put += (put_nudge or {}).get(strike, 0.0)
        for right, mid, spread in (("P", put, put_spread), ("C", call, call_spread)):
            rows.append(
                {
                    "expiry": expiry,
                    "strike": strike,
                    "right": right,
                    "bid": mid - spread / 2.0,
                    "ask": mid + spread / 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_a_coherent_chain_passes_and_recovers_its_own_rate_and_forward() -> None:
    quotes = _chain("20261120", [695.0, 730.0, 755.0], forward=774.4, rate=0.031)
    check = put_call_parity_check(quotes, DAYS).iloc[0]
    assert check["status"] == "ok"
    assert check["pairs"] == 3
    assert check["implied_forward"] == pytest.approx(774.4, abs=1e-6)
    assert check["implied_rate"] == pytest.approx(0.031, abs=1e-9)
    assert parity_verdict(put_call_parity_check(quotes, DAYS)) == "ok"


def test_a_residual_inside_the_quoted_spread_is_not_a_violation() -> None:
    # Deep ITM calls really do quote three points wide; a mid is only known to
    # within half a spread, so a small residual against a wide market is silence.
    quotes = _chain(
        "20261120",
        [695.0, 730.0, 755.0],
        forward=774.4,
        rate=0.031,
        call_spread=3.2,
        put_nudge={730.0: 0.14},
    )
    check = put_call_parity_check(quotes, DAYS).iloc[0]
    assert check["status"] == "ok"
    assert check["max_abs_residual"] < check["worst_residual_tolerance"]


def test_the_same_residual_against_a_tight_market_is_a_violation() -> None:
    quotes = _chain(
        "20261120",
        [695.0, 730.0, 755.0],
        forward=774.4,
        rate=0.031,
        call_spread=0.04,
        put_nudge={730.0: 0.14},
    )
    check = put_call_parity_check(quotes, DAYS).iloc[0]
    assert check["status"] == "parity_violated"
    assert check["max_abs_residual"] > check["worst_residual_tolerance"]
    assert parity_verdict(put_call_parity_check(quotes, DAYS)) == "failed"


def test_two_pairs_fit_a_line_exactly_so_the_check_says_it_proved_nothing() -> None:
    quotes = _chain("20261009", [695.0, 755.0], forward=770.7, rate=0.016)
    check = put_call_parity_check(quotes, DAYS).iloc[0]
    assert check["pairs"] == 2
    assert check["status"] == "slope_only"
    # A clean slope on two points must not be reported as a verified chain.
    assert parity_verdict(put_call_parity_check(quotes, DAYS)) == "unverified"


def test_mismatched_legs_show_up_as_a_broken_slope_not_a_mispricing() -> None:
    quotes = _chain("20261120", [695.0, 730.0, 755.0], forward=774.4, rate=0.031)
    # A wrong multiplier on the call side: the difference is no longer a parity
    # basis at all, and no residual tolerance should rescue it.
    calls = quotes["right"].eq("C")
    quotes.loc[calls, ["bid", "ask"]] *= 2.0
    check = put_call_parity_check(quotes, DAYS).iloc[0]
    assert check["status"] == "slope_out_of_band"
    assert parity_verdict(put_call_parity_check(quotes, DAYS)) == "failed"


def test_an_unpaired_strike_is_ignored_rather_than_guessed_at() -> None:
    quotes = _chain("20261120", [695.0, 730.0, 755.0], forward=774.4, rate=0.031)
    quotes = quotes.loc[~(quotes["strike"].eq(730.0) & quotes["right"].eq("C"))]
    check = put_call_parity_check(quotes, DAYS).iloc[0]
    assert check["pairs"] == 2
    assert check["status"] == "slope_only"


def test_a_puts_only_quote_set_cannot_be_parity_checked() -> None:
    quotes = _chain("20261120", [695.0, 730.0, 755.0], forward=774.4, rate=0.031)
    check = put_call_parity_check(quotes.loc[quotes["right"].eq("P")], DAYS).iloc[0]
    assert check["status"] == "insufficient_pairs"
    assert check["pairs"] == 0
    # This is the default state of the protection table if calls are not fetched,
    # and it must never read as a pass.
    assert parity_verdict(put_call_parity_check(quotes.loc[quotes["right"].eq("P")], DAYS)) == (
        "unverified"
    )


def test_no_quotes_at_all_is_unverified_not_ok() -> None:
    assert put_call_parity_check(pd.DataFrame(), DAYS).empty
    assert parity_verdict(pd.DataFrame()) == "unverified"


def test_each_expiry_is_judged_on_its_own() -> None:
    good = _chain("20261120", [695.0, 730.0, 755.0], forward=774.4, rate=0.031)
    bad = _chain(
        "20261009",
        [695.0, 732.0, 755.0],
        forward=770.7,
        rate=0.016,
        put_nudge={732.0: 0.9},
    )
    checks = put_call_parity_check(pd.concat([good, bad], ignore_index=True), DAYS)
    by_expiry = checks.set_index("expiry")["status"]
    assert by_expiry["20261120"] == "ok"
    assert by_expiry["20261009"] == "parity_violated"
    assert parity_verdict(checks) == "failed"

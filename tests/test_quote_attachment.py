"""A candidate that cannot be priced must survive into the table as a marked row.

Dropping it would leave a comparison silently missing one of its alternatives,
which reads as "these are the choices" when it is really "these are the ones
that happened to quote".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from market_state_lab.defense_tools import attach_contracts, attach_quotes, put_quotes


@dataclass
class FakeContract:
    conId: int  # noqa: N815 - mirrors the ib_async attribute name
    lastTradeDateOrContractMonth: str  # noqa: N815
    strike: float
    localSymbol: str = ""  # noqa: N815


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "expiry": "20261009",
                "strike": 755.0,
                "right": "P",
                "multiplier": "100",
                "status": "ok",
            },
            {
                "symbol": "SPY",
                "expiry": "20261009",
                "strike": 732.0,
                "right": "P",
                "multiplier": "100",
                "status": "ok",
            },
            {
                "symbol": "SPY",
                "expiry": "20261120",
                "strike": 693.0,
                "right": "P",
                "multiplier": "100",
                "status": "strike_far_from_target",
            },
        ]
    )


def _quote(con_id: int, bid, ask, **over) -> dict:
    row = {
        "con_id": con_id,
        "bid": bid,
        "ask": ask,
        "status": "complete",
        "actual_market_data_type_name": "delayed_frozen",
        "quote_age_seconds": 4.0,
        "implied_volatility": 0.1477,
        "delta": -0.3537,
        "underlying_price": 769.73,
    }
    row.update(over)
    return row


def test_contracts_attach_by_expiry_and_strike() -> None:
    attached = attach_contracts(
        _candidates(),
        [
            FakeContract(917830498, "20261009", 755.0, "SPY   261009P00755000"),
            FakeContract(918936749, "20261009", 732.0),
        ],
    )
    assert attached.loc[0, "con_id"] == 917830498
    assert attached.loc[0, "local_symbol"] == "SPY   261009P00755000"
    assert attached.loc[1, "con_id"] == 918936749
    # A row that was already rejected upstream is passed through untouched.
    assert attached.loc[2, "status"] == "strike_far_from_target"


def test_a_candidate_qualification_dropped_is_marked_not_silently_kept() -> None:
    attached = attach_contracts(
        _candidates(), [FakeContract(917830498, "20261009", 755.0)]
    )
    assert attached.loc[1, "status"] == "not_qualified"
    # pandas stores the absent id as NaN in a float column; every consumer here
    # tests it with pd.notna rather than `is None`.
    assert pd.isna(attached.loc[1, "con_id"])


def test_a_two_sided_quote_prices_the_row() -> None:
    attached = attach_contracts(
        _candidates(),
        [
            FakeContract(917830498, "20261009", 755.0),
            FakeContract(918936749, "20261009", 732.0),
        ],
    )
    priced = attach_quotes(
        attached,
        pd.DataFrame([_quote(917830498, 6.42, 6.45), _quote(918936749, 3.02, 3.05)]),
    )
    assert list(priced.loc[:1, "status"]) == ["priced", "priced"]
    assert priced.loc[0, "spread"] == pytest.approx(0.03)
    assert priced.loc[0, "relative_spread"] == pytest.approx(0.03 / 6.45)
    assert priced.loc[0, "market_data_type"] == "delayed_frozen"
    assert priced.loc[0, "implied_volatility"] == 0.1477


def test_a_leg_with_no_ask_stays_in_the_table_and_says_why() -> None:
    attached = attach_contracts(
        _candidates(),
        [
            FakeContract(917830498, "20261009", 755.0),
            FakeContract(918936749, "20261009", 732.0),
        ],
    )
    priced = attach_quotes(
        attached,
        pd.DataFrame(
            [
                _quote(917830498, 6.42, None, status="close_only"),
                _quote(918936749, np.nan, 3.05),
            ]
        ),
    )
    # No ask means no buyer's price, so the row cannot be costed - but it is the
    # finding, not a row to delete.
    assert priced.loc[0, "status"] == "no_ask"
    assert len(priced) == 3
    # A missing bid still leaves a buyable ask; only the spread is unknowable.
    assert priced.loc[1, "status"] == "priced"
    assert pd.isna(priced.loc[1, "spread"])
    assert pd.isna(priced.loc[1, "relative_spread"])


def test_a_contract_that_returned_no_quote_at_all_is_marked() -> None:
    attached = attach_contracts(
        _candidates(),
        [
            FakeContract(917830498, "20261009", 755.0),
            FakeContract(918936749, "20261009", 732.0),
        ],
    )
    priced = attach_quotes(attached, pd.DataFrame([_quote(917830498, 6.42, 6.45)]))
    assert priced.loc[1, "status"] == "no_quote_returned"


def test_put_quotes_carries_only_priced_rows_and_their_provenance() -> None:
    attached = attach_contracts(
        _candidates(),
        [
            FakeContract(917830498, "20261009", 755.0),
            FakeContract(918936749, "20261009", 732.0),
        ],
    )
    priced = attach_quotes(
        attached,
        pd.DataFrame([_quote(917830498, 6.42, 6.45), _quote(918936749, None, None)]),
    )
    quotes = put_quotes(priced)
    assert len(quotes) == 1
    put = quotes[0]
    assert (put.symbol, put.expiry, put.strike) == ("SPY", "20261009", 755.0)
    assert put.multiplier == 100
    assert put.con_id == 917830498
    assert put.market_data_type == "delayed_frozen"
    assert put.spread == pytest.approx(0.03)


def test_a_one_sided_market_gives_a_nan_spread_not_a_zero_bid() -> None:
    attached = attach_contracts(_candidates(), [FakeContract(917830498, "20261009", 755.0)])
    priced = attach_quotes(attached, pd.DataFrame([_quote(917830498, None, 6.45)]))
    put = put_quotes(priced)[0]
    assert math.isnan(put.bid)
    # A zero bid would make the spread read 6.45 - a liquidity claim nobody made.
    assert math.isnan(put.spread)


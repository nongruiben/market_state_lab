"""The fault-injection matrix, plan section 16.1, against the layers that exist.

Each row injects a fault into recorded data and pins the behaviour the plan
demands. Rows whose guarding layer is not built yet are listed at the bottom
with the module that owns them - they are the remaining scope of this phase,
and this file is the checklist that says so. A row listed as unbuilt must not
be claimed as covered; that is exactly the over-claiming this phase exists to
stop.

The injection target is what the previous layer now produces: raw payloads with
a request record each, from which the pipeline is rebuilt. Live connections are
not mutated - a fault is injected into a recorded run, never into TWS.
"""

from __future__ import annotations

import pandas as pd
import pytest

from market_state_lab.data.snapshots import default_eligibility
from market_state_lab.defense_tools import (
    DEGRADED,
    UNAVAILABLE,
    VALID,
    ScreenLimits,
    data_status,
    screen_candidates,
)
from market_state_lab.quote_checks import staleness_note
from market_state_lab.timeutils import market_is_open

LIMITS = ScreenLimits()


def _priced(**over) -> dict:
    row = {
        "status": "priced",
        "bid": 7.65,
        "ask": 7.68,
        "relative_spread": 0.0039,
        "open_interest": 9695.0,
        "quote_age_seconds": None,
    }
    row.update(over)
    return row


def _feed(type_name: str, tick_lag: float = 9.03, quote_age=None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "actual_market_data_type_name": type_name,
                "tick_lag_seconds": tick_lag,
                "quote_age_seconds": quote_age,
            }
        ]
    )


# ---------------------------------------------------------------------------
# Row 4: delayed return, frozen post-close price -> the actual type is reported
# and the data does not enter a real-time instrument recommendation.
# ---------------------------------------------------------------------------


def test_frozen_data_is_reported_as_what_it_is() -> None:
    note = staleness_note(_feed("delayed_frozen"), as_of="2026-09-08T07:59:39+00:00")
    assert note["basis"] == "frozen_last_session"
    assert "frozen book" in note["note"]


def test_a_frozen_book_while_the_session_trades_is_degraded() -> None:
    out = screen_candidates(
        pd.DataFrame([_priced()]), LIMITS, quote_basis="frozen_last_session", market_open=True
    )
    assert out.loc[0, "quote_qualification"] == DEGRADED


def test_one_degraded_row_removes_the_realtime_recommendation_eligibility() -> None:
    # The demotion must surface as an eligibility refusal, not as a shorter
    # table: a comparison silently missing one leg reads as "these are the
    # choices".
    eligible, ineligible = default_eligibility({"VALID", "DEGRADED"})
    assert "instrument_quotes" not in eligible
    assert "DEGRADED" in ineligible["instrument_quotes"]


def test_a_working_screen_does_not_count_as_a_data_fault() -> None:
    # A quarantined contract is the screen succeeding: every field arrived and
    # the instrument was confidently rejected as too thin. If that demoted the
    # snapshot, the better the screen got the less the data would be trusted.
    eligible, _ = default_eligibility({"VALID", "QUARANTINED"})
    assert "instrument_quotes" in eligible


def test_an_unavailable_row_still_demotes_the_snapshot() -> None:
    eligible, ineligible = default_eligibility({"VALID", "UNAVAILABLE"})
    assert "instrument_quotes" not in eligible
    assert "UNAVAILABLE" in ineligible["instrument_quotes"]


def test_a_fully_valid_book_keeps_its_recommendation_eligibility() -> None:
    eligible, _ = default_eligibility({"VALID"})
    assert "instrument_quotes" in eligible


# ---------------------------------------------------------------------------
# Row 5: a bid-only snapshot / an incomplete response -> partial state blocks
# the purpose it cannot serve.
# ---------------------------------------------------------------------------


def test_a_bid_only_market_cannot_be_costed_and_says_so() -> None:
    # The fault enters the real chain: a bid with no ask comes out of the quote
    # attachment as `no_ask` - it is never `priced`, because a buyer's price is
    # the one thing costing needs - and the screen then keeps it out entirely.
    from market_state_lab.defense_tools import attach_quotes

    candidates = pd.DataFrame(
        [
            {
                "status": "ok", "expiry": "20261120", "strike": 730.0,
                "con_id": 882140303, "multiplier": "100", "symbol": "SPY",
            }
        ]
    )
    attached = attach_quotes(
        candidates,
        pd.DataFrame(
            [{"con_id": 882140303, "bid": 6.42, "ask": None, "status": "close_only"}]
        ),
    )
    assert attached.loc[0, "status"] == "no_ask"
    out = screen_candidates(attached, LIMITS, quote_basis="frozen_last_session", market_open=False)
    assert out.loc[0, "quote_qualification"] == UNAVAILABLE
    assert not out.loc[0, "screened"]


def test_no_data_is_distinguished_from_data_that_failed_the_screen() -> None:
    # Row 10 depends on this split: "nothing arrived" must read DATA_INSUFFICIENT,
    # never "the controls are the comparison".
    never_priced = pd.DataFrame([{"status": "no_ask"}, {"status": "no_quote_returned"}])
    priced_but_rejected = pd.DataFrame([{"status": "priced"}, {"status": "priced"}])
    assert data_status(never_priced) == "data_insufficient"
    assert data_status(priced_but_rejected) == "screened_out"
    assert data_status(pd.DataFrame()) == "no_data"


# ---------------------------------------------------------------------------
# Row 6: a date one day off, DST, a half-day -> the calendar, not the clock,
# decides which session a frozen book belongs to.
# ---------------------------------------------------------------------------


def test_a_holiday_is_not_a_session() -> None:
    # 2026-09-07 was Labor Day. A frozen book fetched the next morning belongs
    # to Friday's close, and the age is measured from Friday.
    note = staleness_note(_feed("delayed_frozen"), as_of="2026-09-08T07:59:39+00:00")
    assert note["last_completed_session"] == "2026-09-04"
    assert note["age_hours"] == pytest.approx(83.99, abs=0.05)


def test_the_open_state_comes_from_the_exchange_calendar() -> None:
    from datetime import datetime, timezone

    assert not market_is_open(now=datetime(2026, 9, 7, 18, 0, tzinfo=timezone.utc))  # holiday
    assert not market_is_open(now=datetime(2026, 9, 4, 23, 0, tzinfo=timezone.utc))  # after close
    assert market_is_open(now=datetime(2026, 9, 4, 18, 0, tzinfo=timezone.utc))  # in session


# ---------------------------------------------------------------------------
# Row 7: late callbacks, reconnect, concurrent requests -> never written into
# the current request.
# ---------------------------------------------------------------------------


def test_two_connections_are_two_generations_with_distinct_ids(tmp_path) -> None:
    from market_state_lab.data.ibkr import ArchiveSink

    old = ArchiveSink(tmp_path, session_tag="c917-20260908T100000Z")
    stale_record = old.begin("reqMktData")
    new = ArchiveSink(tmp_path, session_tag="c917-20260908T100500Z")
    current = new.begin("reqMktData")
    assert stale_record.request_id != current.request_id
    # A callback from the old connection completing late must not appear in the
    # new connection's records.
    old.complete(stale_record, [{"bid": 7.65}])
    assert len(new.records) == 1 and new.records[0].request_id == current.request_id


def test_a_failed_record_cannot_be_confused_with_a_completed_one(tmp_path) -> None:
    from market_state_lab.data.ibkr import ArchiveSink

    sink = ArchiveSink(tmp_path, session_tag="g")
    ok = sink.begin("reqMktData")
    sink.complete(ok, [{"bid": 1.0}], rows=1)
    dead = sink.begin("reqMktData")
    sink.fail(dead, "Error 354: not subscribed")
    states = {r.request_id: r.state for r in sink.records}
    assert states[ok.request_id] == "complete"
    assert states[dead.request_id] == "failed"


# ---------------------------------------------------------------------------
# Row 8: same-day history duplicated with different values -> a revision
# conflict, never a silent overwrite.
# ---------------------------------------------------------------------------


def test_same_day_different_bytes_is_a_revision_conflict(tmp_path) -> None:
    from market_state_lab.data.snapshots import RawArchive, RequestRecord

    archive = RawArchive(tmp_path)
    record = RequestRecord("req-1", "ibkr", "g", "reqMktData")
    archive.write(record, {"bid": 7.65}, "2026-09-04")
    with pytest.raises(FileExistsError, match="revision"):
        archive.write(record, {"bid": 7.20}, "2026-09-04")


# ---------------------------------------------------------------------------
# Row 10: all critical inputs fail -> DATA_INSUFFICIENT, not "safe" and not
# "de-risk". An empty quote set can never be approved for anything.
# ---------------------------------------------------------------------------


def test_no_data_grants_no_instrument_eligibility() -> None:
    eligible, ineligible = default_eligibility(set())
    assert "instrument_quotes" not in eligible
    assert "no data" in ineligible["instrument_quotes"]


def test_even_the_day_end_grant_still_refuses_training_and_intraday() -> None:
    eligible, ineligible = default_eligibility({"VALID"})
    assert "training" not in eligible and "intraday_observation" not in eligible
    assert "series" in ineligible["training"]


# ---------------------------------------------------------------------------
# Replay: the same recorded inputs must produce the same verdicts. This is the
# "valid-data replay" half of the phase gate, at the unit level - the live half
# runs against a recorded TWS session.
# ---------------------------------------------------------------------------


def test_replaying_the_same_recorded_plan_reproduces_the_same_screen(tmp_path) -> None:
    from market_state_lab.data.snapshots import read_snapshot, verify_snapshot, write_snapshot

    config = {"project": {"market_calendar": "XNYS"}}
    plan = pd.DataFrame(
        [
            {**_priced(), "expiry": "20261120", "strike": 730.0,
             "quote_qualification": VALID, "screened": True},
        ]
    )
    first = write_snapshot(tmp_path, "2026-09-04", {"plan": plan}, config)
    loaded = read_snapshot(tmp_path, first.snapshot_id)
    assert verify_snapshot(loaded) == []
    replayed = screen_candidates(
        loaded.table("plan"), LIMITS, quote_basis="frozen_last_session", market_open=False
    )
    assert replayed.loc[0, "quote_qualification"] == VALID
    assert replayed.loc[0, "screened"]


# ---------------------------------------------------------------------------
# Unbuilt rows. They stay listed until their layer lands:
#
# 1  prices x100, inverted OHLC   -> data/validation.py (6.2): finite positive
#                                   OHLC with high/low bounds; a quote set x100
#                                   must be quarantined, never a market-crisis
#                                   signal.
# 2  real crash confirmed by a    -> data/reconciliation.py (6.4): a crash the
#    second source                    second source confirms is kept; no outlier
#                                     filter may delete risk.
# 3  split/dividend口径 mismatch  -> data/validation.py (6.3): corporate actions
#                                   recorded with effective_at and known_at.
# 9  multi-source common anomaly -> data/reconciliation.py (6.4): a common
#                                   anomaly is not passed by majority vote.
# 11 historical pollution repair -> features/models: dependent features and
#                                   state invalidated and replayed; original
#                                   report kept.
# 12 low-frequency alert state    -> market_assessment.py: restarting the state
#    machine restart                 machine does not re-send or drop events.
# ---------------------------------------------------------------------------

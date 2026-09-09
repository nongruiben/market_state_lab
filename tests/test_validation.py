"""Section 6.2, and the restraint that matters most: a filter tuned to remove
bad prints removes crashes too, so suspicious data goes to review and only the
structurally impossible is quarantined."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_state_lab.data.validation import (
    DAY_END,
    NOTE,
    QUARANTINE,
    REVIEW,
    CorporateAction,
    QualityIssue,
    attribute_moves,
    blocked_purposes,
    crisis_reading_is_allowed,
    detect_stale_series,
    drop_quarantined,
    quarantined_subjects,
    summarise,
    validate_bar_completeness,
    validate_daily_bars,
    validate_quotes,
)


def _bars(closes: list[float], **over) -> pd.DataFrame:
    index = pd.date_range("2026-09-01", periods=len(closes), freq="D")
    frame = pd.DataFrame(
        {
            "open": [c * 0.99 for c in closes],
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.98 for c in closes],
            "close": closes,
            "volume": [1_000_000.0] * len(closes),
        },
        index=index,
    )
    for key, value in over.items():
        frame[key] = value
    frame.attrs["volume_unit"] = "shares"
    return frame


def _codes(issues) -> list[str]:
    return [i.code for i in issues]


def test_a_clean_series_raises_nothing_that_blocks() -> None:
    issues = validate_daily_bars(_bars([770.0, 771.0, 769.0]), "SPY")
    assert blocked_purposes(issues) == set()
    assert crisis_reading_is_allowed(issues)


# ---------------------------------------------------------------------------
# Fault-injection row 1: prices x100 and inverted OHLC are quarantined, and no
# market-crisis reading may be formed from them.
# ---------------------------------------------------------------------------


def test_a_hundredfold_price_is_a_units_error_not_a_crash() -> None:
    issues = validate_daily_bars(_bars([770.0, 77000.0, 771.0]), "SPY")
    assert "price_scale_break" in _codes(issues)
    scale = next(i for i in issues if i.code == "price_scale_break")
    assert scale.severity == QUARANTINE
    assert "factor of 100" in scale.detail
    # And the bar that came back down is a scale break too, not a second crash.
    assert _codes(issues).count("price_scale_break") == 2
    assert not crisis_reading_is_allowed(issues)


def test_an_inverted_bar_describes_no_market() -> None:
    frame = _bars([770.0, 771.0])
    frame.loc[frame.index[1], ["high", "low"]] = [700.0, 800.0]
    issues = validate_daily_bars(frame, "SPY")
    inverted = next(i for i in issues if i.code == "ohlc_inverted")
    assert inverted.severity == QUARANTINE
    assert not crisis_reading_is_allowed(issues)


def test_a_close_outside_its_own_range_is_quarantined() -> None:
    frame = _bars([770.0, 771.0])
    frame.loc[frame.index[1], "close"] = 900.0
    issues = validate_daily_bars(frame, "SPY")
    assert "ohlc_bounds_violated" in _codes(issues)
    assert not crisis_reading_is_allowed(issues)


def test_quarantined_rows_leave_the_input_but_keep_their_reason() -> None:
    frame = _bars([770.0, 77000.0, 771.0])
    issues = validate_daily_bars(frame, "SPY")
    kept = drop_quarantined(frame, issues, "SPY")
    assert len(kept) < len(frame)
    # Removed from what goes downstream, never from the record of why.
    assert quarantined_subjects(issues)
    assert all(i.detail for i in issues)


# ---------------------------------------------------------------------------
# Row 2's restraint, enforced here in advance: a real crash must survive.
# ---------------------------------------------------------------------------


def test_a_real_crash_goes_to_review_and_is_never_deleted() -> None:
    # -34% in a session is 1987. It is exactly the event this project exists to
    # notice, and a z-score filter would erase it.
    issues = validate_daily_bars(_bars([770.0, 508.0, 520.0]), "SPY")
    crash = next(i for i in issues if i.code == "large_price_move")
    assert crash.severity == REVIEW
    assert crash.blocks == ()
    assert crisis_reading_is_allowed(issues)
    frame = _bars([770.0, 508.0, 520.0])
    assert len(drop_quarantined(frame, issues, "SPY")) == 3


def test_an_ordinary_move_is_not_even_reviewed() -> None:
    issues = validate_daily_bars(_bars([770.0, 755.0, 762.0]), "SPY")
    assert "large_price_move" not in _codes(issues)


def test_a_halving_is_a_move_and_a_hundredfold_is_not() -> None:
    # A market can halve. It does not move by exactly 100x.
    halved = validate_daily_bars(_bars([770.0, 385.0]), "SPY")
    assert _codes(halved) == ["large_price_move"]
    scaled = validate_daily_bars(_bars([770.0, 7700.0]), "SPY")
    assert "price_scale_break" in _codes(scaled)


# ---------------------------------------------------------------------------
# Duplicates, volume, and security types
# ---------------------------------------------------------------------------


def test_identical_repeats_are_a_note_and_disagreeing_ones_are_a_conflict() -> None:
    clean = _bars([770.0, 771.0])
    doubled = pd.concat([clean, clean.iloc[[1]]])
    assert "duplicate_row_identical" in _codes(validate_daily_bars(doubled, "SPY"))

    conflicting = clean.iloc[[1]].copy()
    conflicting["close"] = 900.0
    conflicting[["high", "low"]] = [901.0, 899.0]
    disagreeing = pd.concat([clean, conflicting])
    issues = validate_daily_bars(disagreeing, "SPY")
    conflict = next(i for i in issues if i.code == "duplicate_row_conflict")
    assert conflict.severity == QUARANTINE
    assert "not a row to keep the last of" in conflict.detail


def test_an_index_has_no_volume_and_that_is_not_a_defect() -> None:
    index_bars = _bars([5000.0, 5010.0])
    index_bars["volume"] = 0.0
    assert "volume_zero" not in _codes(validate_daily_bars(index_bars, "SPX", sec_type="IND"))
    # The same zero on a share listing is worth seeing, and still not a rejection.
    stock = validate_daily_bars(index_bars, "THIN", sec_type="STK")
    zero = next(i for i in stock if i.code == "volume_zero")
    assert zero.severity == NOTE and zero.blocks == ()


def test_absent_volume_is_not_a_zero() -> None:
    frame = _bars([770.0, 771.0])
    frame.loc[frame.index[1], "volume"] = np.nan
    missing = next(i for i in validate_daily_bars(frame, "SPY") if i.code == "volume_missing")
    assert "absent is not a zero" in missing.detail


def test_an_undeclared_volume_unit_is_recorded() -> None:
    frame = _bars([770.0])
    frame.attrs.pop("volume_unit")
    assert "volume_unit_undeclared" in _codes(validate_daily_bars(frame, "SPY"))


def test_a_negative_price_is_impossible_for_a_share_but_the_rule_is_typed() -> None:
    frame = _bars([770.0, 771.0])
    frame.loc[frame.index[1], "low"] = -1.0
    assert "price_not_positive" in _codes(validate_daily_bars(frame, "SPY", sec_type="STK"))
    # A spread or a rate series may legitimately go negative, so the same rule
    # must not be applied to it.
    assert "price_not_positive" not in _codes(
        validate_daily_bars(frame, "BAA10Y", sec_type="IND")
    )


# ---------------------------------------------------------------------------
# Quote-level checks
# ---------------------------------------------------------------------------


def _quote(**over) -> dict:
    row = {"symbol": "SPY", "expiry": "20261120", "strike": 730.0,
           "bid": 7.65, "ask": 7.68, "last": 7.66}
    row.update(over)
    return row


def test_a_crossed_book_is_reviewed_not_rejected_on_sight() -> None:
    # The two sides of a stream update independently; a momentary inversion is
    # a sampling artefact, and only persistence justifies rejection.
    issues = validate_quotes(pd.DataFrame([_quote(bid=7.70, ask=7.65, last=7.66)]))
    crossed = next(i for i in issues if i.code == "crossed_book")
    assert crossed.severity == REVIEW and crossed.blocks == ()


def test_a_zero_bid_can_be_real() -> None:
    issues = validate_quotes(pd.DataFrame([_quote(bid=0.0, last=0.01)]))
    zero = next(i for i in issues if i.code == "zero_bid")
    assert zero.severity == NOTE


def test_a_last_outside_the_book_is_not_corrected() -> None:
    issues = validate_quotes(pd.DataFrame([_quote(last=7.20)]))
    outside = next(i for i in issues if i.code == "last_outside_book")
    assert outside.severity == NOTE
    assert "never a reason to correct last" in outside.detail


def test_a_sentinel_implied_vol_is_not_a_number() -> None:
    issues = validate_quotes(pd.DataFrame([_quote(implied_volatility=-1.0)]))
    assert "implied_vol_sentinel" in _codes(issues)


# ---------------------------------------------------------------------------
# The shape of the output
# ---------------------------------------------------------------------------


def test_the_summary_is_counts_and_losses_not_a_score() -> None:
    issues = validate_daily_bars(_bars([770.0, 77000.0]), "SPY")
    summary = summarise(issues)
    assert set(summary) == {
        "counts", "blocked_purposes", "quarantined_subjects", "crisis_reading_allowed"
    }
    assert DAY_END in summary["blocked_purposes"]
    assert summary["crisis_reading_allowed"] is False
    # No single number that could average an impossible bar with a wide spread.
    assert not any(isinstance(v, float) for v in summary.values())


def test_a_note_may_not_block_anything() -> None:
    with pytest.raises(ValueError, match="a note blocks nothing"):
        QualityIssue("x", NOTE, "SPY", "detail", (DAY_END,))


def test_an_unknown_purpose_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown purposes"):
        QualityIssue("x", QUARANTINE, "SPY", "detail", ("trading",))


def test_a_frozen_book_collapses_the_last_outside_note_to_one_line() -> None:
    # On a frozen feed every leg trips this, and seven identical notes on
    # healthy data is how a quality log teaches its reader to skip it.
    legs = pd.DataFrame([_quote(strike=s, last=1.0) for s in (690.0, 730.0, 750.0)])
    noisy = validate_quotes(legs, frozen_book=False)
    assert _codes(noisy).count("last_outside_book") == 3
    quiet = validate_quotes(legs, frozen_book=True)
    assert "last_outside_book" not in _codes(quiet)
    collapsed = next(i for i in quiet if i.code == "last_outside_frozen_book")
    assert "3 of 3 legs" in collapsed.detail
    assert collapsed.blocks == ()


def test_the_collapsed_note_is_absent_when_nothing_trips_it() -> None:
    inside = pd.DataFrame([_quote(last=7.66)])
    assert validate_quotes(inside, frozen_book=True) == []


# ---------------------------------------------------------------------------
# Corporate actions: fault-injection row 3
# ---------------------------------------------------------------------------


def _split(effective: str, known: str, ratio: float = 4.0) -> CorporateAction:
    return CorporateAction("SPY", "split", pd.Timestamp(effective), pd.Timestamp(known), ratio)


def test_a_split_on_record_explains_the_gap_it_caused() -> None:
    # A 4-for-1 quarters the price. The move is real and its cause is known.
    frame = _bars([800.0, 200.0, 201.0])
    moves = validate_daily_bars(frame, "SPY")
    assert "large_price_move" in _codes(moves)
    explained = attribute_moves(moves, [_split("2026-09-02", "2026-08-20")])
    assert "large_price_move" not in _codes(explained)
    cause = next(i for i in explained if i.code == "move_explained_by_corporate_action")
    assert cause.severity == NOTE
    assert "only its attribution is" in cause.detail


def test_a_split_announced_afterwards_explains_nothing_yet() -> None:
    # Attributing today's gap with tomorrow's announcement is the quietest way
    # to build a series that was never tradeable.
    frame = _bars([800.0, 200.0])
    moves = validate_daily_bars(frame, "SPY")
    late = _split("2026-09-02", known="2026-09-05")
    still = attribute_moves(moves, [late], known_by=pd.Timestamp("2026-09-02"))
    assert "large_price_move" in _codes(still)
    # Once it is public, the same action explains the same gap.
    now = attribute_moves(moves, [late], known_by=pd.Timestamp("2026-09-06"))
    assert "move_explained_by_corporate_action" in _codes(now)


def test_an_action_of_the_wrong_size_does_not_explain_the_move() -> None:
    frame = _bars([800.0, 200.0])
    moves = validate_daily_bars(frame, "SPY")
    two_for_one = _split("2026-09-02", "2026-08-20", ratio=2.0)
    assert "large_price_move" in _codes(attribute_moves(moves, [two_for_one]))


def test_an_unexplained_move_keeps_its_review_status() -> None:
    frame = _bars([770.0, 508.0])
    moves = validate_daily_bars(frame, "SPY")
    assert attribute_moves(moves, []) == moves


def test_a_corporate_action_carries_both_of_its_dates() -> None:
    action = _split("2026-09-02", "2026-08-20")
    assert action.effective_at != action.known_at
    with pytest.raises(ValueError, match="unknown corporate action"):
        CorporateAction("SPY", "buyback", pd.Timestamp("2026-09-02"), pd.Timestamp("2026-09-02"))


# ---------------------------------------------------------------------------
# 6.1 remainder: an unfinished bar, and a series that stopped moving
# ---------------------------------------------------------------------------


def test_an_unfinished_bar_cannot_settle_a_day_end_label() -> None:
    frame = _bars([770.0, 771.0])
    assert validate_bar_completeness(frame, "SPY", session_complete=True) == []
    open_session = validate_bar_completeness(frame, "SPY", session_complete=False)
    assert open_session[0].severity == QUARANTINE
    assert DAY_END in open_session[0].blocks
    assert "still moving" in open_session[0].detail


def test_a_series_that_stopped_moving_is_reviewed_not_rejected() -> None:
    # A halt, a thin listing and a frozen feed look identical from here.
    stuck = pd.Series([100.0] * 8, index=pd.date_range("2026-09-01", periods=8))
    issue = detect_stale_series(stuck, "THIN")[0]
    assert issue.severity == REVIEW
    assert issue.blocks == ()
    assert "check activity and a second source" in issue.detail


def test_a_moving_series_is_not_flagged() -> None:
    moving = pd.Series([100.0, 101.0, 100.5, 102.0, 101.0, 103.0],
                       index=pd.date_range("2026-09-01", periods=6))
    assert detect_stale_series(moving, "SPY") == []

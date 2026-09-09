"""Section 9.3, and fault-injection row 12: a restart must not re-announce what
it already said, and must not lose what has not been released."""

from __future__ import annotations

import pandas as pd

from market_state_lab.events import (
    OPENED,
    RELEASED,
    EventLog,
    EventRules,
    confirmation_cost,
    exceptional_review,
)

RULES = EventRules(confirm_sessions=3, release_sessions=5, cooldown_sessions=0)


def _sessions(count: int) -> list[str]:
    return [str(d.date()) for d in pd.bdate_range("2026-09-01", periods=count)]


def _run(log: EventLog, pattern: list[bool], label: str = "trend_damaged") -> list:
    news = []
    for session, present in zip(_sessions(len(pattern)), pattern):
        news += log.observe([label] if present else [], session)
    return news


# ---------------------------------------------------------------------------
# Confirmation and release, and the asymmetry between them
# ---------------------------------------------------------------------------


def test_a_label_must_hold_before_it_becomes_an_event() -> None:
    log = EventLog(rules=RULES)
    assert _run(log, [True, True]) == []
    assert log.open_events() == []


def test_the_third_consecutive_session_opens_it_and_admits_the_delay() -> None:
    log = EventLog(rules=RULES)
    news = _run(log, [True, True, True])
    assert len(news) == 1 and news[0].kind == OPENED
    # Whatever it was, it was already true before anyone heard.
    assert "already true for 2 of them" in news[0].detail


def test_a_flicker_never_opens_and_is_closed_quietly() -> None:
    log = EventLog(rules=RULES)
    news = _run(log, [True, True, False, False, False, False, False, False])
    assert news == []
    assert log.open_events() == []


def test_release_is_slower_than_entry_so_the_alert_outlives_its_reason() -> None:
    log = EventLog(rules=RULES)
    news = _run(log, [True] * 3 + [False] * 5)
    kinds = [n.kind for n in news]
    assert kinds == [OPENED, RELEASED]
    assert "absent for 5 sessions" in news[1].detail


def test_a_label_returning_inside_the_window_rejoins_its_own_event() -> None:
    # The plan's merge rule, which falls out of release being slower than entry
    # rather than needing a rule of its own.
    log = EventLog(rules=RULES)
    news = _run(log, [True] * 3 + [False, False] + [True] * 3)
    assert [n.kind for n in news] == [OPENED]
    assert len(log.events) == 1
    assert log.open_events()[0].event_id.endswith("2026-09-01")


def test_a_release_says_the_reason_left_and_nothing_about_holdings() -> None:
    log = EventLog(rules=RULES)
    released = [n for n in _run(log, [True] * 3 + [False] * 5) if n.kind == RELEASED][0]
    # This project never reads a position, so it cannot tell anyone to unwind one.
    assert "nothing here reads a position" in released.detail
    assert "sell" not in released.detail.lower()


# ---------------------------------------------------------------------------
# Row 12: restart
# ---------------------------------------------------------------------------


def test_a_restart_does_not_re_announce_what_was_already_said(tmp_path) -> None:
    path = tmp_path / "events.json"
    log = EventLog(rules=RULES)
    assert len(_run(log, [True, True, True])) == 1
    log.save(path)

    # A process that forgets and re-sends is indistinguishable, to the reader,
    # from the market doing it again.
    restarted = EventLog.load(path, RULES)
    again = restarted.observe(["trend_damaged"], "2026-09-04")
    assert again == []


def test_a_restart_does_not_lose_an_unreleased_event(tmp_path) -> None:
    path = tmp_path / "events.json"
    log = EventLog(rules=RULES)
    _run(log, [True, True, True])
    log.save(path)

    restarted = EventLog.load(path, RULES)
    assert [e.event_id for e in restarted.open_events()] == [
        e.event_id for e in log.open_events()
    ]
    # And it still knows how to release the event it inherited.
    news = [
        n
        for session in _sessions(9)[3:]
        for n in restarted.observe([], session)
    ]
    assert [n.kind for n in news] == [RELEASED]


def test_an_absent_store_starts_empty_rather_than_failing(tmp_path) -> None:
    assert EventLog.load(tmp_path / "nothing.json").events == []


# ---------------------------------------------------------------------------
# Cooldown, and the exception path
# ---------------------------------------------------------------------------


def test_cooldown_silences_repetition_about_one_event_not_a_different_one() -> None:
    log = EventLog(rules=EventRules(confirm_sessions=1, release_sessions=5, cooldown_sessions=10))
    first = log.observe(["trend_damaged"], "2026-09-01")
    assert len(first) == 1
    # A different reason is news and does not queue behind the first.
    second = log.observe(["trend_damaged", "volatility_elevated"], "2026-09-02")
    assert [n.label for n in second] == ["volatility_elevated"]


def test_a_verified_move_may_skip_the_wait_and_a_single_print_may_not() -> None:
    assert exceptional_review(True, single_print=False, detail="both sources saw -9%") is not None
    # The unconfirmed single print is precisely what the window exists to absorb.
    assert exceptional_review(True, single_print=True, detail="one tick") is None
    assert exceptional_review(False, single_print=False, detail="unconfirmed") is None


# ---------------------------------------------------------------------------
# The replay cost, reported in both directions
# ---------------------------------------------------------------------------


def test_the_window_reports_what_it_cost_as_well_as_what_it_saved() -> None:
    history = {
        "trend_damaged": [True, False, True, False] + [True] * 6 + [False] * 6,
        "volatility_elevated": [True, True] + [False] * 8,
    }
    cost = confirmation_cost(history, RULES)
    # A window that suppresses every flicker also delays every real event, and
    # quoting only the first number is how a filter gets adopted on half its
    # evidence.
    assert cost["flickers_suppressed"] >= 3
    assert cost["events_opened"] >= 1
    assert cost["worst_delay_sessions"] is not None and cost["worst_delay_sessions"] > 0
    assert "both numbers or neither" in cost["note"]


def test_the_defaults_are_marked_as_unmeasured() -> None:
    assert "unmeasured" in EventRules().grade
    assert confirmation_cost({"x": [True] * 5}, EventRules())["grade"] == EventRules().grade

"""Low-frequency event state: confirmation, release, persistence, and restart.

Split out of `market_assessment` for size; it is the rule-state half of that
module's remit. A label that flickers on and off daily is not an event, and a
reader who is pinged every time one does stops reading. So a label has to hold
for several sessions before an event opens, and has to be absent for longer
before it closes.

The asymmetry is deliberate and is the whole design. Entering costs delay:
whatever the event was, it was already true for `confirm_sessions` before
anyone heard. Leaving costs staleness: the alert stands for
`release_sessions` after the reason has gone. Those are the two prices, they
pull opposite ways, and the plan requires the first replay to measure *both* -
so `confirmation_cost` reports the flickers suppressed and the sessions of
delay added side by side, rather than the flattering half.

Three things this refuses to do:

*It does not re-announce on restart.* Events are persisted with stable ids and
the sessions they were announced on. A process that forgets and re-sends is
indistinguishable, to the person reading, from the market doing it again.

*Cooldown silences repetition, never news.* A second notification about the same
standing event waits. A different event is not delayed behind it.

*It never says "sell your protection".* This project does not read positions, so
it does not know whether anything was bought. A release says the reason has
gone; what to do about that is the reader's, and they are the only one who knows
what they hold.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

OPENED, RELEASED = "opened", "released"


@dataclass(frozen=True)
class EventRules:
    """Pending defaults. 3 and 5 are the plan's starting numbers, not findings.

    They are frozen here so the first replay measures a fixed rule rather than
    one tuned while the results were visible.
    """

    confirm_sessions: int = 3
    release_sessions: int = 5
    cooldown_sessions: int = 5
    grade: str = "pending replay; the delay and false-alarm costs are unmeasured"


@dataclass
class Event:
    """One reason, from the session it was first seen to the session it left."""

    event_id: str
    label: str
    first_seen: str
    last_seen: str
    confirmed_on: str | None = None
    released_on: str | None = None
    absent_sessions: int = 0
    seen_sessions: int = 0
    notified_on: list[str] = field(default_factory=list)

    @property
    def open(self) -> bool:
        return self.confirmed_on is not None and self.released_on is None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Notification:
    """Something worth telling the reader once."""

    kind: str
    event_id: str
    label: str
    session: str
    detail: str


class EventLog:
    """The persisted state machine. Reload, observe, and never repeat yourself."""

    def __init__(self, events: list[Event] | None = None, rules: EventRules = EventRules()):
        self.events = events or []
        self.rules = rules

    # -- persistence ------------------------------------------------------

    @classmethod
    def load(cls, path: Path, rules: EventRules = EventRules()) -> "EventLog":
        path = Path(path)
        if not path.exists():
            return cls([], rules)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls([Event(**e) for e in raw.get("events", [])], rules)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"events": [e.as_dict() for e in self.events]}, indent=2),
            encoding="utf-8",
        )

    # -- state ------------------------------------------------------------

    def active(self, label: str) -> Event | None:
        """The event this label belongs to, if one is still running.

        A label that returns within the release window rejoins its own event
        rather than starting a new one - the plan's merge rule, which falls out
        of release being slower than entry rather than needing a rule of its own.
        """
        for event in reversed(self.events):
            if event.label == label and event.released_on is None:
                return event
        return None

    def open_events(self) -> list[Event]:
        return [e for e in self.events if e.open]

    def observe(self, labels: list[str], session: str) -> list[Notification]:
        """Advance every machine by one session and return what is worth saying."""
        news: list[Notification] = []
        present = set(labels)

        for label in sorted(present):
            event = self.active(label)
            if event is None:
                event = Event(
                    event_id=f"{label}:{session}",
                    label=label,
                    first_seen=session,
                    last_seen=session,
                    seen_sessions=1,
                )
                self.events.append(event)
            else:
                event.last_seen = session
                event.seen_sessions += 1
                event.absent_sessions = 0
            if event.confirmed_on is None and event.seen_sessions >= self.rules.confirm_sessions:
                event.confirmed_on = session
                news.append(
                    Notification(
                        OPENED, event.event_id, label, session,
                        f"{label} has held for {event.seen_sessions} sessions since "
                        f"{event.first_seen}; it was already true for "
                        f"{self.rules.confirm_sessions - 1} of them before this said so",
                    )
                )

        for event in self.events:
            if event.released_on is not None or event.label in present:
                continue
            event.absent_sessions += 1
            if event.absent_sessions < self.rules.release_sessions:
                continue
            event.released_on = session
            if event.confirmed_on is None:
                # Never confirmed, so nothing was ever announced and nothing is
                # withdrawn. It flickered and is closed quietly.
                continue
            news.append(
                Notification(
                    RELEASED, event.event_id, event.label, session,
                    f"{event.label} has been absent for {event.absent_sessions} sessions. "
                    "The reason has gone; this says nothing about what anyone holds, "
                    "because nothing here reads a position",
                )
            )

        return [n for n in news if self._allow(n, session)]

    def _allow(self, notification: Notification, session: str) -> bool:
        """Cooldown suppresses repetition about one event, never a different one."""
        event = next((e for e in self.events if e.event_id == notification.event_id), None)
        if event is None:
            return True
        if event.notified_on:
            last = pd.Timestamp(event.notified_on[-1])
            if (pd.Timestamp(session) - last).days < self.rules.cooldown_sessions:
                return False
        event.notified_on.append(session)
        return True


def exceptional_review(
    verified_move: bool,
    single_print: bool,
    detail: str,
) -> Notification | None:
    """A large, cross-checked move may skip the confirmation wait.

    Only a verified one. A single unconfirmed print is exactly the input the
    confirmation window exists to absorb, and letting it jump the queue would
    hand the fast path to the least reliable observation there is.
    """
    if not verified_move or single_print:
        return None
    return Notification(
        OPENED, "exceptional", "exceptional_review", "",
        f"a verified move justifies looking now rather than waiting for "
        f"confirmation: {detail}",
    )


def confirmation_cost(
    history: dict[str, list[bool]],
    rules: EventRules = EventRules(),
) -> dict[str, Any]:
    """What the confirmation window bought and what it cost, side by side.

    The plan insists the first replay report both. A window that suppresses
    every flicker also delays every real event, and quoting only the suppressed
    count is how a filter gets adopted on half its evidence.
    """
    suppressed = 0
    delays: list[int] = []
    opened = 0
    for label, seen in history.items():
        log = EventLog(rules=EventRules(rules.confirm_sessions, rules.release_sessions, 0))
        first_true: int | None = None
        for index, present in enumerate(seen):
            session = str(pd.Timestamp("2026-01-01") + pd.Timedelta(days=index))
            if present and first_true is None:
                first_true = index
            for note in log.observe([label] if present else [], session):
                if note.kind == OPENED:
                    opened += 1
                    delays.append(index - (first_true if first_true is not None else index))
                    first_true = None
        # Runs that never reached confirmation are the flickers it absorbed.
        runs = _runs(seen)
        suppressed += sum(1 for run in runs if run < rules.confirm_sessions)
    return {
        "confirm_sessions": rules.confirm_sessions,
        "release_sessions": rules.release_sessions,
        "flickers_suppressed": suppressed,
        "events_opened": opened,
        "median_delay_sessions": float(pd.Series(delays).median()) if delays else None,
        "worst_delay_sessions": max(delays) if delays else None,
        "grade": rules.grade,
        "note": (
            "both numbers or neither: the window that removes false alarms is the "
            "same window that delays the true ones"
        ),
    }


def _runs(flags: list[bool]) -> list[int]:
    runs: list[int] = []
    current = 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


__all__ = [
    "OPENED",
    "RELEASED",
    "Event",
    "EventLog",
    "EventRules",
    "Notification",
    "confirmation_cost",
    "exceptional_review",
]

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd


@dataclass(frozen=True)
class RunClock:
    generated_at_utc: str
    run_date_local: str
    market_session: str
    market_close_utc: str
    decision_cutoff_utc: str
    ran_after_market_close: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def as_utc(value: datetime | pd.Timestamp | None) -> pd.Timestamp:
    """A tz-aware UTC timestamp, defaulting to now. Naive input is read as UTC."""
    if value is None:
        return pd.Timestamp(datetime.now(timezone.utc))
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    return result.tz_convert("UTC")


def last_completed_session(
    calendar_name: str = "XNYS",
    now: datetime | pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The most recent session whose close is already in the past, and that close.

    A frozen quote comes from this session and carries no timestamp of its own,
    so this is the only defensible answer to "how old is this price".
    """
    now_utc = as_utc(now)
    calendar = xcals.get_calendar(calendar_name)
    sessions = calendar.sessions_in_range(
        (now_utc - pd.Timedelta(days=14)).date(), (now_utc + pd.Timedelta(days=1)).date()
    )
    for session in reversed(list(sessions)):
        close = pd.Timestamp(calendar.session_close(session))
        close = close.tz_localize("UTC") if close.tzinfo is None else close.tz_convert("UTC")
        if close <= now_utc:
            return pd.Timestamp(session), close
    raise RuntimeError(f"No completed {calendar_name} session before {now_utc.isoformat()}")


def completed_market_clock(
    config: dict[str, Any],
    now: datetime | pd.Timestamp | None = None,
) -> RunClock:
    now_utc = as_utc(now)
    project = config["project"]
    local_tz = ZoneInfo(str(project.get("timezone", "UTC")))
    calendar = xcals.get_calendar(str(project.get("market_calendar", "XNYS")))
    buffer_minutes = int(project.get("post_close_buffer_minutes", 20))

    start = (now_utc - pd.Timedelta(days=14)).date()
    end = (now_utc + pd.Timedelta(days=1)).date()
    sessions = calendar.sessions_in_range(start, end)
    completed: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for session in sessions:
        close = pd.Timestamp(calendar.session_close(session))
        if close.tzinfo is None:
            close = close.tz_localize("UTC")
        else:
            close = close.tz_convert("UTC")
        cutoff = close + pd.Timedelta(minutes=buffer_minutes)
        if cutoff <= now_utc:
            completed.append((pd.Timestamp(session), close, cutoff))
    if not completed:
        raise RuntimeError("No completed XNYS session was found before the run timestamp")

    session, close, cutoff = completed[-1]
    session_date = session.tz_localize(None).date() if session.tzinfo else session.date()
    return RunClock(
        generated_at_utc=now_utc.isoformat(),
        run_date_local=now_utc.tz_convert(local_tz).date().isoformat(),
        market_session=session_date.isoformat(),
        market_close_utc=close.isoformat(),
        decision_cutoff_utc=cutoff.isoformat(),
        ran_after_market_close=True,
    )


def market_sessions(
    calendar_name: str,
    start: Any,
    end: Any,
) -> pd.DatetimeIndex:
    """Actual exchange sessions, not `freq="B"`.

    A business-day index invents every market holiday. Forward-filling prices
    onto those rows produced 252 phantom sessions over 2000-2026, each carrying
    a 0.0 return that then entered every volatility and momentum window.
    exchange_calendars defaults to a recent start, so it is passed explicitly.
    """
    first = pd.Timestamp(start).normalize()
    calendar = xcals.get_calendar(
        calendar_name, start=(first - pd.Timedelta(days=30)).date().isoformat()
    )
    sessions = calendar.sessions_in_range(first.date(), pd.Timestamp(end).normalize().date())
    return pd.DatetimeIndex(sessions).tz_localize(None).normalize()


def session_age(calendar_name: str, latest: Any, as_of: Any) -> int | None:
    latest_ts = pd.to_datetime(latest, errors="coerce")
    as_of_ts = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(latest_ts) or pd.isna(as_of_ts):
        return None
    if latest_ts > as_of_ts:
        return 0
    sessions = market_sessions(calendar_name, latest_ts, as_of_ts)
    return max(0, len(sessions) - 1)

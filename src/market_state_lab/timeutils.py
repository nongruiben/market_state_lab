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


def _as_utc(value: datetime | pd.Timestamp | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp(datetime.now(timezone.utc))
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    return result.tz_convert("UTC")


def completed_market_clock(
    config: dict[str, Any],
    now: datetime | pd.Timestamp | None = None,
) -> RunClock:
    now_utc = _as_utc(now)
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


def session_age(calendar_name: str, latest: Any, as_of: Any) -> int | None:
    latest_ts = pd.to_datetime(latest, errors="coerce")
    as_of_ts = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(latest_ts) or pd.isna(as_of_ts):
        return None
    if latest_ts > as_of_ts:
        return 0
    calendar = xcals.get_calendar(calendar_name)
    sessions = calendar.sessions_in_range(latest_ts.date(), as_of_ts.date())
    return max(0, len(sessions) - 1)

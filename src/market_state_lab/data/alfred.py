from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from market_state_lab.data.http import CachedHttpClient


def load_initial_release_series(
    client: CachedHttpClient,
    series_id: str,
    name: str,
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return the first published value indexed by its ALFRED availability date."""
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY is required when data.fred.vintage_mode=point_in_time"
        )
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "output_type": 4,
        "observation_start": start_date.date().isoformat(),
        "limit": 100000,
    }
    url = "https://api.stlouisfed.org/fred/series/observations?" + urlencode(params)
    result = client.get(url, f"alfred_{series_id}_initial.json")
    payload: dict[str, Any] = json.loads(result.content.decode("utf-8"))
    frame = pd.DataFrame(payload.get("observations", []))
    if frame.empty:
        return pd.DataFrame(columns=[name])
    frame["available_date"] = pd.to_datetime(frame["realtime_start"], errors="coerce")
    frame["observation_date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[name] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["available_date", "observation_date", name])
    frame = frame.sort_values(["available_date", "observation_date"])
    frame = frame.groupby("available_date", as_index=False).tail(1)
    return frame.set_index("available_date")[[name]].sort_index()

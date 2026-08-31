from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from market_state_lab.data.alfred import load_initial_release_series


@dataclass
class _Result:
    content: bytes


class _Client:
    def get(self, url: str, cache_name: str) -> _Result:
        assert "output_type=4" in url
        assert cache_name.startswith("alfred_")
        payload = {
            "observations": [
                {"date": "2020-01-01", "realtime_start": "2020-01-03", "value": "1.0"},
                {"date": "2020-01-02", "realtime_start": "2020-01-03", "value": "1.1"},
                {"date": "2020-01-03", "realtime_start": "2020-01-06", "value": "1.2"},
            ]
        }
        return _Result(json.dumps(payload).encode("utf-8"))


def test_alfred_is_indexed_by_first_public_availability(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    frame = load_initial_release_series(_Client(), "TEST", "series", pd.Timestamp("2020-01-01"))
    assert list(frame.index) == [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06")]
    assert frame.loc["2020-01-03", "series"] == 1.1

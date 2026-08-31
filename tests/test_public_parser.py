import io
import json
import zipfile
from dataclasses import dataclass

import pandas as pd
import pytest

from market_state_lab.config import load_config
from market_state_lab.data.public import PublicDataLoader


def test_french_daily_zip_parser() -> None:
    text = "Description\n\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n20000103,1.00,0.20,-0.10,0.05,0.03,0.01\n20000104,-0.50,0.10,0.20,0.01,-0.02,0.01\n\n Annual Factors\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("sample.csv", text)
    frame = PublicDataLoader._parse_french_zip(buffer.getvalue())
    assert list(frame.columns) == ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    assert frame.iloc[0]["mkt_rf"] == pytest.approx(0.01)


@dataclass
class _HttpResult:
    content: bytes
    status: str = "fixture"


class _YahooClient:
    def get(self, url: str, cache_name: str, headers=None) -> _HttpResult:
        assert "finance/chart/SPY" in url
        assert cache_name == "yahoo_spy.json"
        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1577836800, 1577923200],
                        "indicators": {
                            "quote": [{"close": [100.0, 102.0]}],
                            "adjclose": [{"adjclose": [99.0, 101.0]}],
                        },
                    }
                ],
                "error": None,
            }
        }
        return _HttpResult(json.dumps(payload).encode("utf-8"))


def test_yahoo_chart_uses_adjusted_close() -> None:
    config = load_config()
    config["project"]["start_date"] = "2020-01-01"
    loader = PublicDataLoader(config)
    loader.client = _YahooClient()
    frame = loader._yahoo({"spy": "SPY"})
    assert frame.loc[pd.Timestamp("2020-01-02"), "spy"] == 101.0

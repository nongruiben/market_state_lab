import io
import zipfile

import pytest

from market_state_lab.data.public import PublicDataLoader


def test_french_daily_zip_parser() -> None:
    text = "Description\n\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n20000103,1.00,0.20,-0.10,0.05,0.03,0.01\n20000104,-0.50,0.10,0.20,0.01,-0.02,0.01\n\n Annual Factors\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("sample.csv", text)
    frame = PublicDataLoader._parse_french_zip(buffer.getvalue())
    assert list(frame.columns) == ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    assert frame.iloc[0]["mkt_rf"] == pytest.approx(0.01)


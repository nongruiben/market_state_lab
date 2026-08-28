from pathlib import Path

from market_state_lab.config import PROJECT_ROOT, load_config
from market_state_lab.data.ibkr import supported_ibapi_version


def test_ibkr_configuration_is_opt_in_and_read_only() -> None:
    config = load_config()
    assert config["ibkr"]["readonly_required"] is True
    assert config["ibkr"]["auto_connect"] is False
    assert config["ibkr"]["historical_requires_env_flag"] is True
    assert config["ibkr"]["market_data_type"] == 4


def test_ibkr_module_contains_no_trading_calls() -> None:
    source_path = Path(PROJECT_ROOT) / "src" / "market_state_lab" / "data" / "ibkr.py"
    source = source_path.read_text(encoding="utf-8")
    forbidden = ("place" + "Order", "cancel" + "Order", "reqOpenOrders")
    assert not [token for token in forbidden if token in source]


def test_ibkr_version_gate_rejects_old_pypi_build() -> None:
    assert supported_ibapi_version("10.50.1")
    assert not supported_ibapi_version("9.81.1.post1")
    assert not supported_ibapi_version(None)

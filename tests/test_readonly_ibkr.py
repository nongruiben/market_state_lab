from pathlib import Path

import pytest

from market_state_lab.config import PROJECT_ROOT, load_config
from market_state_lab.data.ibkr import ReadOnlyIBKRClient, supported_ibapi_version


def test_ibkr_configuration_is_opt_in_and_read_only() -> None:
    config = load_config()
    assert config["ibkr"]["readonly_required"] is True
    assert config["ibkr"]["historical_requires_env_flag"] is True
    assert config["ibkr"]["market_data_type"] == 4
    # auto_connect is deliberately gone. Nothing ever read it, so it asserted a
    # promise the code did not keep; the gates below are the ones that exist.
    assert "auto_connect" not in config["ibkr"]


def test_pipeline_never_connects_to_ibkr_without_the_flag() -> None:
    """The only path to a connection is the --with-ibkr flag, and the snapshot it
    fetches lands after the model has already run, so it cannot reach an output."""
    source = (Path(PROJECT_ROOT) / "src" / "market_state_lab" / "pipeline.py").read_text(
        encoding="utf-8"
    )
    assert source.count("ReadOnlyIBKRClient(") == 1
    assert "if with_ibkr:" in source
    assert source.index("state = fit_market_state") < source.index("if with_ibkr:")


def test_disabling_ibkr_actually_refuses_to_connect() -> None:
    config = load_config()
    config["ibkr"]["enabled"] = False
    client = ReadOnlyIBKRClient(config)
    with pytest.raises(RuntimeError, match="ibkr.enabled=false"):
        client.connect()


def test_ibkr_module_contains_no_trading_calls() -> None:
    source_path = Path(PROJECT_ROOT) / "src" / "market_state_lab" / "data" / "ibkr.py"
    source = source_path.read_text(encoding="utf-8")
    forbidden = ("place" + "Order", "cancel" + "Order", "reqOpenOrders")
    assert not [token for token in forbidden if token in source]


def test_ibkr_version_gate_rejects_old_pypi_build() -> None:
    assert supported_ibapi_version("10.50.1")
    assert not supported_ibapi_version("9.81.1.post1")
    assert not supported_ibapi_version(None)

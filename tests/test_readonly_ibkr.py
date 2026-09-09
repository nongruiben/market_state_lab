from pathlib import Path

import pytest

from market_state_lab.config import PROJECT_ROOT, load_config
from market_state_lab.data.ibkr import ReadOnlyIBKRClient, supported_client_version


def test_ibkr_configuration_is_opt_in_and_read_only() -> None:
    config = load_config()
    assert config["ibkr"]["readonly_required"] is True
    assert config["ibkr"]["historical_requires_env_flag"] is True
    assert config["ibkr"]["market_data_type"] == 4
    # auto_connect is deliberately gone. Nothing ever read it, so it asserted a
    # promise the code did not keep; the gates below are the ones that exist.
    assert "auto_connect" not in config["ibkr"]


def _pipeline_source() -> str:
    return (Path(PROJECT_ROOT) / "src" / "market_state_lab" / "pipeline.py").read_text(
        encoding="utf-8"
    )


def test_pipeline_never_connects_to_ibkr_without_the_flag() -> None:
    """Every connection is behind --with-ibkr, and there are only two of them."""
    source = _pipeline_source()
    assert source.count("ReadOnlyIBKRClient(") == 2  # cross-source check, and the snapshot
    assert source.count("if not with_ibkr") == 1  # the cross-source guard
    assert source.count("if with_ibkr:") == 1  # the snapshot guard


def test_tws_may_verify_the_data_but_never_feed_the_market_reading() -> None:
    """The reading is formed from public data before TWS is asked anything.

    TWS is allowed to corroborate what the description was built from - that is
    the whole point of a second source - and it is not allowed to be one of the
    inputs it corroborates. The order enforces it: evidence and assessment are
    settled before the first connection, so a TWS answer can reach the data
    status line and nothing above it.
    """
    source = _pipeline_source()
    assert source.index("assessment = assess(") < source.index("_cross_source(config")
    assert source.index("_cross_source(config") < source.index("report = build_report(")
    # And the snapshot still lands after the report, where it can reach nothing.
    assert source.index("report = build_report(") < source.index("if with_ibkr:")


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


def test_client_version_gate_requires_ib_async_2x() -> None:
    assert supported_client_version("2.1.0")
    # 1.x predates the API this module is written against.
    assert not supported_client_version("1.0.3")
    assert not supported_client_version(None)


def test_every_data_method_opens_an_archive_record() -> None:
    """Seven public data methods, seven begin() sites. A method added without one
    would silently leave its requests out of the raw archive - and out of the
    replay the fault-injection matrix depends on."""
    source_path = Path(PROJECT_ROOT) / "src" / "market_state_lab" / "data" / "ibkr.py"
    source = source_path.read_text(encoding="utf-8")
    data_methods = (
        "server_clock", "qualify_stock", "quotes", "historical_daily_bars",
        "option_parameters", "listed_strikes", "qualify_options",
    )
    for method in data_methods:
        assert f"def {method}(" in source, f"{method} missing from the client"
    assert source.count("self.archive.begin(") >= len(data_methods)

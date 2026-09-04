from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from typing import Any

import pandas as pd

try:
    from ibapi.client import EClient
    from ibapi.contract import Contract
    from ibapi.wrapper import EWrapper
except ImportError:  # The public-data pipeline does not require IBKR.
    EClient = None
    EWrapper = object
    Contract = None


_TICK_NAMES = {
    1: "bid",
    2: "ask",
    4: "last",
    6: "high",
    7: "low",
    9: "close",
    66: "bid",
    67: "ask",
    68: "last",
    72: "high",
    73: "low",
    75: "close",
}


@dataclass(frozen=True)
class IBKRConnectionSettings:
    host: str
    port: int
    client_id: int
    timeout_seconds: int
    market_data_type: int
    readonly_required: bool
    enabled: bool


def installed_ibapi_version() -> str | None:
    try:
        return metadata.version("ibapi")
    except metadata.PackageNotFoundError:
        return None


def supported_ibapi_version(version: str | None) -> bool:
    if version is None:
        return False
    match = re.match(r"^(\d+)", version)
    return bool(match and int(match.group(1)) >= 10)


if EClient is not None:

    class _ReadApp(EWrapper, EClient):
        def __init__(self) -> None:
            EClient.__init__(self, self)
            self.ready = threading.Event()
            self.current_time_event = threading.Event()
            self.snapshot_events: dict[int, threading.Event] = {}
            self.historical_events: dict[int, threading.Event] = {}
            self.quotes: dict[int, dict[str, Any]] = {}
            self.historical: dict[int, list[dict[str, Any]]] = {}
            self.errors: list[dict[str, Any]] = []
            self.server_time: int | None = None

        def nextValidId(self, orderId: int) -> None:  # noqa: N802 - IBKR callback name
            self.ready.set()

        def currentTime(self, time_value: int) -> None:  # noqa: N802
            self.server_time = time_value
            self.current_time_event.set()

        def tickPrice(self, reqId: int, tickType: int, price: float, attrib: Any) -> None:  # noqa: N802
            name = _TICK_NAMES.get(tickType)
            if name and price >= 0:
                self.quotes.setdefault(reqId, {})[name] = price

        def tickSnapshotEnd(self, reqId: int) -> None:  # noqa: N802
            event = self.snapshot_events.get(reqId)
            if event:
                event.set()

        def historicalData(self, reqId: int, bar: Any) -> None:  # noqa: N802
            self.historical.setdefault(reqId, []).append(
                {
                    "date": bar.date,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": float(bar.volume),
                }
            )

        def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:  # noqa: N802
            event = self.historical_events.get(reqId)
            if event:
                event.set()

        def error(
            self,
            reqId: int,
            errorCode: int,
            errorString: str,
            advancedOrderRejectJson: str = "",
        ) -> None:
            if errorCode not in {2104, 2106, 2158}:
                self.errors.append(
                    {"request_id": reqId, "code": errorCode, "message": errorString}
                )


class ReadOnlyIBKRClient:
    """Narrow TWS client whose public surface contains data reads only."""

    def __init__(self, config: dict[str, Any]) -> None:
        section = config["ibkr"]
        self.settings = IBKRConnectionSettings(
            host=str(section["host"]),
            port=int(section["port"]),
            client_id=int(section["client_id"]),
            timeout_seconds=int(section.get("timeout_seconds", 12)),
            market_data_type=int(section.get("market_data_type", 4)),
            readonly_required=bool(section.get("readonly_required", True)),
            enabled=bool(section.get("enabled", True)),
        )
        if self.settings.market_data_type not in {1, 2, 3, 4}:
            raise ValueError("ibkr.market_data_type must be one of 1, 2, 3, or 4")
        self.app: Any | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "ReadOnlyIBKRClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.disconnect()

    @staticmethod
    def _stock(symbol: str) -> Any:
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    def connect(self) -> None:
        # First gate, ahead of everything including the SDK probe: if the operator
        # switched IBKR off, the honest answer is "refusing", not "SDK missing".
        # ibkr.enabled used to be decorative - nothing read it, so setting it to
        # false changed nothing at all.
        if not self.settings.enabled:
            raise RuntimeError("Configuration has ibkr.enabled=false; refusing to connect")
        if EClient is None:
            raise RuntimeError(
                "IBKR's official Python API is not installed; run install_ibkr_api.ps1 "
                "after downloading the official TWS API SDK"
            )
        version = installed_ibapi_version()
        if not supported_ibapi_version(version):
            raise RuntimeError(
                f"Unsupported ibapi version {version or 'unknown'}; remove the PyPI build "
                "and install the official IBKR API 10.x SDK with install_ibkr_api.ps1"
            )
        if not self.settings.readonly_required:
            raise RuntimeError("Configuration must keep ibkr.readonly_required=true")
        app = _ReadApp()
        app.connect(
            self.settings.host,
            self.settings.port,
            clientId=self.settings.client_id,
        )
        thread = threading.Thread(target=app.run, name="ibkr-readonly-loop", daemon=True)
        thread.start()
        if not app.ready.wait(self.settings.timeout_seconds):
            app.disconnect()
            raise TimeoutError("TWS did not complete the read-only API handshake")
        self.app = app
        self.thread = thread

    def disconnect(self) -> None:
        if self.app is not None and self.app.isConnected():
            self.app.disconnect()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.app = None
        self.thread = None

    def _require_app(self) -> Any:
        if self.app is None or not self.app.isConnected():
            raise RuntimeError("IBKR client is not connected")
        return self.app

    def server_clock(self) -> datetime:
        app = self._require_app()
        app.current_time_event.clear()
        app.reqCurrentTime()
        if not app.current_time_event.wait(self.settings.timeout_seconds):
            raise TimeoutError("No current-time response from TWS")
        return datetime.fromtimestamp(app.server_time)

    def delayed_snapshots(self, symbols: list[str]) -> pd.DataFrame:
        app = self._require_app()
        app.reqMarketDataType(self.settings.market_data_type)
        request_ids: dict[int, str] = {}
        for offset, symbol in enumerate(symbols, start=10_000):
            event = threading.Event()
            app.snapshot_events[offset] = event
            app.quotes[offset] = {}
            request_ids[offset] = symbol
            app.reqMktData(offset, self._stock(symbol), "", True, False, [])
        deadline = time.monotonic() + self.settings.timeout_seconds
        for request_id in request_ids:
            remaining = max(0.0, deadline - time.monotonic())
            app.snapshot_events[request_id].wait(remaining)
        rows = []
        timestamp = pd.Timestamp.utcnow()
        for request_id, symbol in request_ids.items():
            row = {"symbol": symbol, "timestamp_utc": timestamp}
            row.update(app.quotes.get(request_id, {}))
            rows.append(row)
        return pd.DataFrame(rows).set_index("symbol")

    def historical_daily_bars(
        self,
        symbol: str,
        duration: str = "1 Y",
    ) -> pd.DataFrame:
        if os.environ.get("IBKR_ALLOW_HISTORICAL", "0") != "1":
            raise PermissionError(
                "Historical reads require IBKR_ALLOW_HISTORICAL=1 and an existing entitlement"
            )
        app = self._require_app()
        request_id = 20_000
        event = threading.Event()
        app.historical_events[request_id] = event
        app.historical[request_id] = []
        app.reqHistoricalData(
            request_id,
            self._stock(symbol),
            "",
            duration,
            "1 day",
            "TRADES",
            1,
            1,
            False,
            [],
        )
        if not event.wait(self.settings.timeout_seconds * 2):
            app.cancelHistoricalData(request_id)
            raise TimeoutError(f"Historical data timed out for {symbol}")
        frame = pd.DataFrame(app.historical[request_id])
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.set_index("date").sort_index()
        return frame

    @property
    def errors(self) -> pd.DataFrame:
        if self.app is None:
            return pd.DataFrame(columns=["request_id", "code", "message"])
        return pd.DataFrame(self.app.errors)

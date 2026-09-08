"""Read-only TWS client built on ib_async.

Replaces a hand-rolled ibapi callback layer. The rewrite is not about ergonomics:
several defects in the old client were consequences of managing request state by
hand, and the await model removes them structurally rather than by discipline.

- Request identity. Fixed id bases (10_000 for snapshots, 20_000 for history)
  could collide once several reads overlapped, and a late callback from a prior
  connection had nowhere to be rejected. ib_async owns request ids and resolves
  each to its own future.
- Snapshot completeness. The old code ignored the wait() return value, so a
  timeout returned whatever fields had arrived, indistinguishable from a full
  response. Status is now explicit per quote.
- Requested vs actual market data type. Both were assumed equal; TWS may answer
  a live request with delayed or frozen data. The type TWS actually returned is
  recorded on every quote.
- Timestamps. One collection-end UTC stamp was applied to a whole row, implying
  every field was observed together. Each quote now carries the exchange time
  ib_async reports plus our receive time, and the two are never conflated.
- Contract identity. `symbol/SMART/USD` is ambiguous. Contracts are qualified to
  a unique conId and primaryExchange before any data request.

The public surface is data reads only. `readonly=True` is passed at connect, so
the refusal is enforced by the API rather than by a config flag we set ourselves.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

import pandas as pd

try:
    from ib_async import IB, Option, StartupFetch, Stock
except ImportError:  # The public-data pipeline does not require TWS.
    IB = None
    Stock = None
    Option = None
    StartupFetch = None


@dataclass(frozen=True)
class IBKRConnectionSettings:
    host: str
    port: int
    client_id: int
    timeout_seconds: int
    market_data_type: int
    readonly_required: bool
    enabled: bool


@dataclass
class RequestLog:
    """Per-request provenance, so a quote can always be traced to its call."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, **fields: Any) -> None:
        self.entries.append({"logged_at_utc": datetime.now(timezone.utc).isoformat(), **fields})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)


MARKET_DATA_TYPE_NAMES = {1: "live", 2: "frozen", 3: "delayed", 4: "delayed_frozen"}


def installed_client_version() -> str | None:
    for name in ("ib_async", "ib-async"):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def supported_client_version(version: str | None) -> bool:
    """ib_async 2.x. The 1.x line predates the API this module is written against."""
    if version is None:
        return False
    match = re.match(r"^(\d+)", version)
    return bool(match and int(match.group(1)) >= 2)


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
        self.ib: Any | None = None
        self.log = RequestLog()

    def __enter__(self) -> "ReadOnlyIBKRClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.disconnect()

    def connect(self) -> None:
        if not self.settings.enabled:
            raise RuntimeError("Configuration has ibkr.enabled=false; refusing to connect")
        if IB is None:
            raise RuntimeError("ib_async is not installed; run: pip install 'ib-async>=2.0'")
        version = installed_client_version()
        if not supported_client_version(version):
            raise RuntimeError(f"Unsupported ib_async version {version or 'unknown'}; need 2.x")
        if not self.settings.readonly_required:
            raise RuntimeError("Configuration must keep ibkr.readonly_required=true")
        ib = IB()
        # readonly=True makes TWS itself refuse order operations for this session.
        # fetchFields(0) matters just as much: ib_async's connect() defaults to
        # requesting positions, open and completed orders, account updates and
        # executions. This project reads none of those, so it must not ask - the
        # first probe run showed all four timing out on every connect.
        ib.connect(
            self.settings.host,
            self.settings.port,
            clientId=self.settings.client_id,
            timeout=self.settings.timeout_seconds,
            readonly=True,
            fetchFields=StartupFetch(0),
        )
        ib.reqMarketDataType(self.settings.market_data_type)
        self.ib = ib
        self.log.record(
            event="connect",
            host=self.settings.host,
            port=self.settings.port,
            client_id=self.settings.client_id,
            requested_market_data_type=self.settings.market_data_type,
            client_version=version,
            server_version=ib.client.serverVersion(),
        )

    def disconnect(self) -> None:
        if self.ib is not None and self.ib.isConnected():
            self.ib.disconnect()
        self.ib = None

    def _require(self) -> Any:
        if self.ib is None or not self.ib.isConnected():
            raise RuntimeError("IBKR client is not connected")
        return self.ib

    def server_clock(self) -> dict[str, Any]:
        """Explicit UTC plus the client/server skew.

        The old version returned a naive local datetime, which silently became
        whatever timezone the machine happened to be in.
        """
        ib = self._require()
        before = datetime.now(timezone.utc)
        server = ib.reqCurrentTime()
        after = datetime.now(timezone.utc)
        if server.tzinfo is None:
            server = server.replace(tzinfo=timezone.utc)
        midpoint = before + (after - before) / 2
        return {
            "server_time_utc": server.astimezone(timezone.utc),
            "local_time_utc": midpoint,
            "skew_seconds": (server - midpoint).total_seconds(),
            "round_trip_seconds": (after - before).total_seconds(),
        }

    def qualify_stock(self, symbol: str) -> Any:
        """Resolve to a unique contract before requesting anything about it."""
        ib = self._require()
        candidates = ib.qualifyContracts(Stock(symbol, "SMART", "USD"))
        if not candidates:
            raise LookupError(f"TWS could not qualify a unique US contract for {symbol}")
        contract = candidates[0]
        self.log.record(
            event="qualify",
            symbol=symbol,
            con_id=contract.conId,
            primary_exchange=contract.primaryExchange,
            resolved=len(candidates),
        )
        return contract

    def quotes(self, contracts: list[Any]) -> pd.DataFrame:
        """Snapshot quotes carrying their own status, times and actual data type."""
        ib = self._require()
        received = datetime.now(timezone.utc)
        tickers = ib.reqTickers(*contracts)
        rows: list[dict[str, Any]] = []
        for ticker in tickers:
            contract = ticker.contract
            actual = getattr(ticker, "marketDataType", None)
            exchange_time = getattr(ticker, "time", None)
            if exchange_time is not None and exchange_time.tzinfo is None:
                exchange_time = exchange_time.replace(tzinfo=timezone.utc)
            has_two_sided = _finite(ticker.bid) and _finite(ticker.ask)
            has_any = has_two_sided or _finite(ticker.last) or _finite(
                getattr(ticker, "close", None)
            )
            rows.append(
                {
                    "symbol": contract.symbol,
                    "con_id": contract.conId,
                    "sec_type": contract.secType,
                    "expiry": getattr(contract, "lastTradeDateOrContractMonth", "") or None,
                    "strike": getattr(contract, "strike", 0.0) or None,
                    "right": getattr(contract, "right", "") or None,
                    "multiplier": getattr(contract, "multiplier", "") or None,
                    "bid": _clean(ticker.bid),
                    "ask": _clean(ticker.ask),
                    "last": _clean(ticker.last),
                    "close": _clean(getattr(ticker, "close", None)),
                    "bid_size": _clean(ticker.bidSize),
                    "ask_size": _clean(ticker.askSize),
                    "exchange_time_utc": exchange_time,
                    "received_at_utc": received,
                    "quote_age_seconds": (
                        (received - exchange_time).total_seconds() if exchange_time else None
                    ),
                    "requested_market_data_type": self.settings.market_data_type,
                    "actual_market_data_type": actual,
                    "actual_market_data_type_name": MARKET_DATA_TYPE_NAMES.get(actual),
                    # Requested type does not imply returned type, and a snapshot
                    # that timed out half-filled must not look like a full one.
                    # Two-sided is what pricing a trade needs; a lone close can
                    # describe the market but cannot cost an option.
                    "status": (
                        "complete" if has_two_sided else ("close_only" if has_any else "empty")
                    ),
                }
            )
        self.log.record(event="quotes", requested=len(contracts), returned=len(rows))
        return pd.DataFrame(rows)

    def historical_daily_bars(
        self,
        contract: Any,
        duration: str = "1 Y",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> pd.DataFrame:
        """Daily bars with their own provenance attached.

        `whatToShow` and `useRTH` were hardcoded and unrecorded, so a series could
        not be told apart from one pulled on different terms - which matters
        before anything is spliced onto another source.
        """
        if os.environ.get("IBKR_ALLOW_HISTORICAL", "0") != "1":
            raise PermissionError(
                "Historical reads require IBKR_ALLOW_HISTORICAL=1 and an existing entitlement"
            )
        ib = self._require()
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting="1 day",
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=2,  # epoch seconds, so no local-timezone guessing
        )
        frame = pd.DataFrame(
            [
                {
                    "date": bar.date,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": float(bar.volume),
                    "bar_count": getattr(bar, "barCount", None),
                }
                for bar in bars
            ]
        )
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
            frame = frame.set_index("date").sort_index()
        frame.attrs.update(
            {
                "symbol": contract.symbol,
                "con_id": contract.conId,
                "what_to_show": what_to_show,
                "use_rth": use_rth,
                "duration": duration,
                "bar_size": "1 day",
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.log.record(
            event="historical",
            symbol=contract.symbol,
            con_id=contract.conId,
            what_to_show=what_to_show,
            use_rth=use_rth,
            duration=duration,
            rows=len(frame),
        )
        return frame

    def option_parameters(self, contract: Any) -> pd.DataFrame:
        """Available expiries and strikes per exchange. Not every combination is a
        real contract - each candidate still has to be qualified."""
        ib = self._require()
        params = ib.reqSecDefOptParams(
            contract.symbol, "", contract.secType, contract.conId
        )
        rows = [
            {
                "exchange": p.exchange,
                "trading_class": p.tradingClass,
                "multiplier": p.multiplier,
                "expirations": sorted(p.expirations),
                "strikes": sorted(p.strikes),
                "expiry_count": len(p.expirations),
                "strike_count": len(p.strikes),
            }
            for p in params
        ]
        self.log.record(event="option_params", symbol=contract.symbol, chains=len(rows))
        return pd.DataFrame(rows)

    def qualify_puts(
        self,
        symbol: str,
        expiry: str,
        strikes: list[float],
        exchange: str = "SMART",
        trading_class: str | None = None,
    ) -> list[Any]:
        """Qualify put contracts, dropping strike/expiry combinations TWS rejects."""
        ib = self._require()
        wanted = [
            Option(symbol, expiry, strike, "P", exchange, tradingClass=trading_class or symbol)
            for strike in strikes
        ]
        qualified = [c for c in ib.qualifyContracts(*wanted) if getattr(c, "conId", 0)]
        self.log.record(
            event="qualify_puts",
            symbol=symbol,
            expiry=expiry,
            requested=len(wanted),
            qualified=len(qualified),
        )
        return qualified

    @property
    def request_log(self) -> pd.DataFrame:
        return self.log.frame()


def _finite(value: Any) -> bool:
    try:
        return value is not None and float(value) == float(value) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _clean(value: Any) -> float | None:
    """TWS sends -1 and NaN as 'no value'; neither is a price."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number < 0:
        return None
    return number

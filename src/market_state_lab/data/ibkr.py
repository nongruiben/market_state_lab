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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from market_state_lab.config import project_path
from market_state_lab.data.contracts import contract_identity as _contract_identity
from market_state_lab.data.snapshots import RawArchive, RequestRecord

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
class ArchiveSink:
    """Every request leaves a RequestRecord and its payload in the raw archive.

    The plan's per-request contract: a monotonic request id inside a connection
    generation, endpoint and parameters recorded, a real lifecycle, and the raw
    response written to `data/raw/ibkr/{fetch_date}/{request_id}.json` exactly
    once. The fault-injection matrix mutates these payloads; without them there
    is nothing to inject into and nothing to replay.

    `state` is not a boolean: `complete` means the payload was written, `failed`
    means the call raised and nothing was archived. A call that returned but
    produced no rows is still `complete` - an empty answer is an answer.
    """

    root: Path
    session_tag: str
    sdk_version: str | None = None
    server_version: str | None = None
    provider: str = "ibkr"
    records: list[RequestRecord] = field(default_factory=list)
    _seq: int = 0
    _archive_dates: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.archive = RawArchive(self.root)

    def begin(
        self,
        endpoint: str,
        contract: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> RequestRecord:
        self._seq += 1
        record = RequestRecord(
            request_id=f"{self.session_tag}-{self._seq:04d}",
            provider=self.provider,
            session_tag=self.session_tag,
            endpoint=endpoint,
            contract=contract or {},
            parameters=parameters or {},
            requested_at_utc=datetime.now(timezone.utc).isoformat(),
            sdk_version=self.sdk_version,
            server_version=self.server_version,
        )
        self.records.append(record)
        self._archive_dates[record.request_id] = datetime.now(timezone.utc).date().isoformat()
        return record

    def complete(self, record: RequestRecord, payload: Any, rows: int | None = None) -> None:
        # The record embedded in the archive file must carry its final state.
        # Writing it first and updating the in-memory list after would leave the
        # file saying "pending" forever while the live record says "complete" -
        # two answers to the same question.
        finished = _replace(
            record, state="complete", completed_at_utc=_now_iso(), rows=rows
        )
        _, digest = self.archive.write(
            finished, payload, self._archive_dates[record.request_id]
        )
        self.records[self.records.index(record)] = _replace(finished, raw_sha256=digest)

    def fail(self, record: RequestRecord, error: str) -> None:
        self.records[self.records.index(record)] = _replace(
            record, state="failed", error=error, completed_at_utc=_now_iso(),
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.records])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _replace(record: RequestRecord, **changes: Any) -> RequestRecord:
    return RequestRecord(**{**asdict(record), **changes})


MARKET_DATA_TYPE_NAMES = {1: "live", 2: "frozen", 3: "delayed", 4: "delayed_frozen"}
# A frozen book is whatever the market last printed; TWS re-sends it on request,
# so the tick timestamp is when it was pushed, not when the price was made.
FROZEN_MARKET_DATA_TYPES = {2, 4}


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
        self._data_root = Path(project_path(config, "data"))
        # Created on connect: one generation per connection, so a record from a
        # previous session can never be mistaken for one from this one.
        self.archive: ArchiveSink | None = None

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
        self.archive = ArchiveSink(
            self._data_root,
            session_tag=(
                f"c{self.settings.client_id}-"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
            ),
            sdk_version=version,
            server_version=ib.client.serverVersion(),
        )

    def disconnect(self) -> None:
        if self.ib is not None and self.ib.isConnected():
            self.ib.disconnect()
        self.ib = None
        # The archive is deliberately kept: callers read the request log after
        # the `with` block closes, and dropping the records would erase the very
        # provenance the block exists to produce.

    def _require(self) -> Any:
        if self.ib is None or not self.ib.isConnected():
            raise RuntimeError("IBKR client is not connected")
        return self.ib

    def server_clock(self) -> dict[str, Any]:
        """Explicit UTC plus the client/server skew.

        The old version returned a naive local datetime, which silently became
        whatever timezone the machine happened to be in.
        """
        record = self.archive.begin("reqCurrentTime")
        try:
            ib = self._require()
            before = datetime.now(timezone.utc)
            server = ib.reqCurrentTime()
            after = datetime.now(timezone.utc)
            if server.tzinfo is None:
                server = server.replace(tzinfo=timezone.utc)
            midpoint = before + (after - before) / 2
            payload = {
                "server_time_utc": server.astimezone(timezone.utc),
                "local_time_utc": midpoint,
                "skew_seconds": (server - midpoint).total_seconds(),
                "round_trip_seconds": (after - before).total_seconds(),
            }
            self.archive.complete(record, payload, rows=1)
            return payload
        except Exception as exc:
            self.archive.fail(record, f"{type(exc).__name__}: {exc}")
            raise

    def qualify_stock(self, symbol: str) -> Any:
        """Resolve to a unique contract before requesting anything about it."""
        record = self.archive.begin(
            "qualifyContracts", contract={"symbol": symbol, "sec_type": "STK", "currency": "USD"}
        )
        try:
            ib = self._require()
            candidates = ib.qualifyContracts(Stock(symbol, "SMART", "USD"))
            if not candidates:
                raise LookupError(f"TWS could not qualify a unique US contract for {symbol}")
            contract = candidates[0]
            self.archive.complete(
                record,
                {"resolved": [_contract_identity(c) for c in candidates]},
                rows=len(candidates),
            )
            return contract
        except Exception as exc:
            self.archive.fail(record, f"{type(exc).__name__}: {exc}")
            raise

    def quotes(
        self,
        contracts: list[Any],
        wait_seconds: float = 12.0,
        generic_ticks: str | None = None,
        open_interest_retries: int = 1,
    ) -> pd.DataFrame:
        """Streaming quotes carrying their own status, times and actual data type.

        Streaming, not `reqTickers`. That helper sends a *snapshot* request, and
        IB does not serve delayed data to snapshots: on a delayed entitlement
        every field comes back empty, so the account looks unsubscribed when it
        is not. This module ran that way and reported `close_only` for SPY and
        nothing at all for its options, both of which quote fine over a
        streaming subscription. The request shape was the fault, not the
        entitlement.

        Every contract is subscribed before the wait and read after it, so the
        cost is one wait rather than one per contract, and the subscriptions are
        cancelled on the way out either way - a market-data line left open is a
        real leak against a capped allowance.

        `generic_ticks` defaults to option volume (100), open interest (101) and
        model greeks (106) when any contract is an option. All three are recorded
        and none is used to value anything; open interest is the one a liquidity
        screen can lean on, and it arrives late or not at all, which is why the
        absent case is kept distinct from a genuine zero.

        `open_interest_retries` re-asks only for the legs whose open interest
        never came, because on a real run half of them do not: three of six SPY
        puts came back without it, which cost the whole snapshot its
        instrument-quote eligibility and left those rows' liquidity unjudgeable.
        A bounded retry is the plan's remedy for a partial return - and only a
        remedy for the fetch. What is still missing after the budget stays
        missing and the row stays degraded; retrying is not a fix, and a value
        that never arrived is never inferred.

        Only the open-interest fields are taken from the retry. Prices are left
        at their first reading, because a second read is a second moment and
        mixing the two is precisely the error that once paired a Tuesday spot
        with a Friday option book. Open interest is an end-of-day figure that
        does not move intraday, so it is the one field a later read can supply
        without changing which moment the row describes.
        """
        if generic_ticks is None:
            generic_ticks = (
                "100,101,106"
                if any(getattr(c, "secType", "") == "OPT" for c in contracts)
                else ""
            )
        record = self.archive.begin(
            "reqMktData",
            contract=[_contract_identity(c) for c in contracts],
            parameters={
                "genericTicks": generic_ticks,
                "wait_seconds": wait_seconds,
                "marketDataType": self.settings.market_data_type,
            },
        )
        ib = self._require()
        tickers = [ib.reqMktData(c, generic_ticks, False, False) for c in contracts]
        try:
            ib.sleep(wait_seconds)
            received = datetime.now(timezone.utc)
            rows = [self._quote_row(ticker, received) for ticker in tickers]
        finally:
            for contract in contracts:
                ib.cancelMktData(contract)
        self.archive.complete(record, rows, rows=len(rows))

        for attempt in range(1, open_interest_retries + 1):
            pending = [
                (index, contract)
                for index, (contract, row) in enumerate(zip(contracts, rows))
                if row["sec_type"] == "OPT" and row["open_interest"] is None
            ]
            if not pending:
                break
            self._retry_open_interest(pending, rows, wait_seconds, attempt)
        return pd.DataFrame(rows)

    def _retry_open_interest(
        self,
        pending: list[tuple[int, Any]],
        rows: list[dict[str, Any]],
        wait_seconds: float,
        attempt: int,
    ) -> None:
        """Re-ask for the open interest that did not arrive, and record the ask."""
        ib = self._require()
        contracts = [contract for _, contract in pending]
        record = self.archive.begin(
            "reqMktData",
            contract=[_contract_identity(c) for c in contracts],
            parameters={
                "genericTicks": "101",
                "wait_seconds": wait_seconds,
                "purpose": "open_interest_retry",
                "attempt": attempt,
            },
        )
        tickers = [ib.reqMktData(c, "101", False, False) for c in contracts]
        try:
            ib.sleep(wait_seconds)
            filled: list[dict[str, Any]] = []
            for (index, _), ticker in zip(pending, tickers):
                right = rows[index].get("right") or ""
                value = _clean(
                    ticker.putOpenInterest if right == "P" else ticker.callOpenInterest
                )
                # Untouched when still absent: the row keeps saying it does not
                # know, which a screen must treat differently from a zero.
                if value is not None:
                    rows[index]["open_interest"] = value
                    rows[index]["open_interest_attempt"] = attempt
                    filled.append({"con_id": rows[index]["con_id"], "open_interest": value})
        finally:
            for contract in contracts:
                ib.cancelMktData(contract)
        self.archive.complete(record, filled, rows=len(filled))

    def _quote_row(self, ticker: Any, received: datetime) -> dict[str, Any]:
        """One ticker to one row, carrying what a reader needs to discount it."""
        contract = ticker.contract
        actual = getattr(ticker, "marketDataType", None)
        exchange_time = getattr(ticker, "time", None)
        if exchange_time is not None and exchange_time.tzinfo is None:
            exchange_time = exchange_time.replace(tzinfo=timezone.utc)
        has_two_sided = _finite(ticker.bid) and _finite(ticker.ask)
        has_any = has_two_sided or _finite(ticker.last) or _finite(getattr(ticker, "close", None))
        greeks = getattr(ticker, "modelGreeks", None)
        # Open interest is right-specific and IB fills the opposite side with a
        # zero rather than leaving it out, so reading the wrong field turns a
        # liquid contract into an empty one. A put's callOpenInterest is 0 by
        # construction, not a fact about the market.
        right = getattr(contract, "right", "") or ""
        if right == "P":
            open_interest, option_volume = ticker.putOpenInterest, ticker.putVolume
        elif right == "C":
            open_interest, option_volume = ticker.callOpenInterest, ticker.callVolume
        else:
            open_interest, option_volume = getattr(ticker, "openInterest", None), None
        return {
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
            # How long since TWS pushed the tick. This is NOT the age of the
            # price, and conflating the two is how a book frozen since the
            # previous Friday reported an age of nine seconds.
            "tick_lag_seconds": (
                (received - exchange_time).total_seconds() if exchange_time else None
            ),
            # Only meaningful when the tick timestamp tracks the market. On a
            # frozen feed the book carries no formation time at all, so the
            # honest value is absent and the caller must ask the calendar.
            "quote_age_seconds": (
                (received - exchange_time).total_seconds()
                if exchange_time and actual not in FROZEN_MARKET_DATA_TYPES
                else None
            ),
            "staleness_basis": (
                "frozen_book_has_no_timestamp"
                if actual in FROZEN_MARKET_DATA_TYPES
                else ("tick_timestamp" if exchange_time else "unknown")
            ),
            "requested_market_data_type": self.settings.market_data_type,
            "actual_market_data_type": actual,
            "actual_market_data_type_name": MARKET_DATA_TYPE_NAMES.get(actual),
            # Requested type does not imply returned type, and a subscription
            # that never filled must not look like a full one. Two-sided is what
            # pricing a trade needs; a lone close can describe the market but
            # cannot cost an option.
            "status": ("complete" if has_two_sided else ("close_only" if has_any else "empty")),
            # Reported by TWS, not computed here. `underlying_price` is the spot
            # the option quote was formed against, which is the only spot that
            # makes its moneyness self-consistent.
            "implied_volatility": _clean(getattr(greeks, "impliedVol", None)),
            "delta": getattr(greeks, "delta", None) if greeks else None,
            "underlying_price": _clean(getattr(greeks, "undPrice", None)),
            # Best-effort. Open interest is end-of-day and often arrives after
            # the quote or not at all; `None` means the tick never came, which a
            # screen must treat differently from a contract nobody holds.
            "open_interest": _clean(open_interest),
            "option_volume": _clean(option_volume),
            # 0 = it arrived on the first ask; a retry stamps its own attempt
            # number, so a value that needed chasing is visible as one.
            "open_interest_attempt": 0,
        }

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
        record = self.archive.begin(
            "reqHistoricalData",
            contract=_contract_identity(contract),
            parameters={
                "durationStr": duration,
                "barSizeSetting": "1 day",
                "whatToShow": what_to_show,
                "useRTH": use_rth,
                "formatDate": 2,
            },
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
        raw_bars = [
            {
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": getattr(bar, "volume", None),
                "bar_count": getattr(bar, "barCount", None),
            }
            for bar in bars
        ]
        self.archive.complete(record, raw_bars, rows=len(bars))
        return frame

    def option_parameters(self, contract: Any) -> pd.DataFrame:
        """Available expiries and strikes per exchange. Not every combination is a
        real contract - each candidate still has to be qualified."""
        record = self.archive.begin(
            "reqSecDefOptParams",
            contract={"symbol": contract.symbol, "sec_type": contract.secType,
                      "con_id": contract.conId},
        )
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
        self.archive.complete(record, rows, rows=len(rows))
        return pd.DataFrame(rows)

    def listed_strikes(
        self,
        symbol: str,
        expiry: str,
        right: str = "P",
        trading_class: str | None = None,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> list[float]:
        """The strikes actually listed for one expiry.

        `reqSecDefOptParams` returns the union of strikes across every expiry in
        a trading class, and that grid is not uniform: SPY's 31-day expiry
        carries 1-point strikes around the money while its 73-day expiry carries
        only 5-point ones. A strike taken from the union therefore names
        contracts that do not exist - 20261120 P732 came back "no security
        definition" while 20261009 P732 qualified fine.

        One `reqContractDetails` with the strike left unset settles it per
        expiry. It reads contract definitions, not market data, so it works
        without any quote subscription.
        """
        record = self.archive.begin(
            "reqContractDetails",
            contract={"symbol": symbol, "expiry": expiry, "right": right,
                      "exchange": exchange, "trading_class": trading_class or symbol},
        )
        ib = self._require()
        blank = Option(
            symbol,
            expiry,
            0,
            right,
            exchange,
            tradingClass=trading_class or symbol,
            currency=currency,
        )
        details = ib.reqContractDetails(blank)
        strikes = sorted(
            {float(d.contract.strike) for d in details if getattr(d.contract, "strike", 0)}
        )
        self.archive.complete(record, {"strikes": strikes}, rows=len(strikes))
        return strikes

    def qualify_options(
        self,
        symbol: str,
        expiry: str,
        strikes: list[float],
        right: str = "P",
        exchange: str = "SMART",
        trading_class: str | None = None,
    ) -> list[Any]:
        """Qualify option contracts, dropping combinations TWS rejects.

        Calls matter here even though nothing in this project buys one: the
        put-call parity check needs the call opposite each put, and it is the
        only way to tell a coherent quote set from a plausible-looking one
        without a second data source.
        """
        record = self.archive.begin(
            "qualifyContracts",
            contract={"symbol": symbol, "expiry": expiry, "right": right,
                      "strikes": strikes, "exchange": exchange,
                      "trading_class": trading_class or symbol},
        )
        ib = self._require()
        wanted = [
            Option(symbol, expiry, strike, right, exchange, tradingClass=trading_class or symbol)
            for strike in strikes
        ]
        qualified = [c for c in ib.qualifyContracts(*wanted) if getattr(c, "conId", 0)]
        self.archive.complete(
            record,
            {"requested_strikes": strikes, "qualified": [_contract_identity(c) for c in qualified]},
            rows=len(qualified),
        )
        return qualified

    def qualify_puts(
        self,
        symbol: str,
        expiry: str,
        strikes: list[float],
        exchange: str = "SMART",
        trading_class: str | None = None,
    ) -> list[Any]:
        """Qualify put contracts, dropping strike/expiry combinations TWS rejects."""
        return self.qualify_options(
            symbol, expiry, strikes, "P", exchange, trading_class
        )

    @property
    def records(self) -> list[RequestRecord]:
        """The RequestRecords this connection produced, for the snapshot manifest.

        The manifest is what links a snapshot back to its raw payloads, and it
        stores the records themselves, not their DataFrame transcription.
        """
        return list(self.archive.records) if self.archive is not None else []

    @property
    def request_log(self) -> pd.DataFrame:
        """The records this connection produced, newest first is a caller concern."""
        if self.archive is None:
            return pd.DataFrame()
        return self.archive.frame()


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

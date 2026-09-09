"""Contract identity and the units a number arrives in.

Two things live here because both are places where a value looks fine and means
something else.

*Identity.* `SPY/SMART/USD` is not a contract - it is a query that usually
resolves to one. SPY's option parameters come back on 39 exchange rows spanning
two trading classes, `SPY` and `2SPY`, and an unqualified pick lands on the
decoy. Everything downstream is keyed on the conId that qualification returned,
and `ContractIdentity` is the shape that carries it.

*Units.* A quantity without its unit is a number that will eventually be added
to a different one. IBKR's historical volume differs from other vendors because
of trade filtering, option volume is contracts rather than shares, and a price
may be raw, split-adjusted or dividend-adjusted. None of those can be repaired
by multiplying by a hundred, so each is declared and carried rather than
inferred at the point of use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# Volume units, kept apart because they are not interchangeable and the
# difference is invisible in the number.
SHARES = "shares"
CONTRACTS = "contracts"
LOTS = "lots"
UNKNOWN_UNIT = "unknown"
VOLUME_UNITS = (SHARES, CONTRACTS, LOTS, UNKNOWN_UNIT)

# Price conventions. Comparing across two of these produces a difference that is
# entirely real and entirely meaningless - a year of SPY dividends is 1.1%.
RAW = "raw"
SPLIT_ADJUSTED = "split_adjusted"
TOTAL_RETURN = "dividend_adjusted"
PRICE_CONVENTIONS = (RAW, SPLIT_ADJUSTED, TOTAL_RETURN)


@dataclass(frozen=True)
class ContractIdentity:
    """What uniquely names an instrument, in a JSON-able form.

    `con_id` is the identity; everything else is description. Two rows agreeing
    on symbol, expiry and strike may still be different contracts, which is what
    the trading class is here to reveal.
    """

    con_id: int | None = None
    symbol: str | None = None
    sec_type: str | None = None
    currency: str | None = None
    exchange: str | None = None
    primary_exchange: str | None = None
    expiry: str | None = None
    strike: float | None = None
    right: str | None = None
    multiplier: str | None = None
    trading_class: str | None = None
    local_symbol: str | None = None

    @property
    def qualified(self) -> bool:
        return bool(self.con_id)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def contract_identity(contract: Any) -> dict[str, Any]:
    """Read an ib_async contract into the identity shape.

    Deliberately a plain dict at the boundary: it is written into the raw
    archive, and an archive that depends on a class definition stops being
    readable the day that class changes.
    """
    return ContractIdentity(
        con_id=getattr(contract, "conId", None),
        symbol=getattr(contract, "symbol", None),
        sec_type=getattr(contract, "secType", None),
        currency=getattr(contract, "currency", None),
        exchange=getattr(contract, "exchange", None),
        primary_exchange=getattr(contract, "primaryExchange", None),
        expiry=getattr(contract, "lastTradeDateOrContractMonth", None) or None,
        strike=getattr(contract, "strike", None) or None,
        right=getattr(contract, "right", None) or None,
        multiplier=getattr(contract, "multiplier", None) or None,
        trading_class=getattr(contract, "tradingClass", None) or None,
        local_symbol=getattr(contract, "localSymbol", None) or None,
    ).as_dict()


@dataclass(frozen=True)
class SeriesConvention:
    """How to read a price series. Comparing two that disagree is meaningless.

    Carried alongside the numbers rather than assumed at the point of use,
    because assuming it once cost this project a cross-source check that passed
    on a 2.3% tolerance absorbing a year of dividend drift.
    """

    price: str = RAW
    volume_unit: str = UNKNOWN_UNIT
    session: str = "RTH"
    currency: str = "USD"

    def __post_init__(self) -> None:
        if self.price not in PRICE_CONVENTIONS:
            raise ValueError(f"unknown price convention {self.price!r}")
        if self.volume_unit not in VOLUME_UNITS:
            raise ValueError(f"unknown volume unit {self.volume_unit!r}")

    def comparable_with(self, other: "SeriesConvention") -> bool:
        """Only the price convention, session and currency have to match.

        Volume units may differ between vendors without making the prices
        incomparable, and IBKR's trade filtering means they routinely do.
        """
        return (self.price, self.session, self.currency) == (
            other.price, other.session, other.currency
        )


__all__ = [
    "CONTRACTS",
    "PRICE_CONVENTIONS",
    "RAW",
    "SHARES",
    "TOTAL_RETURN",
    "VOLUME_UNITS",
    "ContractIdentity",
    "SeriesConvention",
    "contract_identity",
]

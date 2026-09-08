"""Protective-put cost and payoff arithmetic against a reference exposure.

Deliberately narrow. This module answers one question a market model cannot:
*given* that someone wants downside protection, what does a specific listed put
actually cost, and what does it actually pay at expiry? That is arithmetic on a
quote and a contract spec, so it needs no forecasting skill and makes no claim
about whether protection is worth buying.

What it does NOT do, on purpose:

- No pre-expiry valuation. Marking a put before expiry needs an American
  pricing model with discrete dividends; until one is validated here, quoting a
  precise early-exit P&L would be inventing precision. Only the expiry curve and
  the live quote are reported.
- No probabilities. The move grid is a set of stress assumptions, not a
  distribution, and nothing here weights them.
- No user positions. Everything is expressed against an explicit reference
  notional so the output is never mistaken for advice about real holdings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_MOVE_GRID = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10)


@dataclass(frozen=True)
class ReferenceExposure:
    """A stated long position used only as a common yardstick.

    Never sourced from an account. Two underlyings compared at the same notional
    are still not interchangeable hedges for the same portfolio.
    """

    symbol: str
    spot: float
    notional_usd: float = 100_000.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.spot) or self.spot <= 0:
            raise ValueError(f"{self.symbol}: spot must be a positive finite price")
        if not np.isfinite(self.notional_usd) or self.notional_usd <= 0:
            raise ValueError(f"{self.symbol}: reference notional must be positive")

    @property
    def shares(self) -> float:
        return self.notional_usd / self.spot


@dataclass(frozen=True)
class PutQuote:
    """One listed put, priced from a two-sided quote.

    `ask` is what a buyer pays, so it is the only price used for cost. `bid` is
    carried for the spread, which is a liquidity fact worth showing, not a price
    anyone gets on entry.
    """

    symbol: str
    expiry: str
    strike: float
    bid: float
    ask: float
    multiplier: int = 100
    con_id: int | None = None
    quote_age_seconds: float | None = None
    market_data_type: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.strike) or self.strike <= 0:
            raise ValueError(f"{self.symbol} {self.expiry}: strike must be positive")
        if self.multiplier <= 0:
            raise ValueError(f"{self.symbol} {self.expiry}: multiplier must be positive")
        if not np.isfinite(self.ask) or self.ask <= 0:
            raise ValueError(
                f"{self.symbol} {self.expiry} {self.strike}: needs a positive ask to price a buy"
            )

    @property
    def spread(self) -> float:
        return float(self.ask - self.bid) if np.isfinite(self.bid) else float("nan")

    @property
    def mid(self) -> float:
        """Valuation reference only - nobody is guaranteed a mid fill."""
        return float((self.ask + self.bid) / 2.0) if np.isfinite(self.bid) else float("nan")


def size_protection(
    exposure: ReferenceExposure,
    put: PutQuote,
    coverage: float = 1.0,
) -> dict[str, Any]:
    """How many whole contracts the reference exposure implies, and what it costs.

    Contracts are integers, so coverage almost never lands exactly on the target
    and the residual is reported rather than rounded away. A delta-matched count
    is not computed here: matching delta today says nothing about payoff in the
    large move this position exists for.
    """
    if coverage <= 0:
        raise ValueError("coverage must be positive")
    exact = exposure.shares * coverage / put.multiplier
    contracts = int(np.floor(exact))
    shares_covered = contracts * put.multiplier
    return {
        "reference_shares": exposure.shares,
        "contracts_exact": exact,
        "contracts": contracts,
        "shares_covered": shares_covered,
        "coverage_ratio": shares_covered / exposure.shares if exposure.shares else float("nan"),
        "uncovered_shares": exposure.shares - shares_covered,
    }


def protective_put_scenarios(
    exposure: ReferenceExposure,
    put: PutQuote,
    moves: tuple[float, ...] = DEFAULT_MOVE_GRID,
    coverage: float = 1.0,
    fee_per_contract: float = 0.65,
) -> pd.DataFrame:
    """Expiry P&L of long stock plus long puts, against the same stock unhedged.

    Every row is closed-form: the put pays `max(strike - price, 0)` per share at
    expiry and nothing before it. Premium is charged at the ask plus fees, so the
    hedged column is a conservative cost case rather than a mid-price fantasy.
    """
    sizing = size_protection(exposure, put, coverage)
    contracts = sizing["contracts"]
    premium = contracts * put.multiplier * put.ask
    fees = contracts * fee_per_contract
    outlay = premium + fees

    rows: list[dict[str, Any]] = []
    for move in moves:
        terminal = exposure.spot * (1.0 + move)
        unhedged = exposure.shares * terminal
        payoff = contracts * put.multiplier * max(put.strike - terminal, 0.0)
        hedged = unhedged + payoff - outlay
        rows.append(
            {
                "move": move,
                "terminal_price": terminal,
                "unhedged_value": unhedged,
                "put_payoff": payoff,
                "premium_and_fees": outlay,
                "hedged_value": hedged,
                "unhedged_pnl": unhedged - exposure.notional_usd,
                "hedged_pnl": hedged - exposure.notional_usd,
                "protection_vs_unhedged": hedged - unhedged,
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs.update(
        {
            "symbol": exposure.symbol,
            "spot": exposure.spot,
            "reference_notional_usd": exposure.notional_usd,
            "expiry": put.expiry,
            "strike": put.strike,
            "moneyness": put.strike / exposure.spot - 1.0,
            "ask": put.ask,
            "bid": put.bid,
            "spread": put.spread,
            "multiplier": put.multiplier,
            "con_id": put.con_id,
            "quote_age_seconds": put.quote_age_seconds,
            "market_data_type": put.market_data_type,
            "premium_pct_of_notional": outlay / exposure.notional_usd,
            **sizing,
        }
    )
    return frame


def summarise_candidate(frame: pd.DataFrame) -> dict[str, Any]:
    """One row per candidate for side-by-side comparison.

    No weighted score. Ranking candidates by a single number would hide the
    trade the reader is supposed to make - a deeper strike costs less and
    protects later, and that is a preference, not an optimum.
    """
    attrs = frame.attrs
    worst = frame.loc[frame["move"].idxmin()]
    flat = frame.loc[frame["move"].abs().idxmin()]
    # Where the put finishes worthless, the hedge has cost exactly its premium.
    return {
        "symbol": attrs["symbol"],
        "expiry": attrs["expiry"],
        "strike": attrs["strike"],
        "moneyness": attrs["moneyness"],
        "contracts": attrs["contracts"],
        "coverage_ratio": attrs["coverage_ratio"],
        "ask": attrs["ask"],
        "spread": attrs["spread"],
        "cost_usd": attrs["premium_and_fees"] if "premium_and_fees" in attrs else float(frame["premium_and_fees"].iloc[0]),
        "cost_pct_of_notional": attrs["premium_pct_of_notional"],
        "pnl_at_worst_move": float(worst["hedged_pnl"]),
        "unhedged_pnl_at_worst_move": float(worst["unhedged_pnl"]),
        "protection_at_worst_move": float(worst["protection_vs_unhedged"]),
        "pnl_if_flat": float(flat["hedged_pnl"]),
        "quote_age_seconds": attrs["quote_age_seconds"],
        "market_data_type": attrs["market_data_type"],
    }

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
    best = frame.loc[frame["move"].idxmax()]
    # Where the put finishes worthless, the hedge has cost exactly its premium.
    return {
        # Same keys as summarise_control, so candidates and the do-nothing and
        # de-risk controls can be read down one table instead of three.
        "label": f"put {attrs['strike']:g} {attrs['expiry']}",
        "structure": "long put",
        "pnl_at_best_move": float(best["hedged_pnl"]),
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
        # Plan 10.3: what the protection actually covers, and what it does not.
        # Below the strike the puts pay one-for-one on the shares they cover;
        # between spot and strike nothing pays, and the shares whole contracts
        # could not cover are never protected at any price.
        "protected_below": attrs["strike"],
        "unprotected_drop_pct": attrs["strike"] / attrs["spot"] - 1.0,
        "uncovered_shares": attrs["uncovered_shares"],
        "maintenance": "none until expiry; the payoff is fixed by the contract",
        "review_when": f"expiry {attrs['expiry']}, or a spot move through {attrs['strike']:g}",
        "quote_age_seconds": attrs["quote_age_seconds"],
        "market_data_type": attrs["market_data_type"],
    }


def _control_frame(
    exposure: ReferenceExposure,
    label: str,
    rows: list[dict[str, Any]],
    cost: float,
    attrs: dict[str, Any],
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.attrs.update(
        {
            "label": label,
            "symbol": exposure.symbol,
            "spot": exposure.spot,
            "reference_notional_usd": exposure.notional_usd,
            "premium_and_fees": cost,
            "premium_pct_of_notional": cost / exposure.notional_usd,
            **attrs,
        }
    )
    return frame


def no_protection_scenarios(
    exposure: ReferenceExposure,
    moves: tuple[float, ...] = DEFAULT_MOVE_GRID,
) -> pd.DataFrame:
    """The reference exposure carried as it is.

    The control that must appear beside every candidate. Without it a table of
    six puts reads as "pick one of these", when the honest first question is
    whether to buy protection at all.
    """
    rows: list[dict[str, Any]] = []
    for move in moves:
        terminal = exposure.spot * (1.0 + move)
        value = exposure.shares * terminal
        rows.append(
            {
                "move": move,
                "terminal_price": terminal,
                "unhedged_value": value,
                "put_payoff": 0.0,
                "premium_and_fees": 0.0,
                "hedged_value": value,
                "unhedged_pnl": value - exposure.notional_usd,
                "hedged_pnl": value - exposure.notional_usd,
                "protection_vs_unhedged": 0.0,
            }
        )
    return _control_frame(exposure, "no new protection", rows, 0.0, {"structure": "unhedged"})


def reduced_exposure_scenarios(
    exposure: ReferenceExposure,
    moves: tuple[float, ...] = DEFAULT_MOVE_GRID,
    reduce_to: float = 0.8,
    horizon_days: int = 30,
    cash_rate: float = 0.0,
    exit_cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Sell down to `reduce_to` of the reference exposure and hold the rest in cash.

    The other way to carry less downside, and the one a put has to be worth more
    than. It is not free: the sale pays a spread, and giving up the position
    gives up its upside too, which the +5% and +10% rows are there to show.

    `cash_rate` defaults to zero and that understates this control - the cash
    really would earn something. It is a flag rather than a guess because
    inventing a rate here would quietly flatter the puts.
    """
    if not 0.0 <= reduce_to <= 1.0:
        raise ValueError("reduce_to must be a fraction of the reference exposure")
    sold_notional = exposure.notional_usd * (1.0 - reduce_to)
    exit_cost = sold_notional * exit_cost_bps / 10_000.0
    cash = sold_notional - exit_cost
    carry = cash * cash_rate * horizon_days / 365.0
    kept_shares = exposure.shares * reduce_to

    rows: list[dict[str, Any]] = []
    for move in moves:
        terminal = exposure.spot * (1.0 + move)
        unhedged = exposure.shares * terminal
        value = kept_shares * terminal + cash + carry
        rows.append(
            {
                "move": move,
                "terminal_price": terminal,
                "unhedged_value": unhedged,
                "put_payoff": 0.0,
                "premium_and_fees": exit_cost,
                "hedged_value": value,
                "unhedged_pnl": unhedged - exposure.notional_usd,
                "hedged_pnl": value - exposure.notional_usd,
                "protection_vs_unhedged": value - unhedged,
            }
        )
    return _control_frame(
        exposure,
        f"reduce to {reduce_to:.0%}",
        rows,
        exit_cost,
        {
            "structure": "de-risked",
            "reduce_to": reduce_to,
            "cash_usd": cash,
            "cash_rate": cash_rate,
            "cash_carry_usd": carry,
            "horizon_days": horizon_days,
        },
    )


def summarise_control(frame: pd.DataFrame) -> dict[str, Any]:
    """A control shaped like a candidate, so one table can hold both."""
    attrs = frame.attrs
    worst = frame.loc[frame["move"].idxmin()]
    flat = frame.loc[frame["move"].abs().idxmin()]
    best = frame.loc[frame["move"].idxmax()]
    return {
        "label": attrs["label"],
        "symbol": attrs["symbol"],
        "structure": attrs["structure"],
        "expiry": None,
        "strike": None,
        "moneyness": None,
        "contracts": None,
        # Neither control buys coverage. Owning less is not the same as being
        # covered: it removes the exposure instead of insuring it, and the
        # +5%/+10% rows are where that difference shows up.
        "coverage_ratio": 0.0,
        "ask": None,
        "spread": None,
        "cost_usd": attrs["premium_and_fees"],
        "cost_pct_of_notional": attrs["premium_pct_of_notional"],
        "pnl_if_flat": float(flat["hedged_pnl"]),
        "pnl_at_best_move": float(best["hedged_pnl"]),
        "unhedged_pnl_at_worst_move": float(worst["unhedged_pnl"]),
        "pnl_at_worst_move": float(worst["hedged_pnl"]),
        "protection_at_worst_move": float(worst["protection_vs_unhedged"]),
        # A control has no strike, so it protects nothing below a level; the
        # de-risked one simply owns less. The maintenance line is where the two
        # really differ from a put: a put ends itself on a known date, while a
        # decision to hold less stays open until someone reverses it.
        "protected_below": None,
        "unprotected_drop_pct": None,
        "uncovered_shares": None,
        "maintenance": (
            "none; nothing was bought"
            if attrs["structure"] == "unhedged"
            else "open-ended; the cash has no expiry and re-entry is a further decision"
        ),
        "review_when": (
            "no expiry sets a date; the decision stays open"
            if attrs["structure"] == "unhedged"
            else "no expiry sets a date; re-entry timing is an unmade decision"
        ),
        "quote_age_seconds": None,
        "market_data_type": None,
    }

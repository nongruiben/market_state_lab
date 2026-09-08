"""Turn an option chain into a short list of contracts worth quoting.

Everything here runs before any quote arrives, which is the point: chain
selection is where contract ambiguity and silent scope creep get in, and those
are cheaper to get right without a live feed in the loop.

The probe found the concrete hazard. SPY's chain comes back on 39 exchange rows
covering two trading classes: `SPY` with 34 expiries and 483 strikes, and `2SPY`
with 2 and 2. Taking the first row - or filtering on exchange alone - silently
picks the wrong universe, so `select_chain` matches the trading class to the
symbol and then takes the widest remaining row.

A strike drawn from the chain is a *proposal*, never a contract. The strike
list in `reqSecDefOptParams` is the union across every expiry, and the grid is
not uniform - SPY's 31-day expiry lists 1-point strikes around the money while
its 73-day expiry lists only 5-point ones, so a strike that is real for one is
absent for the other. `plan_candidates` therefore marks its rows `proposed`, and
only `resolve_strikes`, fed the strikes each expiry actually lists, can promote a
row to `ok`. Nothing unresolved is ever quotable.

Expiry buckets are a *comparison range*, not a recommendation. Nothing here
decides that protection is worth buying, or which horizon a person should want;
it produces a small set of alternatives that later get priced side by side.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ExpiryBucket:
    """A window of days-to-expiry, compared against the others rather than ranked."""

    name: str
    min_days: int
    max_days: int

    def __post_init__(self) -> None:
        if self.min_days <= 0 or self.max_days <= self.min_days:
            raise ValueError(f"{self.name}: need 0 < min_days < max_days")


DEFAULT_BUCKETS = (
    ExpiryBucket("30-60d", 25, 60),
    ExpiryBucket("60-90d", 60, 95),
)

# Moneyness levels, not a view. A shallower strike starts protecting sooner and
# costs more; that trade is the reader's to make once the prices are attached.
DEFAULT_MONEYNESS = (-0.02, -0.05, -0.10)


def select_chain(params: pd.DataFrame, symbol: str) -> pd.Series:
    """Pick the chain that belongs to the underlying, then the widest one.

    Trading class first: `2SPY` is a different contract universe that happens to
    appear alongside `SPY`, and it is the row an unqualified `iloc[0]` lands on.
    """
    if params.empty:
        raise LookupError(f"{symbol}: TWS returned no option parameters")
    own = params.loc[params["trading_class"].astype(str).str.upper() == symbol.upper()]
    if own.empty:
        raise LookupError(
            f"{symbol}: no chain with a matching trading class; "
            f"saw {sorted(set(params['trading_class'].astype(str)))}"
        )
    return own.loc[own["strike_count"].idxmax()]


def _to_date(expiry: str) -> date | None:
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(expiry), fmt).date()
        except ValueError:
            continue
    return None


def select_expiries(
    expirations: list[str],
    as_of: date,
    buckets: tuple[ExpiryBucket, ...] = DEFAULT_BUCKETS,
) -> pd.DataFrame:
    """One expiry per bucket - the nearest that falls inside it.

    A bucket with no listed expiry yields a row with `expiry` unset and a reason,
    rather than quietly borrowing the neighbouring bucket's date.
    """
    dated = [(e, d, (d - as_of).days) for e in expirations if (d := _to_date(e)) is not None]
    rows: list[dict[str, Any]] = []
    for bucket in buckets:
        inside = [(e, days) for e, _, days in dated if bucket.min_days <= days <= bucket.max_days]
        if not inside:
            rows.append(
                {
                    "bucket": bucket.name,
                    "expiry": None,
                    "days_to_expiry": None,
                    "status": "no_listed_expiry",
                }
            )
            continue
        expiry, days = min(inside, key=lambda item: item[1])
        rows.append(
            {"bucket": bucket.name, "expiry": expiry, "days_to_expiry": days, "status": "ok"}
        )
    return pd.DataFrame(rows)


def nearest_strike(strikes: list[float], target: float) -> float | None:
    """Closest listed strike, ties going to the lower (cheaper, deeper) one."""
    available = [float(s) for s in strikes if pd.notna(s)]
    if not available:
        return None
    return min(available, key=lambda s: (abs(s - target), s))


def plan_candidates(
    symbol: str,
    chain: pd.Series,
    spot: float,
    as_of: date,
    buckets: tuple[ExpiryBucket, ...] = DEFAULT_BUCKETS,
    moneyness: tuple[float, ...] = DEFAULT_MONEYNESS,
    max_strike_drift: float = 0.01,
) -> pd.DataFrame:
    """The contracts to qualify and quote, with every rejection recorded.

    Rows come back `proposed`, not `ok`: the strike is drawn from the chain's
    union across expiries, so it may name a contract this particular expiry does
    not list. `resolve_strikes` is what confirms it.

    `max_strike_drift` guards a thin chain: if the nearest strike sits further
    from the target than this, the row is kept but marked, because a "-5% put"
    that is really -8% would otherwise be compared as if it were the protection
    that was asked for.
    """
    if not spot or spot <= 0:
        raise ValueError(f"{symbol}: need a positive spot to compute strikes")
    strikes = list(chain["strikes"])
    expiries = select_expiries(list(chain["expirations"]), as_of, buckets)

    rows: list[dict[str, Any]] = []
    for expiry_row in expiries.itertuples(index=False):
        for target_moneyness in moneyness:
            target = spot * (1.0 + target_moneyness)
            record: dict[str, Any] = {
                "symbol": symbol,
                "trading_class": chain["trading_class"],
                "multiplier": chain["multiplier"],
                "bucket": expiry_row.bucket,
                "expiry": expiry_row.expiry,
                "days_to_expiry": expiry_row.days_to_expiry,
                "target_moneyness": target_moneyness,
                "target_strike": target,
                "right": "P",
            }
            if expiry_row.status != "ok":
                rows.append({**record, "strike": None, "status": expiry_row.status})
                continue
            strike = nearest_strike(strikes, target)
            if strike is None:
                rows.append({**record, "strike": None, "status": "no_listed_strike"})
                continue
            drift = abs(strike / target - 1.0)
            rows.append(
                {
                    **record,
                    "strike": strike,
                    "actual_moneyness": strike / spot - 1.0,
                    "strike_drift": drift,
                    "strike_source": "chain_union",
                    "status": (
                        "proposed" if drift <= max_strike_drift else "strike_far_from_target"
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs.update({"symbol": symbol, "spot": spot, "as_of": as_of.isoformat()})
    return frame


def resolve_strikes(
    candidates: pd.DataFrame,
    listed: dict[str, list[float]],
    max_strike_drift: float = 0.01,
) -> pd.DataFrame:
    """Move each proposed strike onto the grid its expiry actually lists.

    `listed` maps expiry to the strikes confirmed by contract definitions - free
    to fetch, and the only authority on what exists. The proposal is kept beside
    the resolved strike rather than overwritten, because the gap between them is
    how a thinning grid shows up: a target that resolves 5 points away on a
    73-day expiry is a different instrument from the one the plan named.
    """
    if candidates.empty:
        return candidates
    rows: list[dict[str, Any]] = []
    for row in candidates.to_dict("records"):
        if row.get("status") not in {"proposed", "strike_far_from_target"}:
            rows.append(row)
            continue
        target = row["target_strike"]
        # Recovered from the row itself, so the result never depends on attrs
        # surviving a concat or a round trip through disk.
        spot = target / (1.0 + row["target_moneyness"])
        real = listed.get(str(row["expiry"]))
        row = {**row, "strike_proposed": row.get("strike"), "strike_source": "contract_details"}
        if not real:
            rows.append({**row, "strike": None, "status": "expiry_not_listed"})
            continue
        strike = nearest_strike(list(real), target)
        if strike is None:
            rows.append({**row, "strike": None, "status": "no_listed_strike"})
            continue
        drift = abs(strike / target - 1.0)
        rows.append(
            {
                **row,
                "strike": strike,
                "actual_moneyness": strike / spot - 1.0,
                "strike_drift": drift,
                "status": "ok" if drift <= max_strike_drift else "strike_far_from_target",
            }
        )
    resolved = pd.DataFrame(rows)
    resolved.attrs.update(candidates.attrs)
    return resolved


def quotable(candidates: pd.DataFrame) -> pd.DataFrame:
    """The subset worth spending a quote request on.

    Only `resolve_strikes` sets `ok`, so an unresolved plan yields nothing here.
    That is the point: a quote must never be spent on a contract whose existence
    has not been confirmed.
    """
    if candidates.empty:
        return candidates
    return candidates.loc[candidates["status"].eq("ok")].reset_index(drop=True)

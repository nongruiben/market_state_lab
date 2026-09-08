"""Decide whether a set of option quotes is coherent before anything is built on it.

A quote that arrives is not a quote that is right. Delayed feeds, frozen books,
a mis-qualified contract and a stale line all return numbers, and none of them
announce themselves. Put-call parity is the one test that needs no model, no
subscription and no second data source: for a single expiry,

    C - P = e^(-rT) * (F - K)

so `C - P` is linear in the strike with slope `-e^(-rT)`, and every strike must
imply the same forward. Nothing about that depends on a volatility assumption or
on the options being fairly priced - it is enforced by arbitrage, so a quote set
that fails it is broken rather than merely expensive.

The tolerance is the quoted spread itself. A mid is only known to within half a
spread on each leg, so a residual smaller than the two half-spreads combined is
not evidence of anything. Deep in-the-money calls quote three points wide while
the puts opposite them quote two cents wide, and a fixed threshold would either
wave those through or condemn them depending on which side it was tuned to.

What this deliberately does not do: infer a risk-free rate worth using
elsewhere. The implied rate is reported because it is a legible sanity check - a
number near the cash rate, less the dividend yield, means the chain hangs
together - not because this is a way to measure funding.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Parity fixes the slope at -e^(-rT). Anything outside this band is not a
# mispriced chain, it is the wrong contracts paired up - a mismatched
# multiplier, a different underlying, or calls and puts from different expiries.
SLOPE_BAND = (-1.05, -0.90)


def _mid(frame: pd.DataFrame) -> pd.Series:
    return (frame["bid"] + frame["ask"]) / 2.0


def _half_spread(frame: pd.DataFrame) -> pd.Series:
    return (frame["ask"] - frame["bid"]) / 2.0


def put_call_parity_check(
    quotes: pd.DataFrame,
    days_to_expiry: dict[str, int],
) -> pd.DataFrame:
    """One row per expiry saying whether its quotes are arbitrage-coherent.

    `quotes` needs `expiry`, `strike`, `right`, `bid` and `ask`; calls and puts
    at the same strike are paired, and unpaired strikes are ignored rather than
    guessed at. Two pairs fit a line exactly, so the residual test only carries
    information from three pairs up - `status` says which test actually ran.
    """
    if quotes.empty:
        return pd.DataFrame()

    usable = quotes.dropna(subset=["bid", "ask", "strike", "right"])
    rows: list[dict[str, Any]] = []
    for expiry, group in usable.groupby("expiry"):
        expiry = str(expiry)
        priced = group.assign(mid=_mid(group), half_spread=_half_spread(group))
        mids = priced.pivot_table(index="strike", columns="right", values="mid")
        halves = priced.pivot_table(index="strike", columns="right", values="half_spread")
        record: dict[str, Any] = {
            "expiry": expiry,
            "days_to_expiry": days_to_expiry.get(expiry),
            "pairs": 0,
            "slope": None,
            "implied_rate": None,
            "implied_forward": None,
            "max_abs_residual": None,
            "worst_residual_tolerance": None,
            "status": "insufficient_pairs",
        }
        if not {"C", "P"}.issubset(mids.columns):
            rows.append(record)
            continue
        paired = mids.dropna(subset=["C", "P"])
        record["pairs"] = len(paired)
        if len(paired) < 2:
            rows.append(record)
            continue

        strikes = paired.index.to_numpy(dtype=float)
        basis = (paired["C"] - paired["P"]).to_numpy(dtype=float)
        slope, intercept = (float(v) for v in np.polyfit(strikes, basis, 1))
        record["slope"] = slope
        if slope < 0:
            record["implied_forward"] = -intercept / slope
            days = days_to_expiry.get(expiry)
            if days:
                record["implied_rate"] = float(-np.log(-slope) / (days / 365.0))

        if not SLOPE_BAND[0] <= slope <= SLOPE_BAND[1]:
            # Not a pricing complaint. A slope this far off means the legs being
            # differenced are not the same contract pair.
            record["status"] = "slope_out_of_band"
            rows.append(record)
            continue

        residuals = np.abs(basis - (slope * strikes + intercept))
        tolerance = (
            halves.reindex(paired.index)[["C", "P"]].sum(axis=1).to_numpy(dtype=float)
        )
        record["max_abs_residual"] = float(residuals.max())
        record["worst_residual_tolerance"] = float(tolerance[int(residuals.argmax())])
        if len(paired) < 3:
            # An exact fit through two points proves nothing about the quotes.
            record["status"] = "slope_only"
        elif np.any(residuals > tolerance):
            record["status"] = "parity_violated"
        else:
            record["status"] = "ok"
        rows.append(record)
    return pd.DataFrame(rows)


def parity_verdict(checks: pd.DataFrame) -> str:
    """A single word for the report header, biased towards saying less.

    `unverified` is not `ok`. An expiry that could only be slope-checked has not
    been shown to be coherent, and the header must not imply that it has.
    """
    if checks.empty:
        return "unverified"
    statuses = set(checks["status"])
    if statuses == {"ok"}:
        return "ok"
    if "parity_violated" in statuses or "slope_out_of_band" in statuses:
        return "failed"
    return "unverified"

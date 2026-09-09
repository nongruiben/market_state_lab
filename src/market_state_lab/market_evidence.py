"""Five dimensions of market evidence, described and never forecast.

This layer answers "what changed and what disagrees". It does not answer "what
happens next", and the separation is structural rather than a matter of tone:
there is no probability in the output, no composite score, and no field a
reader could sort to find the riskiest thing.

Four rules the plan fixes, and the reason each one exists:

*A percentile is a rank in history, not a chance of a fall.* Volatility in its
90th percentile means nine tenths of the past sat lower. It does not mean a
decline is 90% likely, and the two get conflated the moment they share a
sentence, so `Indicator` carries the boundary text alongside the number.

*Percentiles are computed causally.* A rank against the full sample tells today
where it sits among days that have not happened. Every percentile here uses an
expanding window ending at the observation, which is also what makes the series
replayable.

*Three equal-weighted blocks are not rebuilt.* The predecessor collapsed these
dimensions into one score, and the score then hid which dimension was carrying
it and which was contradicting it. Dimensions stay separate and their
disagreements are output, not averaged away.

*The same price cannot vote twice.* Credit read from a bond ETF's move against
equities is largely the equity move again. The spread series are independent of
equity prices and are what the credit dimension uses; the ETF-relative reading
is carried as an overlapping corroboration and labelled as one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

TREND = "trend"
VOLATILITY = "volatility"
PARTICIPATION = "participation"
CREDIT = "credit"
IMPLIED = "implied_risk"
DIMENSIONS = (TREND, VOLATILITY, PARTICIPATION, CREDIT, IMPLIED)

# What each dimension may not be read as. Carried into the output because the
# misreading is always available and always tempting.
BOUNDARIES = {
    TREND: "describes trend and damage already done; says nothing about direction from here",
    VOLATILITY: "separates level from acceleration; high volatility is not a forecast of decline",
    PARTICIPATION: "a breadth proxy built from index ratios, not a count of advancing shares",
    CREDIT: "cross-dimension confirmation; the spread series are used precisely so the "
    "same equity price is not counted twice",
    IMPLIED: "the market's expected risk and the backdrop to protection cost; not the "
    "price of any actual contract",
}

DETERIORATING, IMPROVING, STABLE = "deteriorating", "improving", "stable"


@dataclass(frozen=True)
class Indicator:
    """One measurement, with everything needed to discount it.

    `percentile` is a historical rank and `percentile_meaning` says so on every
    row, because a bare 0.93 in a report becomes a probability in the reader's
    head within one paragraph.
    """

    dimension: str
    name: str
    value: float | None
    unit: str
    change_1: float | None = None
    change_5: float | None = None
    change_20: float | None = None
    percentile: float | None = None
    percentile_window: str = "expanding, history only"
    percentile_meaning: str = "share of prior observations below this one; not a probability"
    higher_is_riskier: bool = True
    coverage: float = 0.0
    as_of: pd.Timestamp | None = None
    missing_reason: str | None = None
    overlaps: str | None = None

    @property
    def direction(self) -> str:
        """Whether the risk this measures got worse over 20 sessions.

        Deliberately coarse. A finer reading would invite treating the size of
        the change as a signal, which is the claim this layer does not make.
        """
        if self.change_20 is None or not np.isfinite(self.change_20):
            return STABLE
        move = self.change_20 if self.higher_is_riskier else -self.change_20
        if abs(self.change_20) < 1e-12:
            return STABLE
        return DETERIORATING if move > 0 else IMPROVING

    @property
    def risk_rank(self) -> float | None:
        """The percentile oriented so that higher always means riskier.

        Direction alone cannot carry a label. Volatility falling from the 95th
        rank and volatility falling from the 11th are the same sign and
        completely different situations, and a rule keyed on the sign calls a
        calm market stressed the first time three series tick the wrong way.
        """
        if self.percentile is None:
            return None
        return self.percentile if self.higher_is_riskier else 1.0 - self.percentile

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "direction": self.direction, "risk_rank": self.risk_rank}


@dataclass
class MarketEvidence:
    """The indicators, and what they disagree about. No state, no leaning."""

    as_of: pd.Timestamp | None
    indicators: list[Indicator] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        if not self.indicators:
            return pd.DataFrame()
        return pd.DataFrame([i.as_dict() for i in self.indicators])

    def by_dimension(self, dimension: str) -> list[Indicator]:
        return [i for i in self.indicators if i.dimension == dimension]

    @property
    def covered_dimensions(self) -> tuple[str, ...]:
        return tuple(
            d for d in DIMENSIONS
            if any(i.value is not None for i in self.by_dimension(d))
        )

    @property
    def missing_dimensions(self) -> tuple[str, ...]:
        return tuple(d for d in DIMENSIONS if d not in self.covered_dimensions)


def causal_percentile(series: pd.Series, min_history: int = 252) -> pd.Series:
    """Rank of each observation among the ones that preceded it.

    Against the full sample, today's rank is decided partly by days that have
    not happened. That is the lookahead this project has already been caught by
    once, and it is invisible in the output: the series looks reasonable and the
    ranks are wrong in a direction that flatters whatever comes next.
    """
    clean = series.dropna()
    if clean.empty:
        return pd.Series(dtype=float, index=series.index)
    ranks = clean.expanding(min_periods=min_history).apply(
        lambda window: float((window[:-1] < window[-1]).mean()) if len(window) > 1 else np.nan,
        raw=True,
    )
    return ranks.reindex(series.index)


def _change(series: pd.Series, periods: int) -> float | None:
    clean = series.dropna()
    if len(clean) <= periods:
        return None
    return float(clean.iloc[-1] - clean.iloc[-periods - 1])


def _coverage(series: pd.Series, window: int = 252) -> float:
    tail = series.tail(window)
    return float(tail.notna().mean()) if len(tail) else 0.0


def _indicator(
    dimension: str,
    name: str,
    series: pd.Series,
    unit: str,
    higher_is_riskier: bool = True,
    min_history: int = 252,
    overlaps: str | None = None,
) -> Indicator:
    clean = series.dropna()
    if clean.empty:
        return Indicator(
            dimension, name, None, unit, higher_is_riskier=higher_is_riskier,
            missing_reason="no observations in the loaded window", overlaps=overlaps,
        )
    percentiles = causal_percentile(series, min_history)
    last_rank = percentiles.dropna()
    return Indicator(
        dimension=dimension,
        name=name,
        value=float(clean.iloc[-1]),
        unit=unit,
        change_1=_change(series, 1),
        change_5=_change(series, 5),
        change_20=_change(series, 20),
        percentile=float(last_rank.iloc[-1]) if len(last_rank) else None,
        higher_is_riskier=higher_is_riskier,
        coverage=_coverage(series),
        as_of=clean.index[-1],
        missing_reason=(
            None if len(last_rank) else f"fewer than {min_history} sessions of history"
        ),
        overlaps=overlaps,
    )


def build_evidence(
    etf_close: pd.DataFrame,
    vix: pd.DataFrame | None = None,
    macro: pd.DataFrame | None = None,
    min_history: int = 252,
) -> MarketEvidence:
    """Assemble the five dimensions from whatever is present.

    An absent dimension is absent, never zero and never imputed from a
    neighbour: "credit is unknown" and "credit is calm" are different
    statements and the report has to be able to make the first one.
    """
    indicators: list[Indicator] = []
    spy = etf_close["spy"].dropna() if "spy" in etf_close.columns else pd.Series(dtype=float)

    if not spy.empty:
        moving_average = spy.rolling(200, min_periods=200).mean()
        indicators.append(
            _indicator(
                TREND, "spy_vs_200d_ma", (spy / moving_average - 1.0) * 100.0,
                "% above the 200-session average", higher_is_riskier=False,
                min_history=min_history,
            )
        )
        indicators.append(
            _indicator(
                TREND, "spy_return_63d", (spy / spy.shift(63) - 1.0) * 100.0,
                "% over 63 sessions", higher_is_riskier=False, min_history=min_history,
            )
        )
        drawdown = (spy / spy.rolling(252, min_periods=252).max() - 1.0) * 100.0
        indicators.append(
            _indicator(
                TREND, "spy_drawdown_252d", drawdown,
                "% below the 252-session high", higher_is_riskier=False,
                min_history=min_history,
            )
        )

        returns = spy.pct_change()
        vol_20 = returns.rolling(20, min_periods=20).std() * np.sqrt(252) * 100.0
        vol_60 = returns.rolling(60, min_periods=60).std() * np.sqrt(252) * 100.0
        indicators.append(
            _indicator(VOLATILITY, "realised_vol_20d", vol_20, "% annualised",
                       min_history=min_history)
        )
        indicators.append(
            _indicator(VOLATILITY, "realised_vol_60d", vol_60, "% annualised",
                       min_history=min_history)
        )
        # Level and acceleration are different facts. A market can be quiet and
        # speeding up, or loud and settling, and one number cannot say which.
        indicators.append(
            _indicator(VOLATILITY, "vol_ratio_20_over_60", vol_20 / vol_60, "ratio",
                       min_history=min_history)
        )
        indicators.append(
            _indicator(
                VOLATILITY, "ewma_vol_lambda94",
                returns.ewm(alpha=0.06, min_periods=60).std() * np.sqrt(252) * 100.0,
                "% annualised", min_history=min_history,
            )
        )

        for peer, label in (("rsp", "rsp_over_spy"), ("iwm", "iwm_over_spy")):
            if peer in etf_close.columns:
                ratio = (etf_close[peer] / spy).dropna()
                indicators.append(
                    _indicator(
                        PARTICIPATION, label, (ratio / ratio.shift(63) - 1.0) * 100.0,
                        "% over 63 sessions", higher_is_riskier=False,
                        min_history=min_history,
                    )
                )

    if macro is not None and not macro.empty:
        for column, unit in (("hy_oas", "percentage points"), ("baa_spread", "percentage points")):
            if column in macro.columns:
                indicators.append(
                    _indicator(CREDIT, column, macro[column], unit, min_history=min_history)
                )
    if not spy.empty and {"hyg", "lqd"}.issubset(etf_close.columns):
        relative = (etf_close["hyg"] / etf_close["lqd"]).dropna()
        indicators.append(
            _indicator(
                CREDIT, "hyg_over_lqd", (relative / relative.shift(63) - 1.0) * 100.0,
                "% over 63 sessions", higher_is_riskier=False, min_history=min_history,
                overlaps="carries the same risk appetite the equity dimensions already "
                "measure; corroboration, not an independent vote",
            )
        )

    if vix is not None and "vix_close" in getattr(vix, "columns", []):
        indicators.append(
            _indicator(IMPLIED, "vix_close", vix["vix_close"], "index points",
                       min_history=min_history)
        )

    stamps = [i.as_of for i in indicators if i.as_of is not None]
    return MarketEvidence(max(stamps) if stamps else None, indicators)


def contradictions(evidence: MarketEvidence) -> list[dict[str, Any]]:
    """Dimensions moving opposite ways over the same 20 sessions.

    The plan asks which observations disagree, and a score cannot answer it -
    averaging is precisely the operation that removes the disagreement. A
    contradiction is not a problem to resolve here; it is the finding.
    """
    stance: dict[str, str] = {}
    for dimension in evidence.covered_dimensions:
        directions = {
            i.direction for i in evidence.by_dimension(dimension)
            if i.value is not None and not i.overlaps
        }
        if directions == {DETERIORATING}:
            stance[dimension] = DETERIORATING
        elif directions == {IMPROVING}:
            stance[dimension] = IMPROVING
        elif DETERIORATING in directions and IMPROVING in directions:
            stance[dimension] = "internally split"
        else:
            stance[dimension] = STABLE

    found: list[dict[str, Any]] = []
    worsening = [d for d, s in stance.items() if s == DETERIORATING]
    easing = [d for d, s in stance.items() if s == IMPROVING]
    for left in worsening:
        for right in easing:
            found.append(
                {
                    "kind": "dimensions_disagree",
                    "deteriorating": left,
                    "improving": right,
                    "detail": f"{left} worsened over 20 sessions while {right} improved; "
                    "neither reading is discounted here",
                }
            )
    for dimension, s in stance.items():
        if s == "internally split":
            found.append(
                {
                    "kind": "dimension_internally_split",
                    "dimension": dimension,
                    "detail": f"{dimension} indicators moved in opposite directions over "
                    "20 sessions, so the dimension has no single reading",
                }
            )
    return found


def review_triggers(evidence: MarketEvidence, step: float = 0.10) -> list[dict[str, Any]]:
    """What would have to change for the description to read differently.

    Question five of the product, and the only forward-looking thing this layer
    produces - which is safe because it is arithmetic on the current rank, not a
    claim that any of it will happen.
    """
    triggers = []
    for indicator in evidence.indicators:
        if indicator.percentile is None or indicator.value is None:
            continue
        rank = indicator.percentile
        triggers.append(
            {
                "indicator": indicator.name,
                "dimension": indicator.dimension,
                "current_percentile": rank,
                "review_if_above": min(1.0, rank + step),
                "review_if_below": max(0.0, rank - step),
                "detail": f"{indicator.name} sits at the {rank:.0%} rank of its own history; "
                f"a move past {min(1.0, rank + step):.0%} or {max(0.0, rank - step):.0%} "
                "would make this description stale",
            }
        )
    return triggers


def describe(evidence: MarketEvidence) -> dict[str, Any]:
    """The whole description, with what is missing named as missing."""
    return {
        "as_of": evidence.as_of,
        "covered_dimensions": list(evidence.covered_dimensions),
        "missing_dimensions": list(evidence.missing_dimensions),
        "boundaries": {d: BOUNDARIES[d] for d in evidence.covered_dimensions},
        "indicators": [i.as_dict() for i in evidence.indicators],
        "contradictions": contradictions(evidence),
        "review_triggers": review_triggers(evidence),
        "not_a_forecast": (
            "every percentile is a rank in this series' own history. None of it is "
            "a probability, and nothing here says what happens next."
        ),
    }


__all__ = [
    "BOUNDARIES",
    "DIMENSIONS",
    "Indicator",
    "MarketEvidence",
    "build_evidence",
    "causal_percentile",
    "contradictions",
    "describe",
    "review_triggers",
]

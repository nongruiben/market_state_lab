"""Cross-source checks: what a second source can settle, and what it cannot.

Two sources answer one question - did this really happen - and they answer it
in only one direction. Agreement can confirm an event that a single-source rule
found suspicious. Agreement can never clear a quarantine, because two feeds
carrying the same corrupt file agree perfectly, and a majority is not evidence
when the sources are not independent.

Three prohibitions are enforced structurally rather than documented, by simply
not providing the function that would break them:

*Nothing here returns a merged series.* There is no blend, no average, no
"pick the better source". Averaging two conflicting prices manufactures a third
that neither source reported, and choosing the source that makes a model look
healthier is how a data layer starts serving the model instead of the market.
The output is findings; substitution is a decision someone makes with a
recorded reason.

*Tolerance is measured, not chosen.* Two feeds differ normally, by an amount
that depends on the asset, the session convention and the vendor. A fixed
basis-point threshold would flag a quiet ETF and wave through a wide one, so
the band comes from the sources' own recent disagreement.

*An instrument with no second source is unverified, and says so.* Option quotes
here come from TWS alone. Consecutive snapshots of the same feed can show that
it is stable, and stability is not corroboration - labelling it "cross-checked"
would be the most expensive lie this layer could tell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from market_state_lab.data.validation import (
    DAY_END,
    NOTE,
    QUARANTINE,
    REVIEW,
    TRAINING,
    QualityIssue,
)

# Escalation order from the plan. A difference is worked through this list; it
# is never resolved by picking a winner.
ESCALATION = (
    "check timing and convention",
    "re-read the constrained source",
    "check corporate actions and trading days",
    "third source or human review",
    "source substitution or quarantine, with a recorded reason",
)


@dataclass(frozen=True)
class SourceSeries:
    """One source's closes, with the convention that makes them comparable.

    The convention is not decoration. Comparing a live price against an adjusted
    close, or a pre-market print against an RTH daily bar, produces a difference
    that is entirely real and entirely meaningless.
    """

    name: str
    values: pd.Series
    adjusted: bool = False
    session: str = "RTH"
    currency: str = "USD"

    @property
    def convention(self) -> tuple[bool, str, str]:
        return (self.adjusted, self.session, self.currency)


@dataclass
class Reconciliation:
    """Per-date comparison plus the issues it raises. Never a merged price."""

    symbol: str
    primary: str
    secondary: str
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    tolerance: float = 0.0
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def agreed(self) -> bool:
        if self.comparison.empty:
            return False
        return bool(self.comparison["within_tolerance"].all())


def measured_tolerance(
    primary: pd.Series,
    secondary: pd.Series,
    floor: float = 5e-5,
    window: int | None = 250,
    spread_multiple: float = 6.0,
    before: Any = None,
) -> float:
    """The band the two sources normally disagree by, as a relative difference.

    Taken from their own history rather than assumed, because a quiet ETF and a
    thin one do not disagree by the same amount and one threshold cannot serve
    both. The floor keeps a pair that has agreed perfectly so far from setting a
    tolerance of zero and flagging the first rounding difference.

    Median and MAD rather than a high quantile, and `before` rather than the
    whole series, because a quantile lets the outlier set the band that is
    supposed to judge it. A single spurious 34% move raised a 99th-percentile
    tolerance to 68% and then passed itself - an observation choosing its own
    threshold, which is the failure this project rules out everywhere else.
    Neither the median nor the MAD can be moved by one point.
    """
    common = primary.dropna().index.intersection(secondary.dropna().index)
    if before is not None:
        common = common[common < before]
    if window:
        common = common[-window:]
    if len(common) < 2:
        return floor
    diff = (primary.loc[common] / secondary.loc[common] - 1.0).abs()
    median = float(diff.median())
    mad = float((diff - median).abs().median())
    return max(median + spread_multiple * mad, floor)


def reconcile_closes(
    symbol: str,
    primary: SourceSeries,
    secondary: SourceSeries,
    tolerance: float | None = None,
) -> Reconciliation:
    """Compare two sources' closes and report what disagrees.

    Conventions are checked before values: a mismatch there makes every number
    below it meaningless, so it stops the comparison rather than colouring it.
    """
    issues: list[QualityIssue] = []
    if primary.convention != secondary.convention:
        issues.append(
            QualityIssue(
                "convention_mismatch", QUARANTINE, symbol,
                f"{primary.name} is {primary.convention} and {secondary.name} is "
                f"{secondary.convention}; the difference between them would be real "
                "and meaningless",
                (TRAINING, DAY_END),
            )
        )
        return Reconciliation(symbol, primary.name, secondary.name, issues=issues)

    band = tolerance if tolerance is not None else measured_tolerance(
        primary.values, secondary.values
    )
    common = primary.values.dropna().index.intersection(secondary.values.dropna().index)
    only_primary = primary.values.dropna().index.difference(secondary.values.dropna().index)
    only_secondary = secondary.values.dropna().index.difference(primary.values.dropna().index)

    for stamp in only_primary:
        issues.append(
            QualityIssue(
                "date_only_in_primary", NOTE, f"{symbol} {stamp}",
                f"{primary.name} has this session and {secondary.name} does not; "
                "check the trading calendar before treating it as a gap",
            )
        )
    for stamp in only_secondary:
        issues.append(
            QualityIssue(
                "date_only_in_secondary", NOTE, f"{symbol} {stamp}",
                f"{secondary.name} has this session and {primary.name} does not",
            )
        )

    if common.empty:
        issues.append(
            QualityIssue(
                "no_overlap", REVIEW, symbol,
                f"{primary.name} and {secondary.name} share no dated observation, "
                "so neither confirms the other",
                (),
            )
        )
        return Reconciliation(symbol, primary.name, secondary.name, tolerance=band, issues=issues)

    left = primary.values.loc[common]
    right = secondary.values.loc[common]
    relative = (left / right - 1.0)
    comparison = pd.DataFrame(
        {
            primary.name: left,
            secondary.name: right,
            "relative_difference": relative,
            "within_tolerance": relative.abs() <= band,
        }
    )
    for stamp in comparison.index[~comparison["within_tolerance"]]:
        row = comparison.loc[stamp]
        issues.append(
            QualityIssue(
                "sources_disagree", REVIEW, f"{symbol} {stamp}",
                f"{primary.name} {row[primary.name]:.4f} against {secondary.name} "
                f"{row[secondary.name]:.4f} is {row['relative_difference']:+.3%}, outside "
                f"the {band:.3%} the two normally differ by. Work the escalation: "
                + " -> ".join(ESCALATION),
                (),
            )
        )
    return Reconciliation(symbol, primary.name, secondary.name, comparison, band, issues)


def confirm_move(
    reconciliation: Reconciliation,
    stamp: Any,
    primary: SourceSeries,
    secondary: SourceSeries,
) -> QualityIssue:
    """Did the second source see the same move? Fault-injection row 2.

    A large move that both sources report is an event, and the record has to say
    so, because the next temptation is to smooth it away. A move only one source
    reports is not thereby fake either - it escalates.
    """
    subject = f"{reconciliation.symbol} {stamp}"
    moves = {}
    for source in (primary, secondary):
        series = source.values.dropna()
        position = series.index.get_indexer([stamp])[0]
        if position < 1:
            return QualityIssue(
                "move_unconfirmable", REVIEW, subject,
                f"{source.name} has no prior observation, so the move cannot be compared",
                (),
            )
        moves[source.name] = float(series.iloc[position] / series.iloc[position - 1] - 1.0)

    a, b = moves[primary.name], moves[secondary.name]
    # Measured strictly before the move, so the thing being judged has no say in
    # the band that judges it.
    band = measured_tolerance(primary.values, secondary.values, before=stamp)
    if abs(a - b) <= max(band * 2, 0.005):
        return QualityIssue(
            "move_confirmed", NOTE, subject,
            f"both sources report {a:+.1%}; this is an event, not an outlier, and "
            "nothing downstream may filter it away",
        )
    return QualityIssue(
        "move_unconfirmed", REVIEW, subject,
        f"{primary.name} reports {a:+.1%} and {secondary.name} reports {b:+.1%}; "
        "one of them is wrong and neither is automatically the loser. "
        + " -> ".join(ESCALATION),
        (),
    )


def agreement_cannot_clear(
    reconciliation: Reconciliation,
    single_source_issues: list[QualityIssue],
) -> list[QualityIssue]:
    """Fault-injection row 9: a shared anomaly is not a pass.

    Two feeds that agree perfectly may be two copies of one mistake - a vendor
    both of them resell, a file both of them loaded. So agreement is additive
    only: the quarantines a single-source rule raised come back unchanged, and a
    note records that corroboration was offered and refused.
    """
    quarantines = [i for i in single_source_issues if i.severity == QUARANTINE]
    if not quarantines:
        return list(single_source_issues)
    return [
        *single_source_issues,
        QualityIssue(
            "agreement_does_not_clear_quarantine", NOTE, reconciliation.symbol,
            f"{reconciliation.primary} and {reconciliation.secondary} agree, and "
            f"{len(quarantines)} quarantine(s) stand: two sources carrying one "
            "corrupt input agree perfectly, so a majority is not evidence when "
            "they are not independent",
        ),
    ]


def unverifiable(subject: str, reason: str) -> QualityIssue:
    """Say plainly that nothing corroborated this.

    Used for option quotes, which arrive from TWS alone. The absence of a check
    has to appear in the record, or a reader will assume the check passed.
    """
    return QualityIssue("no_independent_source", NOTE, subject, reason)


def stability_only(subject: str, observations: int, spread: float) -> QualityIssue:
    """Repeated reads of one feed. Stability, explicitly not corroboration."""
    return QualityIssue(
        "same_source_stability", NOTE, subject,
        f"{observations} reads of the same feed span {spread:.3%}; this shows the "
        "feed is steady and corroborates nothing, because a source cannot confirm "
        "itself",
    )


def substitution_record(
    symbol: str,
    replaced: str,
    replacement: str,
    reason: str,
) -> dict[str, Any]:
    """The note a source swap must carry. There is no automatic swap to go with it.

    Deliberately a record and not an action: a replacement has to pass the same
    gates as the source it replaces, and choosing it is a judgement someone makes
    and signs, not a fallback the pipeline takes when a number looks wrong.
    """
    if not reason.strip():
        raise ValueError(f"{symbol}: a substitution without a recorded reason is not allowed")
    return {
        "symbol": symbol,
        "replaced_source": replaced,
        "replacement_source": replacement,
        "substitution_reason": reason,
        "must_pass_same_gates": True,
    }


def summarise(reconciliation: Reconciliation) -> dict[str, Any]:
    """What was compared and what disagreed. No score, and no verdict on truth."""
    comparison = reconciliation.comparison
    worst = None
    if not comparison.empty:
        index = comparison["relative_difference"].abs().idxmax()
        worst = {
            "date": str(index),
            "relative_difference": float(comparison.loc[index, "relative_difference"]),
        }
    return {
        "symbol": reconciliation.symbol,
        "sources": [reconciliation.primary, reconciliation.secondary],
        "compared_days": int(len(comparison)),
        "tolerance": reconciliation.tolerance,
        "days_outside_tolerance": int((~comparison["within_tolerance"]).sum())
        if not comparison.empty
        else 0,
        "worst_difference": worst,
        "agreed": reconciliation.agreed,
    }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "ESCALATION",
    "Reconciliation",
    "SourceSeries",
    "agreement_cannot_clear",
    "confirm_move",
    "measured_tolerance",
    "reconcile_closes",
    "stability_only",
    "substitution_record",
    "summarise",
    "unverifiable",
]

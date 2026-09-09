"""Field and internal-consistency checks, and the purposes a failure blocks.

The output is a list of issues with reasons, never a score. A single number
would average a structurally impossible bar together with a wide spread, and
the whole point of gating by purpose is that different failures block different
things: a crossed book stops an option being costed and says nothing about
whether the day's close can be described.

Three restraints shape almost every rule here, and each is a way of not
destroying evidence:

*Suspicious is not invalid.* A large move goes to review, never to deletion. A
z-score or a fixed percentage would delete exactly the crashes the project
exists to notice, and a split, a dividend, a real crash and a bad print are
four different causes that need separating rather than filtering.

*Absent is not zero.* A missing volume, an unreturned open interest and a
sentinel are recorded as unknown. Turning them into 0 invents a fact - no
trades, nobody holding - that nobody observed.

*One security type's rules are not another's.* An index has no volume and that
is normal; an option's bid can genuinely be zero; a spread can be negative.
Applying the stock rules everywhere would quarantine correct data.

What this module deliberately does not do: repair. Nothing is rescaled,
back-filled, averaged with a second source, or nudged onto a pricing identity.
A value that fails is marked and the reason is kept; deciding what to do about
it belongs to whoever reads the issues.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

# What a failure can take away, mirroring the snapshot's purposes.
TRAINING = "training"
DAY_END = "day_end_analysis"
INTRADAY = "intraday_observation"
QUOTES = "instrument_quotes"
ALL_PURPOSES = (TRAINING, DAY_END, INTRADAY, QUOTES)

# Severity is what happens next, not how bad it feels.
QUARANTINE = "quarantine"  # the row does not enter new features or judgments
REVIEW = "review"  # suspicious enough to look at, never deleted for it
NOTE = "note"  # recorded so a reader can discount it; blocks nothing

# Security types whose bars carry a share count. An index has no volume and a
# zero there is the truth, not a defect.
VOLUME_BEARING = frozenset({"STK", "ETF", "OPT", "FUT"})


@dataclass(frozen=True)
class QualityIssue:
    """One finding, with the reason and what it costs.

    `blocks` is empty for a note. It is the mechanism by which a failure in one
    field stops the uses that field supports without stopping the others.
    """

    code: str
    severity: str
    subject: str
    detail: str
    blocks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.severity not in {QUARANTINE, REVIEW, NOTE}:
            raise ValueError(f"{self.code}: unknown severity {self.severity!r}")
        unknown = set(self.blocks) - set(ALL_PURPOSES)
        if unknown:
            raise ValueError(f"{self.code}: unknown purposes {sorted(unknown)}")
        if self.severity == NOTE and self.blocks:
            raise ValueError(f"{self.code}: a note blocks nothing; use review or quarantine")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def issues_frame(issues: list[QualityIssue]) -> pd.DataFrame:
    if not issues:
        return pd.DataFrame(columns=["code", "severity", "subject", "detail", "blocks"])
    return pd.DataFrame([i.as_dict() for i in issues])


def blocked_purposes(issues: list[QualityIssue]) -> set[str]:
    """Everything any issue takes away. Nothing here is majority-voted."""
    blocked: set[str] = set()
    for issue in issues:
        blocked |= set(issue.blocks)
    return blocked


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _scale_factor(ratio: float) -> float | None:
    """The power of ten a ratio sits on, if it sits on one.

    A market can halve. It does not move by a factor of exactly one hundred, so
    a close that lands within a whisker of 10^k against the previous one is a
    units error rather than a price - which is the difference between
    quarantining a mistake and deleting a crash.
    """
    if not _finite(ratio) or ratio <= 0:
        return None
    exponent = round(math.log10(ratio))
    if exponent == 0:
        return None
    power = 10.0**exponent
    return power if abs(ratio / power - 1.0) <= 0.02 else None


def validate_daily_bars(
    frame: pd.DataFrame,
    symbol: str,
    sec_type: str = "STK",
    jump_review_threshold: float = 0.20,
) -> list[QualityIssue]:
    """Structural checks on a daily bar series, indexed by date.

    Bounds violations are quarantined because they are impossible rather than
    unlikely: a high below its low describes no market. A move on a decimal
    factor is quarantined for the same reason - it is arithmetic, not news.
    Everything else large is sent to review with its number attached, because a
    filter tuned to remove bad prints removes crashes too.
    """
    issues: list[QualityIssue] = []
    if frame.empty:
        return issues

    required = [c for c in ("open", "high", "low", "close") if c in frame.columns]
    for column in required:
        bad = frame.index[~frame[column].map(_finite)]
        for stamp in bad:
            issues.append(
                QualityIssue(
                    "price_not_finite", QUARANTINE, f"{symbol} {stamp}",
                    f"{column} is not a finite number",
                    (TRAINING, DAY_END, QUOTES),
                )
            )
        if sec_type in {"STK", "ETF", "FUT"}:
            nonpositive = frame.index[frame[column].map(_finite) & (frame[column] <= 0)]
            for stamp in nonpositive:
                issues.append(
                    QualityIssue(
                        "price_not_positive", QUARANTINE, f"{symbol} {stamp}",
                        f"{column} is {frame.loc[stamp, column]}, and a share price is not",
                        (TRAINING, DAY_END, QUOTES),
                    )
                )

    if {"open", "high", "low", "close"}.issubset(frame.columns):
        usable = frame[["open", "high", "low", "close"]].map(_finite).all(axis=1)
        candidates = frame.loc[usable]
        inverted = candidates.index[candidates["high"] < candidates["low"]]
        for stamp in inverted:
            row = candidates.loc[stamp]
            issues.append(
                QualityIssue(
                    "ohlc_inverted", QUARANTINE, f"{symbol} {stamp}",
                    f"high {row['high']} is below low {row['low']}; no market did that",
                    (TRAINING, DAY_END, QUOTES),
                )
            )
        for edge in ("open", "close"):
            outside = candidates.index[
                (candidates[edge] > candidates["high"]) | (candidates[edge] < candidates["low"])
            ]
            for stamp in outside:
                row = candidates.loc[stamp]
                issues.append(
                    QualityIssue(
                        "ohlc_bounds_violated", QUARANTINE, f"{symbol} {stamp}",
                        f"{edge} {row[edge]} sits outside [{row['low']}, {row['high']}]",
                        (TRAINING, DAY_END, QUOTES),
                    )
                )

    if "close" in frame.columns and len(frame) > 1:
        closes = frame["close"]
        ratios = closes / closes.shift(1)
        for stamp, ratio in ratios.items():
            if not _finite(ratio) or ratio <= 0:
                continue
            factor = _scale_factor(float(ratio))
            if factor is not None:
                issues.append(
                    QualityIssue(
                        "price_scale_break", QUARANTINE, f"{symbol} {stamp}",
                        f"close moved by a factor of {factor:g} against the previous bar; "
                        "a decimal factor is a units error, not a price",
                        (TRAINING, DAY_END, QUOTES),
                    )
                )
            elif abs(float(ratio) - 1.0) > jump_review_threshold:
                # Review, never deletion: a split, a dividend, a real crash and
                # a bad print all look like this, and only three of them are
                # errors.
                issues.append(
                    QualityIssue(
                        "large_price_move", REVIEW, f"{symbol} {stamp}",
                        f"close moved {float(ratio) - 1.0:+.1%} against the previous bar; "
                        "attribute to a split, a distribution, a real move or a bad print "
                        "before using it",
                        (),
                    )
                )

    issues += _duplicate_dates(frame, symbol)
    issues += _volume_units(frame, symbol, sec_type)
    return issues


def _duplicate_dates(frame: pd.DataFrame, symbol: str) -> list[QualityIssue]:
    """Identical repeats are harmless; differing ones are a version conflict.

    Keeping the last would silently choose between two answers about one day.
    """
    issues: list[QualityIssue] = []
    duplicated = frame.index[frame.index.duplicated(keep=False)]
    for stamp in dict.fromkeys(duplicated):
        rows = frame.loc[[stamp]]
        if rows.drop_duplicates().shape[0] == 1:
            issues.append(
                QualityIssue(
                    "duplicate_row_identical", NOTE, f"{symbol} {stamp}",
                    f"{len(rows)} identical rows for one date; safe to de-duplicate",
                )
            )
        else:
            issues.append(
                QualityIssue(
                    "duplicate_row_conflict", QUARANTINE, f"{symbol} {stamp}",
                    f"{len(rows)} rows for one date disagree; this is a revision "
                    "conflict, not a row to keep the last of",
                    (TRAINING, DAY_END, QUOTES),
                )
            )
    return issues


def _volume_units(frame: pd.DataFrame, symbol: str, sec_type: str) -> list[QualityIssue]:
    """Volume needs a declared unit, and an absent one is not a zero."""
    issues: list[QualityIssue] = []
    if "volume" not in frame.columns:
        return issues
    unit = frame.attrs.get("volume_unit")
    if unit is None:
        issues.append(
            QualityIssue(
                "volume_unit_undeclared", NOTE, symbol,
                "volume carries no declared unit; shares, contracts and lots are "
                "not interchangeable and IBKR's counts differ from other sources",
            )
        )
    missing = frame.index[~frame["volume"].map(_finite)]
    for stamp in missing:
        issues.append(
            QualityIssue(
                "volume_missing", NOTE, f"{symbol} {stamp}",
                "volume is absent; absent is not a zero and no trades is not implied",
            )
        )
    if sec_type not in VOLUME_BEARING:
        return issues
    zeros = frame.index[frame["volume"].map(_finite) & (frame["volume"] == 0)]
    for stamp in zeros:
        # A genuinely untraded session exists. It is worth seeing and is not a
        # reason to reject the bar.
        issues.append(
            QualityIssue(
                "volume_zero", NOTE, f"{symbol} {stamp}",
                "zero volume; real for a thin listing, so recorded rather than rejected",
            )
        )
    return issues


def validate_quotes(frame: pd.DataFrame, frozen_book: bool = False) -> list[QualityIssue]:
    """Book-level checks on the quote rows the client returns.

    A crossed book goes to review rather than rejection on sight: the two sides
    of a stream update independently, and a momentary inversion is a sampling
    artefact. Only a caller that sees it persist should reject.

    `frozen_book` collapses the last-outside-book check to a single line. On a
    frozen feed the last trade is intraday and the book is the close, so every
    leg trips it - seven identical notes on healthy data, which is how a quality
    log teaches its reader to skip it. The check is still reported, once, with
    the count.
    """
    issues: list[QualityIssue] = []
    if frame.empty:
        return issues
    outside_count = 0
    for row in frame.to_dict("records"):
        subject = f"{row.get('symbol')} {row.get('expiry') or ''} {row.get('strike') or ''}".strip()
        bid, ask, last = row.get("bid"), row.get("ask"), row.get("last")
        if _finite(bid) and _finite(ask) and float(ask) < float(bid):
            issues.append(
                QualityIssue(
                    "crossed_book", REVIEW, subject,
                    f"ask {ask} below bid {bid}; re-read before rejecting, since the "
                    "two sides update independently",
                    (),
                )
            )
        if _finite(bid) and float(bid) == 0.0:
            issues.append(
                QualityIssue(
                    "zero_bid", NOTE, subject,
                    "bid is zero, which can be real; it is a liquidity fact, not an error",
                )
            )
        if _finite(last) and _finite(bid) and _finite(ask):
            if not (float(bid) <= float(last) <= float(ask)):
                outside_count += 1
                if not frozen_book:
                    # An older trade sits outside the current book all the time.
                    issues.append(
                        QualityIssue(
                            "last_outside_book", NOTE, subject,
                            f"last {last} is outside [{bid}, {ask}]; usually an earlier "
                            "trade, and never a reason to correct last",
                        )
                    )
        implied = row.get("implied_volatility")
        if implied is not None and pd.notna(implied) and float(implied) <= 0:
            issues.append(
                QualityIssue(
                    "implied_vol_sentinel", NOTE, subject,
                    f"implied volatility {implied} is a sentinel, not a number",
                )
            )
    if frozen_book and outside_count:
        issues.append(
            QualityIssue(
                "last_outside_frozen_book", NOTE, "quote set",
                f"last sits outside the book on {outside_count} of {len(frame)} legs, "
                "which is what a frozen book is: the last trade is intraday and the "
                "book is the close",
            )
        )
    return issues


def quarantined_subjects(issues: list[QualityIssue]) -> set[str]:
    """Subjects that must not reach a feature, a judgment, or a recommendation."""
    return {i.subject for i in issues if i.severity == QUARANTINE}


def drop_quarantined(frame: pd.DataFrame, issues: list[QualityIssue], symbol: str) -> pd.DataFrame:
    """The rows that may go downstream. The excluded ones stay in the issues.

    This removes from the *input to the next stage*, not from the archive. The
    raw payload is immutable and the quarantined row keeps its reason, so the
    exclusion can always be examined instead of merely trusted.
    """
    if frame.empty:
        return frame
    blocked = quarantined_subjects(issues)
    if not blocked:
        return frame
    keep = [stamp for stamp in frame.index if f"{symbol} {stamp}" not in blocked]
    return frame.loc[keep]


def crisis_reading_is_allowed(issues: list[QualityIssue]) -> bool:
    """May a market-stress reading be formed from data carrying these issues?

    Fault-injection row 1: a hundredfold price and an inverted bar must not
    become a crisis signal. The rule is not "large moves are suspicious" - row 2
    insists a confirmed crash survives - but that a structurally impossible or
    decimally rescaled record cannot support any reading at all.
    """
    return not any(
        i.severity == QUARANTINE
        and i.code in {"price_scale_break", "ohlc_inverted", "ohlc_bounds_violated",
                       "price_not_finite", "price_not_positive"}
        for i in issues
    )


def summarise(issues: list[QualityIssue]) -> dict[str, Any]:
    """Counts by severity and the purposes lost. Deliberately not a score."""
    counts: dict[str, int] = {}
    for issue in issues:
        counts[issue.severity] = counts.get(issue.severity, 0) + 1
    return {
        "counts": counts,
        "blocked_purposes": sorted(blocked_purposes(issues)),
        "quarantined_subjects": sorted(quarantined_subjects(issues)),
        "crisis_reading_allowed": crisis_reading_is_allowed(issues),
    }


__all__ = [
    "ALL_PURPOSES",
    "NOTE",
    "QUARANTINE",
    "REVIEW",
    "QualityIssue",
    "blocked_purposes",
    "crisis_reading_is_allowed",
    "drop_quarantined",
    "issues_frame",
    "quarantined_subjects",
    "summarise",
    "validate_daily_bars",
    "validate_quotes",
]

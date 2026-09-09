"""The assessment contract, and a judgment layer deliberately left unpopulated.

The five questions the product answers split cleanly. What changed, what the
evidence is and over what horizon, which instruments and at what cost, and what
would make the reading stale - none of those need a forecast, and all of them
are built. Only "keep watching, lean toward reducing risk, or insufficient
evidence" needs one.

That ability was measured on this project's own data and is absent. At twenty
days the model scored a Brier of 0.5721 against a persistence benchmark's
0.5134; building on persistence made it worse by 0.0995 to 0.126, every
comparison significant and every one in the wrong direction; the drawdown edge
disappeared under an exposure-matched control; at 126 days no predictor was
significant after 2013; five feature families returned null.

So the leaning is pre-registered: `DATA_INSUFFICIENT`, with the reason attached,
until an evidence source with demonstrated information is admitted through the
promotion gate. This is a recorded expectation rather than a failure, and the
tests assert it as the expected result. The seam is real - `RuleSet` and
`promote` exist, and a rule that passes the gate changes the answer - but no
rule has passed it, and none is asserted to.

What this file refuses to do is the thing that killed the predecessor: dress a
plausible rule as a validated one. A rule that is interpretable, sensible and
untested is untested, and it is marked that way in every output it touches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from market_state_lab.market_evidence import (
    DETERIORATING,
    MarketEvidence,
    contradictions,
    describe,
    review_triggers,
)

# Co-existing labels, not a single regime. A market can be repairing on one
# dimension while damage persists on another, and a single state would have to
# pick one and hide the other.
TREND_DAMAGED = "trend_damaged"
VOLATILITY_ELEVATED = "volatility_elevated"
STRESS_SPREADING = "stress_spreading"
REPAIR_UNDERWAY = "repair_underway"
STATE_LABELS = (TREND_DAMAGED, VOLATILITY_ELEVATED, STRESS_SPREADING, REPAIR_UNDERWAY)

# The leanings the contract allows. Only the last is ever returned today.
NO_NEW_DEFENSE_CASE = "NO_NEW_DEFENSE_CASE"
WATCH = "WATCH"
REVIEW_REDUCTION = "REVIEW_REDUCTION"
REVIEW_HEDGE = "REVIEW_HEDGE"
CONFLICTED = "CONFLICTED"
DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
LEANINGS = (
    NO_NEW_DEFENSE_CASE, WATCH, REVIEW_REDUCTION, REVIEW_HEDGE, CONFLICTED, DATA_INSUFFICIENT
)

# Research grades. Nothing in this project has ever reached VALIDATED, and the
# grade is printed rather than assumed so that cannot be forgotten.
UNTESTED = "rule judgement, gain unvalidated"
REPLAYED = "replayed, gain measured"
VALIDATED = "validated against a strong benchmark, out of sample"


@dataclass(frozen=True)
class RuleSet:
    """Pre-registered thresholds, frozen before anyone looks at the outcome.

    The predecessor chose three benchmarks in turn, each one it could beat.
    Freezing the numbers before seeing results is what makes the next attempt
    falsifiable rather than a search for a flattering comparison.
    """

    name: str
    frozen_at: str
    trend_damaged_below_ma: float = 0.0
    volatility_accelerating_ratio_percentile: float = 0.90
    stress_percentile: float = 0.90
    confirm_sessions: int = 3
    release_sessions: int = 5
    grade: str = UNTESTED

    def __post_init__(self) -> None:
        if self.grade not in {UNTESTED, REPLAYED, VALIDATED}:
            raise ValueError(f"{self.name}: unknown research grade {self.grade!r}")


DEFAULT_RULES = RuleSet(name="first-version", frozen_at="2026-09-09")


@dataclass
class Assessment:
    """The output contract of section 9.1, with nothing optional about the caveats."""

    as_of: pd.Timestamp | None
    snapshot_id: str | None
    observation_horizon: str
    state_labels: list[str] = field(default_factory=list)
    risk_basis: list[str] = field(default_factory=list)
    contrary_evidence: list[str] = field(default_factory=list)
    action_leaning: str = DATA_INSUFFICIENT
    leaning_reason: str = ""
    applicable_conditions: list[str] = field(default_factory=list)
    research_grade: str = UNTESTED
    triggers: list[dict[str, Any]] = field(default_factory=list)
    next_review_reason: str = ""
    missing_dimensions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.action_leaning not in LEANINGS:
            raise ValueError(f"unknown action leaning {self.action_leaning!r}")
        unknown = set(self.state_labels) - set(STATE_LABELS)
        if unknown:
            raise ValueError(f"unknown state labels {sorted(unknown)}")
        if self.action_leaning != DATA_INSUFFICIENT and self.research_grade == UNTESTED:
            # The one thing that killed the predecessor: an interpretable rule
            # presented as a working one. A leaning may only leave
            # DATA_INSUFFICIENT behind a grade that was earned.
            raise ValueError(
                f"{self.action_leaning} requires a research grade above '{UNTESTED}'; "
                "an untested rule may describe, never lean"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def state_labels(evidence: MarketEvidence, rules: RuleSet = DEFAULT_RULES) -> list[str]:
    """Which descriptions currently fit. Several may, and that is the point.

    These are labels on what has already happened, not predictions, so they are
    produced regardless of research grade - saying "the trend is damaged" when
    price sits below its long average is a reading of the chart, not a forecast.
    """
    labels: list[str] = []
    by_name = {i.name: i for i in evidence.indicators if i.value is not None}

    ma = by_name.get("spy_vs_200d_ma")
    if ma is not None and ma.value is not None and ma.value < rules.trend_damaged_below_ma:
        labels.append(TREND_DAMAGED)

    ratio = by_name.get("vol_ratio_20_over_60")
    if (
        ratio is not None
        and ratio.percentile is not None
        and ratio.percentile >= rules.volatility_accelerating_ratio_percentile
    ):
        labels.append(VOLATILITY_ELEVATED)

    # A label needs a level, not just a sign. Three series ticking the wrong way
    # from the calmest decile of their history is not spreading stress, and a
    # rule that says it is teaches the reader to ignore the label. The plan names
    # the dimensions too: stress spreads when participation or credit joins in.
    def _notable(dimension: str, wanted: str) -> bool:
        return any(
            i.dimension == dimension
            and i.value is not None
            and not i.overlaps
            and i.direction == wanted
            and i.risk_rank is not None
            and (i.risk_rank >= rules.stress_percentile if wanted == DETERIORATING
                 else i.risk_rank <= 1.0 - rules.stress_percentile)
            for i in evidence.indicators
        )

    worsening = {d for d in evidence.covered_dimensions if _notable(d, DETERIORATING)}
    if worsening & {"participation", "credit"} and len(worsening) >= 2:
        labels.append(STRESS_SPREADING)

    easing = {d for d in evidence.covered_dimensions if _notable(d, "improving")}
    if len(easing) >= 2 and not worsening:
        labels.append(REPAIR_UNDERWAY)
    return labels


def derive_leaning(
    labels: set[str],
    rules: RuleSet = DEFAULT_RULES,
) -> tuple[str, str]:
    """The section 9.2 mapping from labels to a conditional leaning.

    Only reached when the rules carry a measured grade; at `UNTESTED` the
    answer is pre-registered as insufficient and this function is not consulted.
    The mapping is coarse on purpose: it turns descriptions into "consider
    looking", never into "the evidence proves you should trade".

    - Trend damaged AND one independent dimension also deteriorated: consider
      reducing market exposure, protection compared alongside.
    - Short-term stress without long-term damage: compare limited-term
      protection against watching.
    - Repair underway while damage persists: conflicted, read both.
    - Repair with nothing worsening: watch; no automatic re-entry signal.
    - Nothing notable: no new defense case, which is not a forecast of safety.
    """
    worsening = TREND_DAMAGED in labels or VOLATILITY_ELEVATED in labels or STRESS_SPREADING in labels
    if TREND_DAMAGED in labels and len(labels & {VOLATILITY_ELEVATED, STRESS_SPREADING}) >= 1:
        return REVIEW_REDUCTION, (
            "trend damage is confirmed by a second, independent dimension; "
            "consider lowering market exposure, with protection compared alongside"
        )
    # Conflict is checked before the hedge branch: repair plus any standing
    # damage is a contradiction, whatever the damage is.
    if REPAIR_UNDERWAY in labels and worsening:
        return CONFLICTED, "repair signals conflict with still-standing damage; read both"
    if VOLATILITY_ELEVATED in labels and TREND_DAMAGED not in labels and STRESS_SPREADING not in labels:
        return REVIEW_HEDGE, (
            "short-term stress is elevated while the long trend is not damaged; "
            "compare limited-term protection against watching"
        )
    if worsening:
        return WATCH, "one dimension is notable; not yet enough to change exposure"
    if REPAIR_UNDERWAY in labels:
        return WATCH, "risk has eased; this is not a re-entry signal"
    return NO_NEW_DEFENSE_CASE, (
        "nothing notable deteriorated; this is not a forecast of safety, "
        "and uncovered risks remain"
    )


def assess(
    evidence: MarketEvidence,
    rules: RuleSet = DEFAULT_RULES,
    snapshot_id: str | None = None,
    observation_horizon: str = "20 sessions",
) -> Assessment:
    """Describe the state, and lean only behind a grade that was earned.

    At `UNTESTED` the leaning is `DATA_INSUFFICIENT` by pre-registration, not
    by accident and not because today's data was thin. A rule set that has been
    promoted to a measured grade answers through `derive_leaning`.
    """
    disagreements = contradictions(evidence)
    labels = state_labels(evidence, rules)
    # Oriented rank: for indicators where lower is riskier, the raw percentile
    # would show 70% in the list of things supporting a risk reading when the
    # oriented risk rank is 30% - the exact misreading the Indicator boundary
    # text warns about.
    basis = [
        f"{i.name} risk rank {i.risk_rank:.0%} "
        f"({i.value:.2f} {i.unit}, {i.percentile:.0%} of history more favourable)"
        for i in evidence.indicators
        if i.value is not None and i.risk_rank is not None and i.direction == DETERIORATING
    ]
    contrary = [
        f"{i.name} improved over 20 sessions"
        for i in evidence.indicators
        if i.value is not None and i.direction == "improving"
    ] + [d["detail"] for d in disagreements]

    leaning = DATA_INSUFFICIENT
    reason = (
        "Pre-registered. The forecasting ability a leaning needs was measured on "
        "this project's data and is absent: Brier 0.5721 against persistence at "
        "0.5134 over 20 days, the drawdown edge gone under an exposure-matched "
        "control, no predictor significant after 2013 at 126 days, five feature "
        "families null. This is the expected answer, not a shortfall of today's "
        "data, and it changes only when a rule passes the promotion gate."
    )
    if rules.grade != UNTESTED:
        leaning, reason = derive_leaning(set(labels), rules)

    return Assessment(
        as_of=evidence.as_of,
        snapshot_id=snapshot_id,
        observation_horizon=observation_horizon,
        state_labels=labels,
        risk_basis=basis,
        contrary_evidence=contrary,
        action_leaning=leaning,
        leaning_reason=reason,
        applicable_conditions=[
            "Reducing exposure and buying protection answer different preferences: "
            "how much upside you are willing to give up, and how much you will pay "
            "to keep it. Market data cannot choose between them.",
            "Nothing here reads a position, so none of it is advice about holdings.",
        ],
        research_grade=rules.grade,
        triggers=review_triggers(evidence),
        next_review_reason=(
            "the description goes stale when any indicator moves out of the rank band "
            "listed in triggers"
        ),
        missing_dimensions=list(evidence.missing_dimensions),
    )


def promote(rules: RuleSet, grade: str, evidence_of_gain: str) -> RuleSet:
    """The seam. A rule leaves 'untested' only with measured gain attached.

    Deliberately requires the evidence text: a promotion with nothing to point
    at is how a rule becomes validated by having been around a while.
    """
    if grade not in {REPLAYED, VALIDATED}:
        raise ValueError(f"promotion must be to a measured grade, not {grade!r}")
    if not evidence_of_gain.strip():
        raise ValueError(f"{rules.name}: promotion needs the measurement that earned it")
    return RuleSet(**{**asdict(rules), "grade": grade})


def report(evidence: MarketEvidence, assessment: Assessment) -> dict[str, Any]:
    """Description and assessment together, in the order the questions are asked."""
    return {
        "what_changed": describe(evidence),
        "assessment": assessment.as_dict(),
        "reading_order": [
            "what changed and what disagrees",
            "the leaning, which is pre-registered as insufficient evidence",
            "the evidence for and against, and over what horizon",
            "instrument candidates and their cost, from the protection table",
            "what would make this stale",
        ],
    }


__all__ = [
    "DATA_INSUFFICIENT",
    "DEFAULT_RULES",
    "LEANINGS",
    "STATE_LABELS",
    "UNTESTED",
    "Assessment",
    "RuleSet",
    "assess",
    "derive_leaning",
    "promote",
    "report",
    "state_labels",
]

"""The promotion ladder, and the three capabilities it grades separately.

This exists because the predecessor failed here specifically. It chose three
benchmarks in turn, each one it could beat; it called an interpretable rule an
effective one; and it reported a drawdown improvement that turned out to be
de-leveraging. None of those were coding mistakes. They were a research process
with no gate in it.

So the gate is code. Four rungs, climbed in order, and no rung may be skipped:

    computed correctly -> replayed and explainable -> frozen shadow output
    -> tested on data that did not exist when the rule was frozen

The last rung is the only one that produces forward evidence, and it is the
only one that cannot be reached by trying harder - it needs time to pass.

Three capabilities are graded apart, because they fail apart. A data layer that
catches contamination says nothing about whether a rule sees risk coming, and a
rule that sees risk coming says nothing about whether acting on it pays after
costs. "The description is plausible" promotes nothing at all.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# The three capabilities of 12.1. Each has its own evidence and its own grade.
DATA_QUALITY = "data_quality"
RISK_IDENTIFICATION = "risk_identification"
ACTION_VALUE = "action_value"
CAPABILITIES = (DATA_QUALITY, RISK_IDENTIFICATION, ACTION_VALUE)

# The ladder. Index order is promotion order, and skipping is refused.
COMPUTED = "computed_correctly"
REPLAYED = "replayed_and_explainable"
SHADOW = "frozen_shadow_output"
FORWARD = "tested_on_new_data"
LADDER = (COMPUTED, REPLAYED, SHADOW, FORWARD)

# Outcomes of a comparison. "Not significant" is not "equivalent", and the
# difference is where the predecessor's three headline claims lived.
PASSED = "passed"
EVIDENCE_OF_HARM = "evidence_of_harm"
INSUFFICIENT = "insufficient_evidence"
OUTCOMES = (PASSED, EVIDENCE_OF_HARM, INSUFFICIENT)

# The benchmarks 12.3 requires be kept. A rule beats these or it beats nothing.
REQUIRED_BENCHMARKS = (
    "no_new_defense",
    "fixed_low_exposure",
    "trailing_or_ewma_volatility_target",
    "fixed_long_term_trend_rule",
)


@dataclass(frozen=True)
class EventDefinition:
    """The research event, frozen before anything is measured against it.

    5% over 20 sessions is a research definition and not a risk budget. Changing
    either number starts a new experiment rather than continuing this one, which
    is the rule that stops a threshold drifting toward whatever was significant.
    """

    name: str
    horizon_sessions: int
    drawdown_threshold: float
    frozen_on: str
    note: str = "a research definition, not an investment budget"

    def __post_init__(self) -> None:
        if not 0.0 < self.drawdown_threshold < 1.0:
            raise ValueError(f"{self.name}: threshold must be a fraction between 0 and 1")


PRIMARY_EVENT = EventDefinition(
    name="drawdown_5pct_20_sessions",
    horizon_sessions=20,
    drawdown_threshold=0.05,
    frozen_on="2026-09-09",
)


@dataclass
class Experiment:
    """One pre-registered attempt, with everything that would let it cheat named.

    `family` is the multiple-comparison bookkeeping: a candidate tried alongside
    nineteen others is not the same evidence as one tried alone, and recording
    the family is what makes that visible later.
    """

    name: str
    capability: str
    hypothesis: str
    event: EventDefinition
    benchmarks: list[str]
    frozen_on: str
    family: str
    family_size: int = 1
    primary_metric: str = ""
    sacrifice_budget: str = ""
    grade: str = COMPUTED
    outcome: str = INSUFFICIENT
    evidence: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.capability not in CAPABILITIES:
            raise ValueError(f"{self.name}: unknown capability {self.capability!r}")
        if self.grade not in LADDER:
            raise ValueError(f"{self.name}: unknown grade {self.grade!r}")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"{self.name}: unknown outcome {self.outcome!r}")
        missing = set(REQUIRED_BENCHMARKS) - set(self.benchmarks)
        if missing:
            # Choosing the benchmark after seeing the result is the specific
            # failure this project already made three times.
            raise ValueError(
                f"{self.name}: pre-register every required benchmark; missing {sorted(missing)}"
            )
        if not self.primary_metric:
            raise ValueError(
                f"{self.name}: name one primary metric in advance, or the significant one "
                "gets chosen afterwards from drawdown, Calmar and Sharpe"
            )

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "event": asdict(self.event)}


def promote(experiment: Experiment, to: str, evidence: dict[str, Any]) -> Experiment:
    """Move one rung, with the measurement that earned it. Never two rungs.

    A skipped rung is how "it replays nicely" becomes "it works": the shadow
    period and the forward test are the two that cost time, and they are exactly
    the two a hurry removes.
    """
    if to not in LADDER:
        raise ValueError(f"unknown grade {to!r}")
    here, there = LADDER.index(experiment.grade), LADDER.index(to)
    if there != here + 1:
        raise ValueError(
            f"{experiment.name}: promotion is one rung at a time, "
            f"{experiment.grade} -> {LADDER[here + 1] if here + 1 < len(LADDER) else 'nothing'}"
        )
    for required in ("metric", "value", "benchmark", "window"):
        if required not in evidence:
            raise ValueError(f"{experiment.name}: promotion evidence needs {required!r}")
    if to == FORWARD and not evidence.get("data_after_freeze"):
        # Re-scoring 2000-2026 is research history, however carefully it is
        # sliced. Only dates that did not exist when the rule was frozen are
        # forward evidence.
        raise ValueError(
            f"{experiment.name}: forward evidence needs data that postdates the freeze; "
            "re-scoring history the rule was built on is not a blind test"
        )
    return Experiment(
        **{
            **asdict(experiment),
            "event": experiment.event,
            "grade": to,
            "evidence": [*experiment.evidence, {"grade": to, **evidence}],
        }
    )


def record_outcome(experiment: Experiment, outcome: str, detail: str) -> Experiment:
    """Passed, harmed, or not enough evidence. There is no fourth answer.

    "Not significant" is recorded as insufficient evidence and never as
    equivalence: the two look identical in a p-value and mean opposite things
    when someone decides what to do next.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome {outcome!r}; not-significant is {INSUFFICIENT}")
    return Experiment(
        **{
            **asdict(experiment),
            "event": experiment.event,
            "outcome": outcome,
            "notes": [*experiment.notes, detail],
        }
    )


def may_influence_recommendation(experiment: Experiment) -> bool:
    """Only a forward-tested rule that passed may change what a report suggests.

    Everything below that may be described, with its grade printed. The
    predecessor's whole failure fits in the gap between "describes well" and
    "may influence".
    """
    return experiment.grade == FORWARD and experiment.outcome == PASSED


@dataclass
class Registry:
    """Every experiment ever registered, including the ones that failed.

    Deletion is the quiet version of publication bias, so nothing is removed;
    an abandoned attempt keeps its record and its outcome.
    """

    experiments: list[Experiment] = field(default_factory=list)

    def register(self, experiment: Experiment) -> Experiment:
        if any(e.name == experiment.name for e in self.experiments):
            raise ValueError(f"{experiment.name} is already registered; use a new name")
        self.experiments.append(experiment)
        return experiment

    def by_capability(self, capability: str) -> list[Experiment]:
        return [e for e in self.experiments if e.capability == capability]

    def summary(self) -> dict[str, Any]:
        return {
            "registered": len(self.experiments),
            "by_capability": {
                capability: {
                    "count": len(self.by_capability(capability)),
                    "grades": sorted({e.grade for e in self.by_capability(capability)}),
                    "may_influence": [
                        e.name for e in self.by_capability(capability)
                        if may_influence_recommendation(e)
                    ],
                }
                for capability in CAPABILITIES
            },
            "note": (
                "an experiment that is not forward-tested and passed may be described "
                "with its grade, and may not change what the report suggests"
            ),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"experiments": [e.as_dict() for e in self.experiments]}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Registry":
        path = Path(path)
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        experiments = []
        for item in raw.get("experiments", []):
            payload = dict(item)
            payload["event"] = EventDefinition(**payload["event"])
            experiments.append(Experiment(**payload))
        return cls(experiments)


def open_registry() -> Registry:
    """The registry as it actually stands: the closed findings plus whatever
    replays have been recorded on this machine.

    The closed findings are recorded so the next attempt starts from what is
    known rather than rediscovering it. A replay run (scripts/replay_trend_rule.py)
    persists its record to data/research_registry.json, and it is merged here so
    the report's registry summary reflects experiments actually run.
    """
    registry = Registry()
    closed = [
        (
            "state_ensemble_vs_persistence",
            RISK_IDENTIFICATION,
            "the calibrated state ensemble forecasts 20-session volatility state better "
            "than last-settled-tercile persistence",
            EVIDENCE_OF_HARM,
            "Brier 0.5721 against persistence at 0.5134 on the common day set; the "
            "ensemble is worse, not merely unproven",
        ),
        (
            "state_on_top_of_persistence",
            RISK_IDENTIFICATION,
            "adding the state layer on top of persistence improves the forecast",
            EVIDENCE_OF_HARM,
            "worse by 0.0995 to 0.126 across variants, every comparison significant and "
            "every one the wrong way; refitting on persistence alone cost 0.148",
        ),
        (
            "state_drawdown_edge",
            ACTION_VALUE,
            "state-driven exposure reduces drawdown beyond what the de-leveraging alone "
            "would give",
            INSUFFICIENT,
            "significant against vol_only at every haircut, and not significant at any "
            "against an exposure-matched control; CI [-0.0061, +0.0087] at 0.65",
        ),
        (
            "long_horizon_predictors",
            RISK_IDENTIFICATION,
            "predictors that fade at 20 sessions recover at 126",
            EVIDENCE_OF_HARM,
            "every predictor significant in 2000-2013 and none in 2013-2026, a window "
            "containing 2015, 2018, 2020 and 2022",
        ),
    ]
    for name, capability, hypothesis, outcome, detail in closed:
        experiment = Experiment(
            name=name,
            capability=capability,
            hypothesis=hypothesis,
            event=PRIMARY_EVENT,
            benchmarks=list(REQUIRED_BENCHMARKS),
            frozen_on="2026-09-09",
            family="closed-findings",
            family_size=len(closed),
            primary_metric="forward Brier against the strongest benchmark",
            sacrifice_budget="not applicable; none reached a stage where a budget mattered",
        )
        registry.register(record_outcome(experiment, outcome, detail))

    from market_state_lab.config import PROJECT_ROOT

    persisted = PROJECT_ROOT / "data" / "research_registry.json"
    if persisted.exists():
        for experiment in Registry.load(persisted).experiments:
            if not any(e.name == experiment.name for e in registry.experiments):
                registry.register(experiment)
    return registry


__all__ = [
    "CAPABILITIES",
    "FORWARD",
    "LADDER",
    "OUTCOMES",
    "PRIMARY_EVENT",
    "REQUIRED_BENCHMARKS",
    "EventDefinition",
    "Experiment",
    "Registry",
    "may_influence_recommendation",
    "open_registry",
    "promote",
    "record_outcome",
]

"""Section 12. The gate is code because the predecessor failed here specifically:
three benchmarks chosen in turn, an interpretable rule called an effective one,
and a drawdown edge that was de-leveraging."""

from __future__ import annotations

import pytest

from market_state_lab.research import (
    ACTION_VALUE,
    COMPUTED,
    EVIDENCE_OF_HARM,
    FORWARD,
    INSUFFICIENT,
    LADDER,
    PASSED,
    PRIMARY_EVENT,
    REPLAYED,
    REQUIRED_BENCHMARKS,
    RISK_IDENTIFICATION,
    SHADOW,
    EventDefinition,
    Experiment,
    Registry,
    may_influence_recommendation,
    open_registry,
    promote,
    record_outcome,
)


def _experiment(**over) -> Experiment:
    fields = {
        "name": "candidate-a",
        "capability": RISK_IDENTIFICATION,
        "hypothesis": "the rule anticipates a 5% drawdown better than persistence",
        "event": PRIMARY_EVENT,
        "benchmarks": list(REQUIRED_BENCHMARKS),
        "frozen_on": "2026-09-09",
        "family": "first-version",
        "primary_metric": "forward Brier against the strongest benchmark",
    }
    fields.update(over)
    return Experiment(**fields)


def _evidence(**over) -> dict:
    payload = {
        "metric": "forward Brier",
        "value": 0.49,
        "benchmark": "persistence",
        "window": "2024-2026",
    }
    payload.update(over)
    return payload


# ---------------------------------------------------------------------------
# Pre-registration
# ---------------------------------------------------------------------------


def test_every_required_benchmark_must_be_named_in_advance() -> None:
    # Picking the benchmark after seeing the result is the failure this project
    # already made three times.
    with pytest.raises(ValueError, match="pre-register every required benchmark"):
        _experiment(benchmarks=["no_new_defense"])


def test_one_primary_metric_must_be_named_before_the_result_exists() -> None:
    with pytest.raises(ValueError, match="the significant one gets chosen afterwards"):
        _experiment(primary_metric="")


def test_the_event_is_a_research_definition_and_not_a_budget() -> None:
    assert PRIMARY_EVENT.drawdown_threshold == 0.05
    assert PRIMARY_EVENT.horizon_sessions == 20
    assert "not an investment budget" in PRIMARY_EVENT.note
    with pytest.raises(ValueError, match="fraction between 0 and 1"):
        EventDefinition("bad", 20, 5.0, "2026-09-09")


def test_the_family_size_is_recorded_so_multiplicity_stays_visible() -> None:
    alone = _experiment(family_size=1)
    among = _experiment(name="candidate-b", family_size=20)
    # A candidate tried alongside nineteen others is not the same evidence.
    assert alone.family_size != among.family_size


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def test_promotion_climbs_one_rung_at_a_time() -> None:
    experiment = _experiment()
    assert experiment.grade == COMPUTED
    replayed = promote(experiment, REPLAYED, _evidence())
    assert replayed.grade == REPLAYED
    # The two rungs that cost time are the two a hurry removes.
    with pytest.raises(ValueError, match="one rung at a time"):
        promote(replayed, FORWARD, _evidence(data_after_freeze=True))


def test_promotion_needs_the_measurement_that_earned_it() -> None:
    with pytest.raises(ValueError, match="needs 'benchmark'"):
        promote(_experiment(), REPLAYED, {"metric": "Brier", "value": 0.49, "window": "x"})


def test_forward_evidence_must_postdate_the_freeze() -> None:
    shadow = promote(promote(_experiment(), REPLAYED, _evidence()), SHADOW, _evidence())
    # Re-scoring 2000-2026 is research history however carefully it is sliced.
    with pytest.raises(ValueError, match="is not a blind test"):
        promote(shadow, FORWARD, _evidence(window="2000-2026"))
    forward = promote(shadow, FORWARD, _evidence(window="2026-09 onward", data_after_freeze=True))
    assert forward.grade == FORWARD


def test_the_evidence_trail_keeps_every_rung() -> None:
    forward = promote(
        promote(promote(_experiment(), REPLAYED, _evidence()), SHADOW, _evidence()),
        FORWARD,
        _evidence(data_after_freeze=True),
    )
    assert [e["grade"] for e in forward.evidence] == [REPLAYED, SHADOW, FORWARD]


# ---------------------------------------------------------------------------
# Outcomes, and what may influence a report
# ---------------------------------------------------------------------------


def test_not_significant_is_insufficient_evidence_and_never_equivalence() -> None:
    # The two look identical in a p-value and mean opposite things when someone
    # decides what to do next.
    assert INSUFFICIENT in ("insufficient_evidence",)
    with pytest.raises(ValueError, match="not-significant is insufficient_evidence"):
        record_outcome(_experiment(), "equivalent", "no significant difference")


def test_only_a_forward_tested_pass_may_change_what_a_report_suggests() -> None:
    plausible = record_outcome(_experiment(), PASSED, "replays beautifully")
    # The predecessor's whole failure fits in the gap between "describes well"
    # and "may influence".
    assert not may_influence_recommendation(plausible)

    earned = promote(
        promote(promote(_experiment(name="c"), REPLAYED, _evidence()), SHADOW, _evidence()),
        FORWARD,
        _evidence(data_after_freeze=True),
    )
    assert may_influence_recommendation(record_outcome(earned, PASSED, "held up"))
    assert not may_influence_recommendation(record_outcome(earned, INSUFFICIENT, "flat"))


# ---------------------------------------------------------------------------
# The registry as it stands
# ---------------------------------------------------------------------------


def test_the_closed_findings_are_registered_rather_than_forgotten() -> None:
    registry = open_registry()
    names = {e.name for e in registry.experiments}
    assert {"state_ensemble_vs_persistence", "state_drawdown_edge"} <= names
    ensemble = next(e for e in registry.experiments if e.name == "state_ensemble_vs_persistence")
    assert ensemble.outcome == EVIDENCE_OF_HARM
    assert "0.5721" in ensemble.notes[0]


def test_the_drawdown_edge_is_insufficient_rather_than_harmful() -> None:
    # Significant against vol_only, not significant against an exposure-matched
    # control. That is an absence of evidence, not evidence of harm.
    edge = next(e for e in open_registry().experiments if e.name == "state_drawdown_edge")
    assert edge.outcome == INSUFFICIENT
    assert edge.capability == ACTION_VALUE


def test_nothing_in_the_registry_may_influence_a_recommendation() -> None:
    registry = open_registry()
    assert all(e.grade == COMPUTED for e in registry.experiments)
    assert not [e for e in registry.experiments if may_influence_recommendation(e)]
    summary = registry.summary()
    assert all(not c["may_influence"] for c in summary["by_capability"].values())


def test_capabilities_are_graded_apart_because_they_fail_apart() -> None:
    summary = open_registry().summary()
    # A data layer that catches contamination says nothing about whether a rule
    # sees risk coming.
    assert set(summary["by_capability"]) == {"data_quality", "risk_identification", "action_value"}
    assert summary["by_capability"]["data_quality"]["count"] == 0


def test_a_failed_experiment_is_never_deleted(tmp_path) -> None:
    registry = open_registry()
    before = len(registry.experiments)
    registry.save(tmp_path / "registry.json")
    # Deletion is the quiet version of publication bias.
    assert len(Registry.load(tmp_path / "registry.json").experiments) == before
    assert not hasattr(registry, "delete")


def test_a_name_cannot_be_reused_for_a_second_attempt() -> None:
    registry = Registry()
    registry.register(_experiment())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_experiment())


def test_the_ladder_order_is_the_promotion_order() -> None:
    assert LADDER == (
        "computed_correctly", "replayed_and_explainable",
        "frozen_shadow_output", "tested_on_new_data",
    )

"""The description layer. A percentile is a rank, the ranks are causal, the
dimensions stay apart, and the leaning is pre-registered as insufficient."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_state_lab.market_assessment import (
    DATA_INSUFFICIENT,
    DEFAULT_RULES,
    REPLAYED,
    STRESS_SPREADING,
    TREND_DAMAGED,
    UNTESTED,
    VALIDATED,
    Assessment,
    RuleSet,
    assess,
    promote,
    report,
    state_labels,
)
from market_state_lab.market_evidence import (
    BOUNDARIES,
    CREDIT,
    DIMENSIONS,
    PARTICIPATION,
    TREND,
    VOLATILITY,
    Indicator,
    MarketEvidence,
    build_evidence,
    causal_percentile,
    contradictions,
    describe,
    review_triggers,
)

SESSIONS = 900


def _prices(drift: float = 0.0003, vol: float = 0.008, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-03", periods=SESSIONS)
    out = {}
    for name in ("spy", "rsp", "iwm", "hyg", "lqd"):
        steps = rng.normal(drift, vol, SESSIONS)
        out[name] = 100.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame(out, index=index)


def _macro() -> pd.DataFrame:
    index = pd.bdate_range("2022-01-03", periods=SESSIONS)
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {"hy_oas": 3.0 + np.cumsum(rng.normal(0, 0.01, SESSIONS)),
         "baa_spread": 1.8 + np.cumsum(rng.normal(0, 0.005, SESSIONS))},
        index=index,
    )


def _vix() -> pd.DataFrame:
    index = pd.bdate_range("2022-01-03", periods=SESSIONS)
    rng = np.random.default_rng(13)
    return pd.DataFrame({"vix_close": 16.0 + np.cumsum(rng.normal(0, 0.15, SESSIONS))}, index=index)


# ---------------------------------------------------------------------------
# The percentile: causal, and not a probability
# ---------------------------------------------------------------------------


def test_a_percentile_uses_only_what_preceded_it() -> None:
    # Against the full sample, today is ranked partly by days that have not
    # happened. That lookahead is invisible in the output and flatters whatever
    # comes next.
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    ranks = causal_percentile(series, min_history=2)
    assert ranks.iloc[1] == pytest.approx(1.0)  # 2 beat the one prior value
    assert ranks.iloc[4] == pytest.approx(1.0)  # 5 beat all four prior
    # A later collapse cannot change an earlier rank.
    extended = pd.concat([series, pd.Series([-99.0])], ignore_index=True)
    assert causal_percentile(extended, min_history=2).iloc[4] == ranks.iloc[4]


def test_a_short_history_yields_no_rank_rather_than_a_confident_one() -> None:
    ranks = causal_percentile(pd.Series([1.0, 2.0, 3.0]), min_history=252)
    assert ranks.dropna().empty


def test_every_indicator_carries_what_its_percentile_is_not() -> None:
    evidence = build_evidence(_prices(), _vix(), _macro())
    ranked = [i for i in evidence.indicators if i.percentile is not None]
    assert ranked
    for indicator in ranked:
        # A bare 0.93 becomes a probability in a reader's head within a paragraph.
        assert "not a probability" in indicator.percentile_meaning


def test_no_output_field_could_be_read_as_a_forecast() -> None:
    described = describe(build_evidence(_prices(), _vix(), _macro()))
    flat = str(described).lower()
    disclaimer = described["not_a_forecast"].lower()
    assert "none of it is a probability" in disclaimer
    assert "nothing here says what happens next" in disclaimer
    for banned in ("probability_of", "expected_return", "forecast_", "risk_score", "composite"):
        assert banned not in flat


# ---------------------------------------------------------------------------
# The five dimensions stay apart
# ---------------------------------------------------------------------------


def test_all_five_dimensions_are_populated_when_the_data_is_there() -> None:
    evidence = build_evidence(_prices(), _vix(), _macro())
    assert set(evidence.covered_dimensions) == set(DIMENSIONS)
    assert evidence.missing_dimensions == ()


def test_an_absent_dimension_is_absent_and_not_imputed() -> None:
    # "Credit is unknown" and "credit is calm" are different statements, and the
    # report has to be able to make the first one.
    evidence = build_evidence(_prices()[["spy", "rsp", "iwm"]], _vix(), macro=None)
    assert CREDIT in evidence.missing_dimensions
    assert not [i for i in evidence.by_dimension(CREDIT) if i.value is not None]
    assert CREDIT not in describe(evidence)["boundaries"]


def test_each_dimension_states_what_it_may_not_be_read_as() -> None:
    described = describe(build_evidence(_prices(), _vix(), _macro()))
    for dimension, text in described["boundaries"].items():
        assert text == BOUNDARIES[dimension]
    assert "not a count of advancing shares" in BOUNDARIES["participation"]
    assert "not the price of any actual contract" in BOUNDARIES["implied_risk"]


def test_level_and_acceleration_are_separate_indicators() -> None:
    evidence = build_evidence(_prices(), _vix(), _macro())
    names = {i.name for i in evidence.by_dimension(VOLATILITY)}
    # A market can be quiet and speeding up, or loud and settling.
    assert {"realised_vol_20d", "vol_ratio_20_over_60"} <= names


def test_the_bond_etf_reading_is_marked_as_overlapping_not_independent() -> None:
    evidence = build_evidence(_prices(), _vix(), _macro())
    overlapping = next(i for i in evidence.by_dimension(CREDIT) if i.name == "hyg_over_lqd")
    assert "corroboration, not an independent vote" in overlapping.overlaps
    # The spread series carry no such warning: they are independent of equities.
    assert all(i.overlaps is None for i in evidence.by_dimension(CREDIT) if "oas" in i.name)


def test_there_is_no_composite_score_to_sort_by() -> None:
    import market_state_lab.market_evidence as module

    # The predecessor collapsed these into one number, and the number then hid
    # which dimension carried it and which contradicted it.
    assert not {"risk_score", "composite", "total_score"} & set(dir(module))
    assert "score" not in describe(build_evidence(_prices(), _vix(), _macro()))


# ---------------------------------------------------------------------------
# Disagreement is the finding, not a problem to average away
# ---------------------------------------------------------------------------


def _indicator(dimension: str, name: str, change_20: float, riskier: bool = True) -> Indicator:
    return Indicator(dimension, name, 1.0, "unit", change_20=change_20,
                     percentile=0.5, higher_is_riskier=riskier)


def test_dimensions_pointing_opposite_ways_are_reported_as_disagreeing() -> None:
    evidence = MarketEvidence(
        pd.Timestamp("2026-09-08"),
        [_indicator(VOLATILITY, "vol", +5.0), _indicator(CREDIT, "hy_oas", -0.4)],
    )
    found = contradictions(evidence)
    assert found[0]["kind"] == "dimensions_disagree"
    assert found[0]["deteriorating"] == VOLATILITY
    assert found[0]["improving"] == CREDIT
    assert "neither reading is discounted" in found[0]["detail"]


def test_a_dimension_at_odds_with_itself_has_no_single_reading() -> None:
    evidence = MarketEvidence(
        pd.Timestamp("2026-09-08"),
        [_indicator(TREND, "a", +1.0), _indicator(TREND, "b", -1.0)],
    )
    assert contradictions(evidence)[0]["kind"] == "dimension_internally_split"


def test_an_overlapping_indicator_does_not_get_a_vote_in_contradictions() -> None:
    overlapping = Indicator(CREDIT, "hyg_over_lqd", 1.0, "%", change_20=-5.0,
                            percentile=0.5, overlaps="same equity move again")
    evidence = MarketEvidence(
        pd.Timestamp("2026-09-08"), [_indicator(VOLATILITY, "vol", +5.0), overlapping]
    )
    assert contradictions(evidence) == []


def test_review_triggers_say_what_would_make_the_reading_stale() -> None:
    triggers = review_triggers(build_evidence(_prices(), _vix(), _macro()))
    assert triggers
    for trigger in triggers:
        assert 0.0 <= trigger["review_if_below"] <= trigger["review_if_above"] <= 1.0
        assert "would make this description stale" in trigger["detail"]


# ---------------------------------------------------------------------------
# The judgment layer: a real seam, deliberately unpopulated
# ---------------------------------------------------------------------------


def test_the_leaning_is_pre_registered_as_insufficient_and_that_is_the_expected_result() -> None:
    assessment = assess(build_evidence(_prices(), _vix(), _macro()))
    assert assessment.action_leaning == DATA_INSUFFICIENT
    assert "Pre-registered" in assessment.leaning_reason
    assert "expected answer, not a shortfall" in assessment.leaning_reason
    assert assessment.research_grade == UNTESTED


def test_an_untested_rule_may_describe_but_never_lean() -> None:
    # The single thing that killed the predecessor: an interpretable rule
    # presented as a working one.
    with pytest.raises(ValueError, match="an untested rule may describe, never lean"):
        Assessment(
            as_of=None, snapshot_id=None, observation_horizon="20 sessions",
            action_leaning="REVIEW_HEDGE", research_grade=UNTESTED,
        )


def test_a_promoted_rule_is_allowed_to_lean() -> None:
    # The seam is real: passing the gate changes what the contract permits.
    promoted = promote(DEFAULT_RULES, REPLAYED, "beat persistence by 0.03 Brier out of sample")
    assert promoted.grade == REPLAYED
    allowed = Assessment(
        as_of=None, snapshot_id=None, observation_horizon="20 sessions",
        action_leaning="WATCH", research_grade=promoted.grade,
    )
    assert allowed.action_leaning == "WATCH"


def test_a_promotion_needs_the_measurement_that_earned_it() -> None:
    with pytest.raises(ValueError, match="needs the measurement"):
        promote(DEFAULT_RULES, VALIDATED, "   ")
    with pytest.raises(ValueError, match="must be to a measured grade"):
        promote(DEFAULT_RULES, UNTESTED, "it has been running a while")


def test_state_labels_coexist_rather_than_picking_one_regime() -> None:
    evidence = MarketEvidence(
        pd.Timestamp("2026-09-08"),
        [
            Indicator(TREND, "spy_vs_200d_ma", -4.0, "%", change_20=-3.0,
                      percentile=0.05, higher_is_riskier=False),
            Indicator(VOLATILITY, "vol_ratio_20_over_60", 1.4, "ratio", change_20=+0.3,
                      percentile=0.95),
            Indicator(CREDIT, "hy_oas", 6.5, "pp", change_20=+1.2, percentile=0.96),
        ],
    )
    labels = state_labels(evidence)
    # A market can be damaged and accelerating at once; one regime would hide one.
    assert TREND_DAMAGED in labels and "volatility_elevated" in labels
    assert STRESS_SPREADING in labels


def test_a_label_needs_a_level_and_not_only_a_sign() -> None:
    # The real reading that exposed this: SPY 7.8% above its 200-day average,
    # credit spreads in the calmest 7% of their history, VIX at 15.7 - and three
    # 20-session changes with an adverse sign. Calling that spreading stress is
    # how a label teaches its reader to ignore it.
    calm = MarketEvidence(
        pd.Timestamp("2026-09-08"),
        [
            Indicator(TREND, "spy_vs_200d_ma", 7.8, "%", change_20=-2.5,
                      percentile=0.70, higher_is_riskier=False),
            Indicator(PARTICIPATION, "rsp_over_spy", 0.9, "%", change_20=-2.2,
                      percentile=0.67, higher_is_riskier=False),
            Indicator(CREDIT, "hy_oas", 2.68, "pp", change_20=-0.02, percentile=0.07),
        ],
    )
    assert state_labels(calm) == []


def test_stress_spreads_only_when_participation_or_credit_joins_in() -> None:
    # The plan names the dimensions: two others moving together is not spreading.
    without = MarketEvidence(
        pd.Timestamp("2026-09-08"),
        [
            Indicator(TREND, "spy_vs_200d_ma", -6.0, "%", change_20=-4.0,
                      percentile=0.02, higher_is_riskier=False),
            Indicator(VOLATILITY, "realised_vol_20d", 38.0, "%", change_20=+14.0,
                      percentile=0.97),
        ],
    )
    assert STRESS_SPREADING not in state_labels(without)


def test_the_thresholds_are_frozen_with_a_date() -> None:
    assert DEFAULT_RULES.frozen_at == "2026-09-09"
    assert DEFAULT_RULES.grade == UNTESTED
    with pytest.raises(ValueError, match="unknown research grade"):
        RuleSet(name="x", frozen_at="2026-09-09", grade="looks right")


def test_the_assessment_carries_the_evidence_against_itself() -> None:
    assessment = assess(build_evidence(_prices(), _vix(), _macro()))
    assert assessment.applicable_conditions
    assert any("different preferences" in c for c in assessment.applicable_conditions)
    assert any("reads a position" in c for c in assessment.applicable_conditions)
    assert assessment.triggers and assessment.next_review_reason


def test_the_report_puts_the_description_before_the_leaning() -> None:
    evidence = build_evidence(_prices(), _vix(), _macro())
    out = report(evidence, assess(evidence))
    assert out["reading_order"][0].startswith("what changed")
    assert "pre-registered" in out["reading_order"][1]

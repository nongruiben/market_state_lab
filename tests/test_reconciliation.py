"""Section 6.4. A second source answers in one direction only: it can confirm
an event, and it can never clear a quarantine, because two feeds carrying one
corrupt file agree perfectly."""

from __future__ import annotations

import pandas as pd
import pytest

from market_state_lab.data.reconciliation import (
    SourceSeries,
    agreement_cannot_clear,
    confirm_move,
    measured_tolerance,
    reconcile_closes,
    stability_only,
    substitution_record,
    summarise,
    unverifiable,
)
from market_state_lab.data.validation import QUARANTINE, REVIEW, QualityIssue, validate_daily_bars


def _series(values: list[float], start: str = "2026-09-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"))


def _pair(left: list[float], right: list[float], **over):
    primary = SourceSeries("TWS", _series(left), **over)
    secondary = SourceSeries("Yahoo", _series(right))
    return primary, secondary


def _codes(issues) -> list[str]:
    return [i.code for i in issues]


# ---------------------------------------------------------------------------
# Agreement, disagreement, and where the band comes from
# ---------------------------------------------------------------------------


def test_two_sources_that_match_agree() -> None:
    # The real pair: TWS 770.19 / 765.96 against Yahoo 770.190002 / 765.960022.
    primary, secondary = _pair([770.19, 765.96], [770.190002, 765.960022])
    result = reconcile_closes("SPY", primary, secondary)
    assert result.agreed
    assert not [i for i in result.issues if i.severity != "note"]


def test_the_band_is_measured_from_the_sources_own_disagreement() -> None:
    quiet = measured_tolerance(_series([100.0] * 30), _series([100.0001] * 30))
    noisy = measured_tolerance(
        _series([100.0 + i * 0.01 for i in range(30)]),
        _series([(100.0 + i * 0.01) * 1.004 for i in range(30)]),
    )
    # A fixed threshold would flag the quiet pair or wave the noisy one through.
    assert noisy > quiet
    assert noisy == pytest.approx(0.004, abs=1e-4)
    # And a pair that has always agreed does not get a tolerance of zero.
    assert quiet > 0


def test_one_outlier_cannot_set_the_band_that_judges_it() -> None:
    # A quantile let a spurious 34% move raise the tolerance to 68% and then
    # pass itself. Neither the median nor the MAD can be moved by one point.
    steady = _series([100.0] * 10 + [66.0])
    other = _series([100.0] * 11)
    assert measured_tolerance(steady, other) < 0.01


def test_a_difference_outside_the_measured_band_is_reviewed_with_the_escalation() -> None:
    primary, secondary = _pair([100.0] * 10 + [130.0], [100.0] * 10 + [100.0])
    result = reconcile_closes("SPY", primary, secondary)
    disagreement = next(i for i in result.issues if i.code == "sources_disagree")
    assert disagreement.severity == REVIEW
    assert "check timing and convention" in disagreement.detail
    # Reviewed, never resolved by picking a winner.
    assert disagreement.blocks == ()


def test_a_convention_mismatch_stops_the_comparison_before_the_numbers() -> None:
    primary, secondary = _pair([770.0], [700.0], adjusted=True)
    result = reconcile_closes("SPY", primary, secondary)
    mismatch = next(i for i in result.issues if i.code == "convention_mismatch")
    assert mismatch.severity == QUARANTINE
    # Comparing a live price to an adjusted close gives a real, meaningless number.
    assert result.comparison.empty


def test_dates_present_in_only_one_source_are_noted_not_treated_as_gaps() -> None:
    primary = SourceSeries("TWS", _series([100.0, 101.0, 102.0]))
    secondary = SourceSeries("Yahoo", _series([100.0, 101.0]))
    result = reconcile_closes("SPY", primary, secondary)
    assert "date_only_in_primary" in _codes(result.issues)


def test_no_overlap_means_neither_confirms_the_other() -> None:
    primary = SourceSeries("TWS", _series([100.0, 101.0], start="2026-09-01"))
    secondary = SourceSeries("Yahoo", _series([100.0, 101.0], start="2026-10-01"))
    result = reconcile_closes("SPY", primary, secondary)
    assert "no_overlap" in _codes(result.issues)
    assert not result.agreed


# ---------------------------------------------------------------------------
# Row 2: a real crash confirmed by a second source is kept
# ---------------------------------------------------------------------------


def test_a_crash_both_sources_report_is_an_event_and_may_not_be_filtered() -> None:
    closes = [770.0] * 5 + [508.0]
    primary, secondary = _pair(closes, [c * 1.00001 for c in closes])
    result = reconcile_closes("SPY", primary, secondary)
    verdict = confirm_move(result, primary.values.index[-1], primary, secondary)
    assert verdict.code == "move_confirmed"
    assert "nothing downstream may filter it away" in verdict.detail
    # And the single-source rule had only reviewed it, never quarantined it.
    frame = pd.DataFrame({"close": primary.values})
    assert not any(i.severity == QUARANTINE for i in validate_daily_bars(frame, "SPY"))


def test_a_move_only_one_source_reports_escalates_rather_than_losing() -> None:
    primary, secondary = _pair([770.0] * 5 + [508.0], [770.0] * 6)
    result = reconcile_closes("SPY", primary, secondary)
    verdict = confirm_move(result, primary.values.index[-1], primary, secondary)
    assert verdict.code == "move_unconfirmed"
    assert "neither is automatically the loser" in verdict.detail


def test_a_move_with_no_prior_observation_cannot_be_confirmed() -> None:
    primary, secondary = _pair([770.0], [770.0])
    verdict = confirm_move(
        reconcile_closes("SPY", primary, secondary), primary.values.index[0], primary, secondary
    )
    assert verdict.code == "move_unconfirmable"


# ---------------------------------------------------------------------------
# Row 9: a shared anomaly is not a pass
# ---------------------------------------------------------------------------


def test_perfect_agreement_does_not_clear_a_quarantine() -> None:
    # Both feeds carry the same hundredfold price. They agree completely, and
    # that is exactly the case where agreement means nothing.
    closes = [770.0, 77000.0]
    primary, secondary = _pair(closes, closes)
    result = reconcile_closes("SPY", primary, secondary)
    assert result.agreed
    single = validate_daily_bars(pd.DataFrame({"close": primary.values}), "SPY")
    assert any(i.severity == QUARANTINE for i in single)

    after = agreement_cannot_clear(result, single)
    still = [i for i in after if i.severity == QUARANTINE]
    assert len(still) == len([i for i in single if i.severity == QUARANTINE])
    assert "a majority is not evidence" in next(
        i for i in after if i.code == "agreement_does_not_clear_quarantine"
    ).detail


def test_with_nothing_quarantined_agreement_adds_no_noise() -> None:
    primary, secondary = _pair([770.0, 771.0], [770.0, 771.0])
    result = reconcile_closes("SPY", primary, secondary)
    clean = [QualityIssue("x", REVIEW, "SPY", "detail", ())]
    assert agreement_cannot_clear(result, clean) == clean


# ---------------------------------------------------------------------------
# What a second source cannot be faked into being
# ---------------------------------------------------------------------------


def test_an_instrument_with_no_second_source_says_so() -> None:
    issue = unverifiable("SPY 20261120 730P", "option quotes arrive from TWS alone")
    assert issue.code == "no_independent_source"
    # The absence of a check has to be in the record, or a reader assumes it passed.
    assert issue.blocks == ()


def test_repeated_reads_of_one_feed_are_stability_and_not_corroboration() -> None:
    issue = stability_only("SPY 20261120 730P", 3, 0.0001)
    assert "corroborates nothing" in issue.detail
    assert "a source cannot confirm itself" in issue.detail


def test_there_is_no_function_that_merges_two_sources() -> None:
    import market_state_lab.data.reconciliation as module

    # Averaging two conflicting prices invents a third nobody reported, and
    # picking the friendlier source is how a data layer starts serving a model.
    forbidden = {"merge", "blend", "average", "combine", "prefer", "best_source", "resolve"}
    assert not forbidden & set(dir(module))


def test_a_substitution_needs_a_recorded_reason_and_is_not_automatic() -> None:
    record = substitution_record("SPY", "TWS", "Yahoo", "TWS history entitlement missing")
    assert record["substitution_reason"]
    assert record["must_pass_same_gates"] is True
    with pytest.raises(ValueError, match="without a recorded reason"):
        substitution_record("SPY", "TWS", "Yahoo", "   ")


def test_the_summary_reports_what_was_compared_not_who_was_right() -> None:
    primary, secondary = _pair([100.0] * 10 + [130.0], [100.0] * 11)
    summary = summarise(reconcile_closes("SPY", primary, secondary))
    assert summary["compared_days"] == 11
    assert summary["days_outside_tolerance"] == 1
    assert summary["agreed"] is False
    # No field names a winner or a true price.
    assert "true_price" not in summary and "preferred_source" not in summary

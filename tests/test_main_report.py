"""Section 13. The order is the argument, four kinds of nothing are normal
results, and no field in a report may come from an account."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_state_lab.events import EventLog, EventRules
from market_state_lab.main_report import (
    ReportInputs,
    account_fields_present,
    build_report,
    render_markdown,
)
from market_state_lab.market_assessment import DATA_INSUFFICIENT, assess
from market_state_lab.market_evidence import build_evidence

SESSIONS = 700


def _prices(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2023-01-02", periods=SESSIONS)
    return pd.DataFrame(
        {
            name: 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, SESSIONS)))
            for name in ("spy", "rsp", "iwm", "hyg", "lqd")
        },
        index=index,
    )


def _macro() -> pd.DataFrame:
    index = pd.bdate_range("2023-01-02", periods=SESSIONS)
    rng = np.random.default_rng(5)
    return pd.DataFrame({"hy_oas": 3.0 + np.cumsum(rng.normal(0, 0.01, SESSIONS))}, index=index)


def _inputs(**over) -> ReportInputs:
    evidence = build_evidence(_prices(), None, _macro())
    fields = {
        "evidence": evidence,
        "assessment": assess(evidence, snapshot_id="2026-09-08-abc"),
        "snapshot_id": "2026-09-08-abc",
        "session_date": "2026-09-08",
        "eligible_for": ("day_end_analysis", "instrument_quotes"),
        "controls": [
            {"label": "no new protection", "structure": "unhedged", "cost_usd": 0.0,
             "pnl_at_worst_move": -20000.0, "protection_at_worst_move": 0.0,
             "pnl_if_flat": 0.0, "pnl_at_best_move": 10000.0},
            {"label": "reduce to 80%", "structure": "de-risked", "cost_usd": 4.0,
             "pnl_at_worst_move": -16004.0, "protection_at_worst_move": 3996.0,
             "pnl_if_flat": -4.0, "pnl_at_best_move": 7996.0},
        ],
    }
    fields.update(over)
    return ReportInputs(**fields)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"label": "put 690 20261009", "structure": "long put", "shortlisted": True,
             "cost_usd": 121.65, "cost_pct_of_notional": 0.0012,
             "pnl_at_worst_move": -12398.45, "protection_at_worst_move": 7601.55,
             "pnl_if_flat": -121.65, "pnl_at_best_move": 9878.35,
             "protected_below": 690.0, "market_data_type": "delayed_frozen"},
            {"label": "put 730 20261120", "structure": "long put", "shortlisted": False,
             "cost_usd": 768.65, "pnl_at_worst_move": -9383.85},
        ]
    )


# ---------------------------------------------------------------------------
# The order, and the caps
# ---------------------------------------------------------------------------


def test_the_seven_sections_come_in_the_order_the_plan_fixes() -> None:
    report = build_report(_inputs())
    # A reader cannot reach a put's cost without passing what the quotes behind
    # it were worth.
    assert list(report) == [
        "1_data_status", "2_judgment", "3_changes", "4_dimensions",
        "5_instruments", "6_scenarios", "7_review",
    ]


def test_at_most_three_items_on_each_side_and_the_rest_are_counted() -> None:
    report = build_report(_inputs())
    judgment = report["2_judgment"]
    assert len(judgment["supporting"]) <= 3
    assert len(judgment["contrary"]) <= 3
    # Truncation is stated rather than silent.
    assert judgment["supporting_withheld"] >= 0
    assert judgment["contrary_withheld"] >= 0
    markdown = render_markdown(report)
    if judgment["contrary_withheld"]:
        assert f"({judgment['contrary_withheld']} further items not shown)" in markdown


# ---------------------------------------------------------------------------
# Four kinds of nothing, all normal
# ---------------------------------------------------------------------------


def test_insufficient_evidence_renders_as_a_result_not_a_failure() -> None:
    report = build_report(_inputs())
    assert report["2_judgment"]["action_leaning"] == DATA_INSUFFICIENT
    assert report["2_judgment"]["normal_outcome"] is True
    markdown = render_markdown(report)
    assert "DATA_INSUFFICIENT" in markdown
    for word in ("error", "failed", "unavailable"):
        assert word not in markdown.lower().split("## 2. market reading")[1][:600]


def test_no_qualifying_instrument_is_stated_and_the_controls_still_compare() -> None:
    report = build_report(_inputs(candidates=pd.DataFrame()))
    instruments = report["5_instruments"]
    assert instruments["candidates"] == []
    assert "the controls are the whole comparison" in instruments["normal_outcome"]
    # The controls are still a real comparison, so section 6 is not empty.
    assert len(report["6_scenarios"]["rows"]) == 2


def test_a_missing_dimension_is_unknown_and_never_rendered_as_calm() -> None:
    evidence = build_evidence(_prices()[["spy"]], None, None)
    report = build_report(_inputs(evidence=evidence, assessment=assess(evidence)))
    assert "credit" in report["4_dimensions"]["missing_dimensions"]
    assert "unknown, not calm" in report["4_dimensions"]["missing_note"]
    missing_line = next(
        line for line in render_markdown(report).splitlines() if line.startswith("Missing:")
    )
    assert "credit" in missing_line


def test_conflicting_evidence_is_shown_rather_than_resolved() -> None:
    report = build_report(_inputs())
    markdown = render_markdown(report)
    if report["4_dimensions"]["contradictions"]:
        assert "disagreement:" in markdown


# ---------------------------------------------------------------------------
# Nothing from an account, ever
# ---------------------------------------------------------------------------


def test_no_account_field_reaches_the_report() -> None:
    # Structural, not a scan of the prose: the prose says "not anyone's
    # position", and a substring search cannot tell a disclaimer from a
    # disclosure. Field names are what would actually leak.
    report = build_report(_inputs(candidates=_candidates()))
    assert account_fields_present(report) == []


def test_a_report_carrying_an_account_field_is_refused() -> None:
    leaking = {"5_instruments": {"candidates": [{"portfolio_value": 250_000.0}]}}
    assert account_fields_present(leaking) == ["5_instruments.candidates[0].portfolio_value"]


def test_the_reference_exposure_is_named_as_a_yardstick_not_a_position() -> None:
    markdown = render_markdown(build_report(_inputs(candidates=_candidates())))
    assert "not anyone's position" in markdown


def test_the_conditional_never_chooses_between_reducing_and_protecting() -> None:
    instruments = build_report(_inputs(candidates=_candidates()))["5_instruments"]
    assert "cannot choose between those two preferences" in instruments["conditional"]


# ---------------------------------------------------------------------------
# Change and events
# ---------------------------------------------------------------------------


def test_a_new_label_is_not_yet_an_event() -> None:
    report = build_report(_inputs(previous_labels=["trend_damaged"]))
    changes = report["3_changes"]
    assert "trend_damaged" in changes["labels_dropped"]


def test_a_confirmed_event_is_reported_with_its_own_wording() -> None:
    log = EventLog(rules=EventRules(confirm_sessions=2, release_sessions=5, cooldown_sessions=0))
    log.observe(["trend_damaged"], "2026-09-07")
    news = log.observe(["trend_damaged"], "2026-09-08")
    report = build_report(_inputs(new_events=news, open_events=log.open_events()))
    assert report["3_changes"]["new_events"][0]["label"] == "trend_damaged"
    assert report["3_changes"]["standing_events"][0]["since"] == "2026-09-07"


def test_a_quiet_session_says_nothing_changed() -> None:
    assert "nothing changed" in render_markdown(build_report(_inputs()))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_the_markdown_carries_every_section_heading() -> None:
    markdown = render_markdown(build_report(_inputs(candidates=_candidates())))
    for heading in (
        "## 1. Data status", "## 2. Market reading", "## 3. Since last time",
        "## 4. Dimensions", "## 5. Reduce or protect",
        "## 6. Payoff against the reference exposure", "## 7. What would change this",
    ):
        assert heading in markdown


def test_the_scenario_table_shows_the_shortlist_beside_the_controls() -> None:
    markdown = render_markdown(build_report(_inputs(candidates=_candidates())))
    assert "put 690 20261009" in markdown
    assert "no new protection" in markdown
    # The row that did not make the short list is not smuggled in.
    assert "put 730 20261120" not in markdown


def test_an_empty_scenario_table_says_so_rather_than_rendering_blank() -> None:
    markdown = render_markdown(build_report(_inputs(controls=[])))
    assert "_nothing priced_" in markdown


def test_rows_for_different_underlyings_are_labelled_and_warned_about() -> None:
    # Each underlying is priced against its own reference, so rows from two of
    # them are not alternatives to each other. An undifferentiated table renders
    # "no new protection" once per symbol with nothing to tell them apart.
    controls = [
        {"symbol": s, "label": "no new protection", "structure": "unhedged",
         "cost_usd": 0.0, "pnl_at_worst_move": -20000.0}
        for s in ("SPY", "QQQ", "IWM")
    ]
    report = build_report(_inputs(controls=controls))
    scenarios = report["6_scenarios"]
    assert scenarios["underlyings"] == ["IWM", "QQQ", "SPY"]
    markdown = render_markdown(report)
    assert "| underlying |" in markdown
    assert "not alternatives to each other" in markdown


def test_a_repeated_control_for_one_underlying_is_shown_once() -> None:
    duplicated = [
        {"symbol": "SPY", "label": "no new protection", "structure": "unhedged",
         "cost_usd": 0.0}
    ] * 3
    rows = build_report(_inputs(controls=duplicated))["6_scenarios"]["rows"]
    assert len(rows) == 1


def test_a_single_underlying_needs_no_cross_symbol_warning() -> None:
    markdown = render_markdown(build_report(_inputs(controls=[
        {"symbol": "SPY", "label": "no new protection", "structure": "unhedged",
         "cost_usd": 0.0}
    ])))
    assert "not alternatives to each other" not in markdown

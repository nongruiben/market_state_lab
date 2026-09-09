"""The Chinese report, the reference-exposure line, and the oriented risk basis.

The product answers in two languages from one report structure, so the two
cannot drift into two answers; the reference exposure is the one number with a
measured 26-year ledger behind it; and a supporting-risk list must show the
oriented rank or it misleads on every inverted indicator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_state_lab.evaluation import (
    build_episodes,
    forward_drawdown_event,
    replay_metrics,
)
from market_state_lab.main_report import ReportInputs, build_report, render_html, render_markdown
from market_state_lab.market_assessment import (
    NO_NEW_DEFENSE_CASE,
    REVIEW_REDUCTION,
    WATCH,
    assess,
    derive_leaning,
)
from market_state_lab.market_evidence import build_evidence


def _prices(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2023-01-02", periods=700)
    return pd.DataFrame(
        {
            name: 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, 700)))
            for name in ("spy", "rsp", "iwm", "hyg", "lqd")
        },
        index=index,
    )


def _inputs(**over) -> ReportInputs:
    evidence = build_evidence(_prices(), None, None)
    fields = {
        "evidence": evidence,
        "assessment": assess(evidence, snapshot_id="2026-09-08-abc"),
        "session_date": "2026-09-08",
        "reference_exposure": {
            "trailing_exposure": 0.62,
            "ewma_exposure": 0.55,
            "trend_exposure": 1.0,
            "spot": 765.96,
            "trend_trigger_price": 710.43,
            "trend_distance_pct": 0.0782,
            "volatility_now_annual": 0.083,
            "volatility_reduce_above": 0.10,
            "target_volatility_annual": 0.10,
            "measured_history": "26-year ledger",
            "note": "a yardstick, not advice",
        },
        "registry_summary": {"registered": 5, "by_capability": {
            "risk_identification": {"may_influence": []},
            "action_value": {"may_influence": []},
            "data_quality": {"may_influence": []},
        }},
    }
    fields.update(over)
    return ReportInputs(**fields)


def test_the_chinese_report_carries_every_section_heading() -> None:
    markdown = render_markdown(build_report(_inputs()), "zh")
    for heading in (
        "## 1. 数据状态", "## 2. 对操作的含义", "## 3. 与上次相比",
        "## 4. 五个维度", "## 5. 减仓还是买保护",
        "## 6. 参考暴露下的情景损益", "## 7. 什么会改变这份读数,以及方法声明",
    ):
        assert heading in markdown


def test_the_chinese_report_translates_the_leaning_and_reason() -> None:
    markdown = render_markdown(build_report(_inputs()), "zh")
    assert "行动倾向:**证据不足**" in markdown
    assert "预先登记" in markdown
    assert "持续性基准是 0.5134" in markdown


def test_the_chinese_html_page_renders() -> None:
    page = render_html(build_report(_inputs()), "市场状态 2026-09-08", "zh")
    assert "<title>市场状态 2026-09-08</title>" in page
    assert "对操作的含义" in page


def test_the_english_report_is_unchanged_in_structure() -> None:
    markdown = render_markdown(build_report(_inputs()))
    assert "## 2. What this means for a position" in markdown
    assert "action leaning: **DATA_INSUFFICIENT**" in markdown


def test_the_reference_exposure_line_is_present_in_both_languages() -> None:
    """It moved to the page headline; the guarantee is that it is in both, once.

    Stating it twice left two copies of one number that could drift apart, and
    the buried copy was the reason the useful line was hard to find.
    """
    report = build_report(_inputs())
    english = render_markdown(report).split("## 1.")[0]
    chinese = render_markdown(report, "zh").split("## 1.")[0]
    assert "Volatility target 62%" in english and "EWMA variant 55%" in english
    assert "波动率目标 62%" in chinese and "EWMA 变体 55%" in chinese
    assert "yardstick, not advice" in english
    assert "标尺不是建议" in chinese
    # Once, not twice.
    assert render_markdown(report).count("Volatility target 62%") == 1


def test_the_conditional_frequencies_block_renders_in_both_languages() -> None:
    from market_state_lab.evaluation import conditional_frequencies
    rng = np.random.default_rng(7)
    spy = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 1200))),
        index=pd.bdate_range("2019-01-02", periods=1200),
    )
    conditionals = conditional_frequencies(spy)
    assert conditionals, "the conditionals block needs enough history to exist"
    report = build_report(_inputs(conditionals=conditionals))
    english = render_markdown(report)
    chinese = render_markdown(report, "zh")
    assert "historical base rate: a 5% drawdown within 20 sessions" in english
    assert "200-session average" in english
    assert "frequencies over settled history, not predicted probabilities" in english
    assert "历史基准:20 个交易日内出现 5% 跌幅" in chinese
    assert "200 日均线" in chinese
    assert "已结算历史的条件频率" in chinese


def test_the_method_declaration_lives_at_the_end_in_both_languages() -> None:
    report = build_report(_inputs())
    english = render_markdown(report)
    chinese = render_markdown(report, "zh")
    # The full honesty block sits in section 7, not in front of the operational
    # content: a report whose first page is disclaimers stops being read.
    assert english.index("## 2. What this means for a position") < english.index("Pre-registered.")
    assert chinese.index("## 2. 对操作的含义") < chinese.index("预先登记。")
    assert "## 7. What would change this, and the method behind it" in english
    assert "## 7. 什么会改变这份读数,以及方法声明" in chinese


def test_the_reference_exposure_is_a_yardstick_in_both_languages() -> None:
    report = build_report(_inputs())
    assert "not advice" in render_markdown(report)
    assert "不是建议" in render_markdown(report, "zh")


def test_the_registry_summary_line_names_what_may_influence() -> None:
    report = build_report(_inputs())
    assert "experiments allowed to influence a recommendation: none" in render_markdown(report)
    assert "被允许影响建议的实验:无" in render_markdown(report, "zh")


def test_the_supporting_risk_list_uses_the_oriented_rank() -> None:
    # spy_vs_200d_ma is inverted: its raw percentile is high exactly when risk
    # is low. The basis line must carry the oriented rank, not the raw one.
    evidence = build_evidence(_prices(), None, None)
    assessment = assess(evidence)
    for line in assessment.risk_basis:
        if "spy_vs_200d_ma" in line:
            assert "risk rank" in line
            assert "of history more favourable" in line
    report = build_report(_inputs())
    for line in report["2_judgment"]["supporting"]:
        if "spy_vs_200d_ma" in line:
            assert "risk rank" in line


def test_derive_leaning_maps_the_section_9_2_table() -> None:
    assert derive_leaning({"trend_damaged", "volatility_elevated"})[0] == REVIEW_REDUCTION
    assert derive_leaning({"volatility_elevated"})[0] == "REVIEW_HEDGE"
    assert derive_leaning({"trend_damaged"})[0] == WATCH
    assert derive_leaning({"repair_underway", "volatility_elevated"})[0] == "CONFLICTED"
    assert derive_leaning(set())[0] == NO_NEW_DEFENSE_CASE


def test_an_untested_rule_set_never_leaves_data_insufficient() -> None:
    assessment = assess(build_evidence(_prices(), None, None))
    assert assessment.action_leaning == "DATA_INSUFFICIENT"
    # And the Assessment constructor still refuses an untested grade leaning.
    import pytest as _pytest

    from market_state_lab.market_assessment import UNTESTED, Assessment
    with _pytest.raises(ValueError):
        Assessment(
            as_of=None, snapshot_id=None, observation_horizon="20 sessions",
            action_leaning=WATCH, research_grade=UNTESTED,
        )


# ---------------------------------------------------------------------------
# Episodes and replay metrics
# ---------------------------------------------------------------------------


def test_episodes_merge_only_fired_event_days() -> None:
    index = pd.bdate_range("2020-01-02", periods=200)
    event = pd.Series(0.0, index=index)
    event.iloc[10:30] = 1.0
    event.iloc[80:100] = 1.0
    event.iloc[160:175] = 1.0
    episodes = build_episodes(event, gap_sessions=5)
    assert len(episodes) == 3
    assert episodes["event_days"].tolist() == [20, 20, 15]
    # Zero days do not form an episode.
    assert build_episodes(event * 0.0, gap_sessions=5).empty


def test_replay_metrics_counts_episodes_and_false_alarms() -> None:
    index = pd.bdate_range("2020-01-02", periods=200)
    event = pd.Series(0.0, index=index)
    event.iloc[10:30] = 1.0
    event.iloc[80:100] = 1.0
    alerts = pd.Series(0.0, index=index)
    alerts.iloc[15:25] = 1.0   # inside episode 1
    alerts.iloc[50:60] = 1.0   # a false alarm
    metrics = replay_metrics(alerts, event, build_episodes(event))
    assert metrics["episodes"] == 2
    assert metrics["episodes_detected"] == 1
    assert metrics["episode_recall"] == 0.5
    assert metrics["false_alert_sessions"] == 10


def test_the_forward_event_is_never_available_when_it_describes() -> None:
    spy = pd.Series(
        100.0 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.02, 300))),
        index=pd.bdate_range("2021-01-04", periods=300),
    )
    event = forward_drawdown_event(spy, 20, 0.05)
    # The last horizon sessions cannot be known yet.
    assert event.iloc[-20:].notna().sum() == 0


def test_the_rank_strip_leads_the_page_and_is_ordered_by_risk() -> None:
    """Twelve rows of numbers do not answer "is anything unusual today"; the eye
    has to read and compare every one. A sorted bar answers it at a glance."""
    from market_state_lab.main_report import render_html

    html = render_html(build_report(_inputs()))
    strip = html.index("class='strip'")
    # Above the prose: every paragraph before it delays the answer.
    assert strip < html.index("<h2>1.") if "<h2>1." in html else strip < len(html)
    assert html.index("<h1>") < strip
    # Bars are widths, so no chart library is needed and the page opens offline.
    assert "cdn" not in html and "<script" not in html


def test_the_bars_are_ordered_riskiest_first() -> None:
    import re

    from market_state_lab.main_report import render_html

    # Only the bars, not the stylesheet's own `td.bar{width:55%}`.
    widths = [
        float(w)
        for w in re.findall(
            r"width:(\d+)%;background:", render_html(build_report(_inputs()))
        )
    ]
    assert widths
    assert widths == sorted(widths, reverse=True)


def test_an_overlapping_indicator_gets_no_bar() -> None:
    import re

    from market_state_lab.main_report import render_html

    # It carries the same risk appetite an equity dimension already measures;
    # a bar would let it vote twice on the same glance.
    html = render_html(build_report(_inputs()))
    assert "hyg_over_lqd" not in re.sub(r"<table(?! class='strip').*?</table>", "", html,
                                        flags=re.S).split("</table>")[0]


def test_the_frequency_chart_marks_the_condition_that_holds_today() -> None:
    """"Which of these am I in" is the question a reader brings; the chart
    should not make them work it out."""
    from market_state_lab.evaluation import conditional_frequencies
    from market_state_lab.main_report import render_html

    spy = _prices()["spy"]
    html = render_html(build_report(_inputs(conditionals=conditional_frequencies(spy))))
    assert "How often a 5% drawdown followed" in html
    # The marker is an entity, not the literal text of one.
    assert "&amp;larr;" not in html
    assert "class=here" in html


def test_the_headline_states_both_rules_and_the_level_each_turns_at() -> None:
    # An exposure without its trigger says what to hold and not when that
    # changes, which is the half a reader has to act on.
    markdown = render_markdown(build_report(_inputs()))
    head = markdown.split("## 1.")[0]
    assert "Trend rule" in head and "Volatility target" in head
    assert "200-session average" in head
    assert "scales down one for one" in head
    assert "Neither is a forecast" in head


def test_an_exposure_without_its_trigger_is_still_stated() -> None:
    # The exposure is useful on its own; suppressing it for want of the trigger
    # would lose the more important half.
    bare = {"trend_exposure": 0.0, "trailing_exposure": 0.45}
    markdown = render_markdown(build_report(_inputs(reference_exposure=bare)))
    head = markdown.split("## 1.")[0]
    # 0% is the rule saying hold nothing - the most consequential thing it says,
    # and exactly what a truthiness check would delete.
    assert "Trend rule 0%" in head
    assert "Volatility target 45%" in head


def test_no_exposure_at_all_renders_no_heading() -> None:
    markdown = render_markdown(build_report(_inputs(reference_exposure={})))
    # A heading and a caveat with nothing between them is worse than silence.
    assert "Reference exposure" not in markdown.split("## 1.")[0]

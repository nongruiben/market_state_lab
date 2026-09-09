"""The main report: seven sections, in the order section 13 fixes them.

Order is the argument. Data status comes before any reading of the market, the
reading comes before the instruments, and the instruments come before the
payoff arithmetic - so a reader cannot arrive at a put's cost without having
passed the sentence saying what the quotes behind it were worth.

Four outcomes are normal results and are rendered as such rather than as
failures: insufficient data, no suitable instrument, risk that is real while
protection is expensive, and evidence that conflicts. A report that can only
say useful things will eventually say one that is not true.

No field here comes from an account. There is no holding, no position, no cash
balance and no order, and a test asserts their names never appear - the
reference exposure is a stated yardstick and the report has no way to know what
anyone owns.

Text is templated structure over facts already computed upstream. Nothing in
this module derives a price, invents an event, or explains a cause.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from market_state_lab.market_assessment import DATA_INSUFFICIENT, Assessment
from market_state_lab.market_evidence import MarketEvidence, contradictions

MAX_EVIDENCE_ITEMS = 3

# Names that must never reach a report. The project reads none of them, and the
# cheapest way to keep it that way is to fail loudly if one appears.
ACCOUNT_FIELDS = (
    "position", "holding", "portfolio_value", "cash_balance", "buying_power",
    "order", "execution", "account_id", "net_liquidation",
)


@dataclass
class ReportInputs:
    """Everything the report renders, computed elsewhere and passed in."""

    evidence: MarketEvidence
    assessment: Assessment
    snapshot_id: str | None = None
    session_date: str | None = None
    data_quality: dict[str, Any] = field(default_factory=dict)
    eligible_for: tuple[str, ...] = ()
    intraday_provisional: bool = False
    new_events: list[Any] = field(default_factory=list)
    open_events: list[Any] = field(default_factory=list)
    previous_labels: list[str] = field(default_factory=list)
    candidates: pd.DataFrame | None = None
    controls: list[dict[str, Any]] = field(default_factory=list)
    reference_notional: float = 100_000.0
    reconciliation: dict[str, Any] | None = None
    reference_exposure: dict[str, Any] = field(default_factory=dict)
    conditionals: dict[str, Any] = field(default_factory=dict)
    registry_summary: dict[str, Any] = field(default_factory=dict)


def _data_status(inputs: ReportInputs) -> dict[str, Any]:
    """Section 1. What the numbers below are worth, before any of them appear."""
    return {
        "session_date": inputs.session_date,
        "snapshot_id": inputs.snapshot_id,
        "evidence_as_of": inputs.evidence.as_of,
        "assessment_updated": inputs.assessment.as_of,
        "intraday_provisional": inputs.intraday_provisional,
        "approved_uses": list(inputs.eligible_for),
        "quality": inputs.data_quality,
        "cross_source": inputs.reconciliation,
        "note": (
            "a provisional intraday reading is not a settled session"
            if inputs.intraday_provisional
            else "a settled session"
        ),
    }


def _judgment(inputs: ReportInputs) -> dict[str, Any]:
    """Section 2. The leaning, and at most three items on each side."""
    assessment = inputs.assessment
    return {
        "state_labels": assessment.state_labels or ["none of the state labels currently fit"],
        "action_leaning": assessment.action_leaning,
        "leaning_reason": assessment.leaning_reason,
        "research_grade": assessment.research_grade,
        "supporting": assessment.risk_basis[:MAX_EVIDENCE_ITEMS],
        "contrary": assessment.contrary_evidence[:MAX_EVIDENCE_ITEMS],
        "supporting_withheld": max(0, len(assessment.risk_basis) - MAX_EVIDENCE_ITEMS),
        "contrary_withheld": max(0, len(assessment.contrary_evidence) - MAX_EVIDENCE_ITEMS),
        "normal_outcome": assessment.action_leaning == DATA_INSUFFICIENT,
        # The one number with a measured history behind it: the volatility
        # target never predicts, and its 26-year ledger is on record.
        "reference_exposure": inputs.reference_exposure,
        # Causal frequencies of the research event around today's state - the
        # operational answer a percentile cannot give.
        "conditionals": inputs.conditionals,
    }


def _changes(inputs: ReportInputs) -> dict[str, Any]:
    """Section 3. What moved since last time, and whether it is an event yet."""
    now = set(inputs.assessment.state_labels)
    before = set(inputs.previous_labels)
    return {
        "labels_added": sorted(now - before),
        "labels_dropped": sorted(before - now),
        "new_events": [
            {"label": n.label, "kind": n.kind, "detail": n.detail} for n in inputs.new_events
        ],
        "standing_events": [
            {"label": e.label, "since": e.first_seen} for e in inputs.open_events
        ],
        "note": (
            "a label appearing is not yet an event; it becomes one only after it holds"
            if (now - before) and not inputs.new_events
            else ""
        ),
    }


def _dimensions(inputs: ReportInputs) -> dict[str, Any]:
    """Section 4. Every indicator, and every dimension that is absent."""
    return {
        "indicators": [i.as_dict() for i in inputs.evidence.indicators],
        "missing_dimensions": list(inputs.evidence.missing_dimensions),
        "contradictions": contradictions(inputs.evidence),
        "missing_note": (
            "a missing dimension is unknown, not calm"
            if inputs.evidence.missing_dimensions
            else "every dimension reported"
        ),
    }


def _instruments(inputs: ReportInputs) -> dict[str, Any]:
    """Section 5. Reduction against protection, as a conditional comparison."""
    candidates = inputs.candidates
    shortlisted: list[dict[str, Any]] = []
    if candidates is not None and not candidates.empty and "shortlisted" in candidates:
        shortlisted = candidates.loc[candidates["shortlisted"]].to_dict("records")
    return {
        "reference_notional_usd": inputs.reference_notional,
        "candidates": shortlisted[:MAX_EVIDENCE_ITEMS],
        "controls": inputs.controls,
        "conditional": (
            "If you would rather hold less, compare the de-risk control. If you would "
            "rather keep the exposure and pay to insure it, compare the puts. Market "
            "data cannot choose between those two preferences."
        ),
        "normal_outcome": (
            "no candidate cleared the screen; the controls are the whole comparison"
            if not shortlisted
            else ""
        ),
    }


def _scenarios(inputs: ReportInputs) -> dict[str, Any]:
    """Section 6. Payoff, cost and what is still exposed, on one yardstick."""
    # One control pair per underlying, not per snapshot row: a controls table
    # covering three symbols renders "no new protection" three times with
    # nothing to tell them apart.
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for row in inputs.controls:
        key = (str(row.get("symbol", "")), str(row.get("label", "")))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    if inputs.candidates is not None and not inputs.candidates.empty:
        picked = inputs.candidates
        if "shortlisted" in picked:
            picked = picked.loc[picked["shortlisted"]]
        rows = picked.to_dict("records")[:MAX_EVIDENCE_ITEMS] + rows
    return {
        "reference_notional_usd": inputs.reference_notional,
        "rows": [
            {
                key: row.get(key)
                for key in (
                    "symbol", "label", "structure", "contracts", "cost_usd",
                    "cost_pct_of_notional", "pnl_at_worst_move", "protection_at_worst_move",
                    "pnl_if_flat", "pnl_at_best_move", "protected_below", "uncovered_shares",
                    "coverage_ratio", "market_data_type",
                )
            }
            for row in rows
        ],
        "underlyings": sorted({str(row.get("symbol")) for row in rows if row.get("symbol")}),
        "note": (
            "payoff is at expiry only; the reference exposure is a stated yardstick and "
            "not anyone's position"
        ),
        "cross_symbol_warning": (
            "each underlying is priced against its own reference exposure, so rows for "
            "different symbols are not alternatives to each other; comparing them as "
            "hedges for one portfolio would need a fixed common reference and a stated "
            "mapping, which this does not have"
        ),
    }


def _review(inputs: ReportInputs) -> dict[str, Any]:
    """Section 7. What would change this, and how well tested any of it is."""
    return {
        "triggers": inputs.assessment.triggers[:6],
        "next_review_reason": inputs.assessment.next_review_reason,
        "research_grade": inputs.assessment.research_grade,
        "applicable_conditions": inputs.assessment.applicable_conditions,
        "research_registry": inputs.registry_summary,
    }


def account_fields_present(value: Any, path: str = "") -> list[str]:
    """Any key in the structure that names something from an account.

    A structural check rather than a scan of the prose, because the prose says
    "not anyone's position" and a substring search cannot tell a disclaimer from
    a disclosure. Field names are the thing that would actually leak.
    """
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            here = f"{path}.{key}" if path else str(key)
            if any(bad in str(key).lower() for bad in ACCOUNT_FIELDS):
                found.append(here)
            found += account_fields_present(item, here)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found += account_fields_present(item, f"{path}[{index}]")
    return found


def build_report(inputs: ReportInputs) -> dict[str, Any]:
    """The seven sections, in the order the plan fixes them.

    Refuses to return a report carrying an account-shaped field. The project
    reads none of them, and a loud failure is cheaper than discovering one in a
    rendered page.
    """
    report = _sections(inputs)
    leaked = account_fields_present(report)
    if leaked:
        raise ValueError(f"account-shaped fields must never reach a report: {leaked}")
    return report


def _sections(inputs: ReportInputs) -> dict[str, Any]:
    return {
        "1_data_status": _data_status(inputs),
        "2_judgment": _judgment(inputs),
        "3_changes": _changes(inputs),
        "4_dimensions": _dimensions(inputs),
        "5_instruments": _instruments(inputs),
        "6_scenarios": _scenarios(inputs),
        "7_review": _review(inputs),
    }


def _leaning_zh(leaning: str) -> str:
    return {
        "NO_NEW_DEFENSE_CASE": "无新增防御依据",
        "WATCH": "观察",
        "REVIEW_REDUCTION": "考虑减仓",
        "REVIEW_HEDGE": "考虑保护",
        "CONFLICTED": "证据冲突",
        "DATA_INSUFFICIENT": "证据不足",
    }.get(leaning, leaning)


def _state_zh(label: str) -> str:
    return {
        "trend_damaged": "趋势受损",
        "volatility_elevated": "波动升高",
        "stress_spreading": "压力扩散",
        "repair_underway": "修复进行中",
        "none of the state labels currently fit": "当前没有任何状态标签适用",
    }.get(label, label)


_DIMENSION_ZH = {
    "trend": "趋势",
    "volatility": "波动",
    "participation": "参与面",
    "credit": "信用",
    "implied_risk": "隐含风险",
}
_DIRECTION_ZH = {"deteriorating": "恶化", "improving": "改善", "stable": "平稳"}
_UNIT_ZH = {
    "% above the 200-session average": "% 高于200日均线",
    "% over 63 sessions": "% 63个交易日变化",
    "% below the 252-session high": "% 低于252日高点",
    "% annualised": "% 年化",
    "ratio": "比值",
    "percentage points": "个百分点",
    "index points": "指数点位",
}

# The pre-registered insufficient-evidence paragraph, in both languages. The
# English text is what assess() stores; the Chinese text is its translation,
# fixed here rather than generated so the two cannot drift in meaning.
_PRE_REGISTERED_REASON_ZH = (
    "预先登记。做出行动倾向所需的预测能力已在本项目的数据上被测量过,且不存在:"
    "20 日波动率状态上校准集成的 Brier 为 0.5721,而持续性基准是 0.5134;"
    "回撤优势在等仓位对照下消失;126 日上没有预测器在 2013 年后显著;"
    "连续五个特征家族返回空值。这是预期答案,不是今天数据不足,"
    "只有当某条规则通过晋级闸门后它才会改变。"
)

_LEANING_REASON_ZH = {
    "REVIEW_REDUCTION": (
        "趋势受损得到第二个独立维度的确认;考虑降低市场暴露,并并列比较保护方案"
    ),
    "REVIEW_HEDGE": (
        "短期压力升高而长期趋势未明显受损;比较有限期保护与继续观察"
    ),
    "CONFLICTED": "修复信号与仍然存在的损伤互相冲突;两份证据都要读",
    "WATCH": "单一维度值得注意;尚不足以改变暴露",
    "NO_NEW_DEFENSE_CASE": "没有值得注意的恶化;这不是对安全的预测,未覆盖风险仍然存在",
}

_PURPOSE_ZH = {
    "day_end_analysis": "日终分析",
    "intraday_observation": "盘中观察",
    "instrument_quotes": "工具报价",
    "training": "训练",
}

# Fixed phrases produced by upstream modules, mapped rather than re-derived so
# the two languages are one document with two fixed renderings.
_ZH_PHRASES = {
    "none of the state labels currently fit": "当前没有任何状态标签适用",
    "a settled session": "已结算交易日",
    "a provisional intraday reading is not a settled session": "盘中临时读数不是已结算交易日",
    "rule judgement, gain unvalidated": "规则判断,增益未验证",
    "replayed, gain measured": "已回放,增益已测量",
    "validated against a strong benchmark, out of sample": "已对强基准做样本外验证",
    "pending replay; the delay and false-alarm costs are unmeasured": "待回放;延迟与误报代价未测量",
    "If you would rather hold less, compare the de-risk control. If you would "
    "rather keep the exposure and pay to insure it, compare the puts. Market "
    "data cannot choose between those two preferences.": (
        "若你宁愿少持有,对照减仓方案;若你宁愿保留暴露并付费投保,对照 put 候选。"
        "市场数据无法在两种偏好之间替你选择。"
    ),
    "no candidate cleared the screen; the controls are the whole comparison": (
        "没有候选通过筛选;对照项就是全部比较内容"
    ),
    "payoff is at expiry only; the reference exposure is a stated yardstick and "
    "not anyone's position": "损益仅按到期结算;参考暴露是明确声明的标尺,不是任何人的持仓",
    "each underlying is priced against its own reference exposure, so rows for "
    "different symbols are not alternatives to each other; comparing them as "
    "hedges for one portfolio would need a fixed common reference and a stated "
    "mapping, which this does not have": (
        "每个标的都按其各自的参考暴露计价,因此不同标的的行彼此不是替代方案;"
        "要把它们当作同一组合的对冲来比较,需要一个固定的共同参考和明确的映射假设,"
        "而本报告没有这些"
    ),
    "a missing dimension is unknown, not calm": "缺失的维度是未知,不是平静",
    "every dimension reported": "所有维度均有报告",
    "a label appearing is not yet an event; it becomes one only after it holds": (
        "标签出现还不是事件;只有持续成立后才成为事件"
    ),
    "Reducing exposure and buying protection answer different preferences: "
    "how much upside you are willing to give up, and how much you will pay "
    "to keep it. Market data cannot choose between them.": (
        "减仓与买保护回答的是两种不同偏好:你愿意放弃多少上涨,又愿意付出多少来保留它。"
        "市场数据无法在两者之间替你选择。"
    ),
    "Nothing here reads a position, so none of it is advice about holdings.": (
        "本报告不读取任何持仓,因此其中没有任何内容是关于持仓的建议。"
    ),
    "no independent second source in this run; re-run with --with-ibkr, or use "
    "scripts/reconcile_spy.py, to compare against TWS": (
        "本次运行没有独立第二数据源;加 --with-ibkr 重跑,或用 scripts/reconcile_spy.py "
        "与 TWS 对照"
    ),
}

_EVENT_KIND_ZH = {"opened": "开启", "released": "解除", "exceptional_review": "例外复查"}


def _zh(text: str) -> str:
    """A fixed phrase, or a patterned one, or the text unchanged.

    Upstream modules produce English prose for triggers, contrary evidence and
    the like. Their templates are stable, so the Chinese rendering translates
    the template rather than duplicating the logic that produced it.
    """
    if text in _ZH_PHRASES:
        return _ZH_PHRASES[text]
    import re

    match = re.match(
        r"^(.+) sits at the (\d+)% rank of its own history; "
        r"a move past (\d+)% or (\d+)% would make this description stale$",
        text,
    )
    if match:
        name, rank, above, below = match.groups()
        return f"{name} 处于自身历史的 {rank}% 分位;越过 {above}% 或 {below}% 会令这份描述过时"
    match = re.match(r"^(.+) improved over 20 sessions$", text)
    if match:
        return f"{match.group(1)} 在 20 个交易日改善"
    match = re.match(
        r"^(.+) risk rank (\d+)% \((.+), (\d+)% of history more favourable\)$",
        text,
    )
    if match:
        name, rank, value_unit, favourable = match.groups()
        value, _, unit = value_unit.partition(" ")
        return (
            f"{name} 风险分位 {rank}%({value} {_UNIT_ZH.get(unit, unit)},"
            f"{favourable}% 的历史更有利)"
        )
    return text


def _indicator_table(indicators: list[dict[str, Any]], language: str = "en") -> str:
    if language == "zh":
        header = "| 维度 | 指标 | 数值 | 单位 | 20日变化 | 分位 | 方向 |"
        rule = "|---|---|---|---|---|---|---|"
        rows = [
            "| {dimension} | {name} | {value} | {unit} | {change} | {rank} | {direction} |".format(
                dimension=_DIMENSION_ZH.get(i["dimension"], i["dimension"]),
                name=i["name"],
                value="—" if i["value"] is None else f"{i['value']:.2f}",
                unit=_UNIT_ZH.get(i["unit"], i["unit"]),
                change="—" if i["change_20"] is None else f"{i['change_20']:+.2f}",
                rank="—" if i["percentile"] is None else f"{i['percentile']:.0%}",
                direction=_DIRECTION_ZH.get(i["direction"], i["direction"]),
            )
            for i in indicators
        ]
        return "\n".join([header, rule, *rows])
    header = "| dimension | indicator | value | unit | 20-session change | rank | direction |"
    rule = "|---|---|---|---|---|---|---|"
    rows = [
        "| {dimension} | {name} | {value} | {unit} | {change} | {rank} | {direction} |".format(
            dimension=i["dimension"],
            name=i["name"],
            value="—" if i["value"] is None else f"{i['value']:.2f}",
            unit=i["unit"],
            change="—" if i["change_20"] is None else f"{i['change_20']:+.2f}",
            rank="—" if i["percentile"] is None else f"{i['percentile']:.0%}",
            direction=i["direction"],
        )
        for i in indicators
    ]
    return "\n".join([header, rule, *rows])


def _condition_lines(conditionals: dict[str, Any], language: str = "en") -> list[str]:
    """The operational block: what history says about states like today's.

    Frequencies over settled history, computed causally and labelled as
    frequencies. This is the answer the report exists to give - a percentile
    alone cannot say what a given state was followed by.
    """
    if not conditionals or "current" not in conditionals:
        return []

    def pct(value: Any) -> str:
        return "—" if value is None else f"{float(value):.0%}"

    base = conditionals.get("base_rate", {})
    above = conditionals.get("above_200d_ma", {})
    below = conditionals.get("below_200d_ma", {})
    tail = conditionals.get("vol_at_least_current", {})
    current = conditionals.get("current", {})
    event_name = "5% 跌幅" if language == "zh" else "a 5% drawdown within 20 sessions"

    if language == "zh":
        lines = [
            f"- 历史基准:20 个交易日内出现 {event_name}的频率,历史上有 "
            f"{pct(base.get('event_frequency'))} 个交易日之后发生,"
            f"2013 年以来为 {pct(base.get('event_frequency_post_2013'))}",
            f"- 价格在 200 日均线**上方**:该频率为 {pct(above.get('event_frequency'))}"
            f"(2013 年以来 {pct(above.get('event_frequency_post_2013'))});"
            f"**下方**:{pct(below.get('event_frequency'))}"
            f"(2013 年以来 {pct(below.get('event_frequency_post_2013'))})",
            f"- 20 日已实现波动不低于当前水平(当前处于历史 "
            f"{pct(current.get('vol_rank'))} 分位)的日子:该频率为 "
            f"{pct(tail.get('event_frequency'))}"
            f"(2013 年以来 {pct(tail.get('event_frequency_post_2013'))})",
            "- 以上是已结算历史的条件频率,不是对未来概率的预测",
        ]
        return lines
    return [
        f"- historical base rate: a 5% drawdown within 20 sessions followed "
        f"{pct(base.get('event_frequency'))} of sessions "
        f"({pct(base.get('event_frequency_post_2013'))} since 2013)",
        f"- with price **above** its 200-session average that frequency was "
        f"{pct(above.get('event_frequency'))} "
        f"({pct(above.get('event_frequency_post_2013'))} since 2013); "
        f"**below** it, {pct(below.get('event_frequency'))} "
        f"({pct(below.get('event_frequency_post_2013'))} since 2013)",
        f"- on days when 20-session realised volatility ranked at least as high "
        f"as now (the {pct(current.get('vol_rank'))} rank), the frequency was "
        f"{pct(tail.get('event_frequency'))} "
        f"({pct(tail.get('event_frequency_post_2013'))} since 2013)",
        "- frequencies over settled history, not predicted probabilities",
    ]


def _method_lines(
    judgment: dict[str, Any],
    review: dict[str, Any],
    language: str = "en",
) -> list[str]:
    """The compressed honesty block: grade, reason and registry in one place.

    Everything a sceptical reader needs to discount the report, kept out of the
    operational sections and put where it cannot be missed either.
    """
    grade = judgment.get("research_grade", "")
    reason = judgment.get("leaning_reason", "")
    if language == "zh":
        if judgment.get("action_leaning") == DATA_INSUFFICIENT:
            reason = _PRE_REGISTERED_REASON_ZH
        else:
            reason = _LEANING_REASON_ZH.get(judgment.get("action_leaning"), reason)
        lines = [
            f"- 研究等级:{_zh(grade)}",
            f"- 为什么行动倾向是{_leaning_zh(judgment.get('action_leaning', ''))}:{reason}",
        ]
        lines += _registry_lines(review.get("research_registry", {}), "zh")
        return lines
    lines = [
        f"- research grade: {grade}",
        f"- why the leaning is {judgment.get('action_leaning', '')}: {reason}",
    ]
    lines += _registry_lines(review.get("research_registry", {}), "en")
    return lines


def _registry_lines(summary: dict[str, Any], language: str = "en") -> list[str]:
    if not summary:
        return []
    may_influence = [
        name
        for capability in summary.get("by_capability", {}).values()
        for name in capability.get("may_influence", [])
    ]
    if language == "zh":
        lines = [f"- 研究登记:共 {summary.get('registered', 0)} 个实验"]
        lines.append(
            f"- 被允许影响建议的实验:{'、'.join(may_influence) if may_influence else '无'}"
        )
        return lines
    return [
        f"- research registry: {summary.get('registered', 0)} experiments registered",
        f"- experiments allowed to influence a recommendation: "
        f"{', '.join(may_influence) if may_influence else 'none'}",
    ]


def render_markdown(report: dict[str, Any], language: str = "en") -> str:
    """A templated rendering. No sentence here is generated from the numbers.

    `language` selects the fixed template set; the report structure is the same
    document in both, so the two cannot drift into two answers.
    """
    if language not in {"en", "zh"}:
        raise ValueError(f"unknown report language {language!r}")
    status = report["1_data_status"]
    judgment = report["2_judgment"]
    changes = report["3_changes"]
    dimensions = report["4_dimensions"]
    instruments = report["5_instruments"]
    scenarios = report["6_scenarios"]
    review = report["7_review"]
    zh = language == "zh"

    if zh:
        state_text = "、".join(_state_zh(s) for s in judgment["state_labels"])
        leaning_text = _leaning_zh(judgment["action_leaning"])
        lines = [
            f"# 市场状态,交易日 {status['session_date']}",
            "",
            *_headline(inputs_of(report), "zh"),
            "## 1. 数据状态",
            f"- 快照:`{status['snapshot_id']}`",
            f"- 证据截至:{status['evidence_as_of']}",
            f"- 已批准用途:{'、'.join(_PURPOSE_ZH.get(u, u) for u in status['approved_uses']) or '未授予任何用途'}",
            f"- {_zh(status['note'])}",
        ]
        if status.get("cross_source"):
            cross = status["cross_source"]
            if cross.get("compared_days"):
                lines.append(
                    f"- 跨源核验:{cross.get('compared_days')} 个交易日对照 "
                    f"{'、'.join(cross.get('sources', []))},"
                    f"{cross.get('days_outside_tolerance')} 天超容差"
                )
            else:
                lines.append(f"- 跨源核验:{_zh(cross.get('note', '未运行'))}")
        leaning_short = (
            "没有规则赢得过输出行动倾向的资格;请用下面的历史频率和参考暴露自己判断"
            if judgment["action_leaning"] == DATA_INSUFFICIENT
            else _LEANING_REASON_ZH.get(judgment["action_leaning"], "")
        )
        lines += [
            "",
            "## 2. 对操作的含义",
            f"- 状态:{state_text}",
            f"- 行动倾向:**{leaning_text}**——{leaning_short}",
            "",
            * _condition_lines(judgment.get("conditionals", {}), "zh"),
            "",
            "",
            "**支持当前状态值得注意的证据**" if judgment["supporting"] else "**支持** — 无",
        ]
        lines += [f"- {_zh(item)}" for item in judgment["supporting"]]
        if judgment["supporting_withheld"]:
            lines.append(f"-(另有 {judgment['supporting_withheld']} 项未显示)")
        lines += ["", "**相反的证据**" if judgment["contrary"] else "**相反的证据** — 无"]
        lines += [f"- {_zh(item)}" for item in judgment["contrary"]]
        if judgment["contrary_withheld"]:
            lines.append(f"-(另有 {judgment['contrary_withheld']} 项未显示)")

        lines += ["", "## 3. 与上次相比"]
        lines += [f"- 新增标签:{_state_zh(label)}" for label in changes["labels_added"]]
        lines += [f"- 消失标签:{_state_zh(label)}" for label in changes["labels_dropped"]]
        lines += [
            f"- {_EVENT_KIND_ZH.get(e['kind'], e['kind'])}:{e['detail']}" for e in changes["new_events"]
        ]
        if not (changes["labels_added"] or changes["labels_dropped"] or changes["new_events"]):
            lines.append("- 无变化")
        if changes["note"]:
            lines.append(f"- {_zh(changes['note'])}")

        lines += ["", "## 4. 五个维度", "", _indicator_table(dimensions["indicators"], "zh")]
        if dimensions["missing_dimensions"]:
            missing = "、".join(_DIMENSION_ZH.get(d, d) for d in dimensions["missing_dimensions"])
            lines.append(f"\n缺失:{missing} — {_zh(dimensions['missing_note'])}。")
        for clash in dimensions["contradictions"]:
            if clash.get("kind") == "dimensions_disagree":
                left = _DIMENSION_ZH.get(clash.get("deteriorating"), clash.get("deteriorating"))
                right = _DIMENSION_ZH.get(clash.get("improving"), clash.get("improving"))
                lines.append(f"- 维度矛盾:{left}在 20 个交易日恶化,而{right}改善;两份读数都不被消减")
            else:
                dimension = _DIMENSION_ZH.get(clash.get("dimension"), clash.get("dimension"))
                lines.append(f"- 维度内部分歧:{dimension}的指标方向相反,该维度没有单一读数")

        lines += ["", "## 5. 减仓还是买保护", "", _zh(instruments["conditional"])]
        if instruments["normal_outcome"]:
            lines.append(f"\n{_zh(instruments['normal_outcome'])}")

        lines += ["", "## 6. 参考暴露下的情景损益", ""]
        lines.append(
            f"参考:每个标的 ${scenarios['reference_notional_usd']:,.0f}。{_zh(scenarios['note'])}"
        )
        if len(scenarios.get("underlyings", [])) > 1:
            lines.append("")
            lines.append(_zh(scenarios["cross_symbol_warning"]))
        lines.append("")
        lines.append(_scenario_table(scenarios["rows"]))

        lines += ["", "## 7. 什么会改变这份读数,以及方法声明"]
        lines += [f"- {_zh(t['detail'])}" for t in review["triggers"]]
        lines += [f"- {_zh(c)}" for c in review["applicable_conditions"]]
        lines += _method_lines(judgment, review, "zh")
        return "\n".join(lines)

    state_text = ", ".join(judgment["state_labels"])
    lines = [
        f"# Market state, session {status['session_date']}",
        "",
        *_headline(inputs_of(report), "en"),
        "## 1. Data status",
        f"- snapshot: `{status['snapshot_id']}`",
        f"- evidence as of: {status['evidence_as_of']}",
        f"- approved uses: {', '.join(status['approved_uses']) or 'none granted'}",
        f"- {status['note']}",
    ]
    if status.get("cross_source"):
        cross = status["cross_source"]
        if cross.get("compared_days"):
            lines.append(
                f"- cross-source: {cross.get('compared_days')} sessions against "
                f"{', '.join(cross.get('sources', []))}, "
                f"{cross.get('days_outside_tolerance')} outside tolerance"
            )
        else:
            # An absent check has to appear, or a reader assumes it passed.
            lines.append(f"- cross-source: {cross.get('note', 'not run')}")

    leaning_short = (
        "no rule has earned the right to lean; use the frequencies and the "
        "reference exposure below"
        if judgment["action_leaning"] == DATA_INSUFFICIENT
        else judgment["leaning_reason"]
    )
    lines += [
        "",
        "## 2. What this means for a position",
        f"- state: {state_text}",
        f"- action leaning: **{judgment['action_leaning']}** - {leaning_short}",
        "",
        * _condition_lines(judgment.get("conditionals", {}), "en"),
        "",
        "",
        "**Evidence for attention**" if judgment["supporting"] else "**Supporting** — none",
    ]
    lines += [f"- {item}" for item in judgment["supporting"]]
    if judgment["supporting_withheld"]:
        lines.append(f"- ({judgment['supporting_withheld']} further items not shown)")
    lines += ["", "**Evidence against**" if judgment["contrary"] else "**Against** — none"]
    lines += [f"- {item}" for item in judgment["contrary"]]
    if judgment["contrary_withheld"]:
        lines.append(f"- ({judgment['contrary_withheld']} further items not shown)")

    lines += ["", "## 3. Since last time"]
    lines += [f"- label added: {label}" for label in changes["labels_added"]]
    lines += [f"- label dropped: {label}" for label in changes["labels_dropped"]]
    lines += [f"- {e['kind']}: {e['detail']}" for e in changes["new_events"]]
    if not (changes["labels_added"] or changes["labels_dropped"] or changes["new_events"]):
        lines.append("- nothing changed")
    if changes["note"]:
        lines.append(f"- {changes['note']}")

    lines += ["", "## 4. Dimensions", "", _indicator_table(dimensions["indicators"], "en")]
    if dimensions["missing_dimensions"]:
        lines.append(
            f"\nMissing: {', '.join(dimensions['missing_dimensions'])} — "
            f"{dimensions['missing_note']}."
        )
    for clash in dimensions["contradictions"]:
        lines.append(f"- disagreement: {clash['detail']}")

    lines += ["", "## 5. Reduce or protect", "", instruments["conditional"]]
    if instruments["normal_outcome"]:
        lines.append(f"\n{instruments['normal_outcome']}")

    lines += ["", "## 6. Payoff against the reference exposure", ""]
    lines.append(
        f"Reference: ${scenarios['reference_notional_usd']:,.0f} per underlying. "
        f"{scenarios['note']}"
    )
    if len(scenarios.get("underlyings", [])) > 1:
        lines.append("")
        lines.append(scenarios["cross_symbol_warning"])
    lines.append("")
    lines.append(_scenario_table(scenarios["rows"]))

    lines += ["", "## 7. What would change this, and the method behind it"]
    lines += [f"- {t['detail']}" for t in review["triggers"]]
    lines += [f"- {c}" for c in review["applicable_conditions"]]
    lines += _method_lines(judgment, review, "en")
    return "\n".join(lines)


def _scenario_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| underlying | candidate | structure | cost | at -20% | vs unhedged | "
        "if flat | at +10% |"
    )
    rule = "|---|---|---|---|---|---|---|---|"
    out = []
    for row in rows:
        out.append(
            "| {symbol} | {label} | {structure} | {cost} | {worst} | {protection} | "
            "{flat} | {best} |".format(
                symbol=row.get("symbol") or "—",
                label=row.get("label", "—"),
                structure=row.get("structure", "—"),
                cost=_money(row.get("cost_usd")),
                worst=_money(row.get("pnl_at_worst_move")),
                protection=_money(row.get("protection_at_worst_move")),
                flat=_money(row.get("pnl_if_flat")),
                best=_money(row.get("pnl_at_best_move")),
            )
        )
    return "\n".join([header, rule, *out]) if out else "_nothing priced_"


def _money(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{float(value):,.0f}"


__all__ = [
    "ACCOUNT_FIELDS",
    "ReportInputs",
    "account_fields_present",
    "build_report",
    "render_html",
    "render_markdown",
]


def inputs_of(report: dict[str, Any]) -> dict[str, Any]:
    """The judgment section already carries the exposure the headline needs."""
    return report["2_judgment"]


def _headline(judgment: dict[str, Any], language: str = "en") -> list[str]:
    """The mechanical positions and the level each one turns at.

    Everything else in the report is context for these lines. They were
    previously a clause inside a paragraph in section 2, which is the wrong
    place for the only output a reader can act on directly.

    Neither is a forecast. Both are arithmetic on today's price and volatility
    against a rule frozen in advance, which is exactly why they can be stated
    plainly while the leaning below them cannot.

    A trigger is added when it is known and the exposure is stated without one
    when it is not: an exposure is useful on its own, and suppressing it for
    want of the trigger would lose the more important half.
    """
    exposure = judgment.get("reference_exposure") or {}
    zh = language == "zh"
    body: list[str] = []

    trend = exposure.get("trend_exposure")
    if trend is not None:
        line = f"- **趋势规则 {trend:.0%}**" if zh else f"- **Trend rule {trend:.0%}**"
        spot, trigger = exposure.get("spot"), exposure.get("trend_trigger_price")
        distance = exposure.get("trend_distance_pct")
        if spot and trigger:
            line += (
                f" — SPY {spot:,.2f},200 日均线 {trigger:,.2f}"
                + (f",高出 {distance:+.1%}" if distance is not None else "")
                + ";跌破即转 0"
                if zh
                else f" — SPY at {spot:,.2f} against a 200-session average of {trigger:,.2f}"
                + (f", {distance:+.1%} above it" if distance is not None else "")
                + "; below it the rule goes to 0"
            )
        body.append(line)

    trailing = exposure.get("trailing_exposure")
    if trailing is not None:
        line = (
            f"- **波动率目标 {trailing:.0%}**" if zh
            else f"- **Volatility target {trailing:.0%}**"
        )
        ewma = exposure.get("ewma_exposure")
        if ewma is not None:
            line += f"(EWMA 变体 {ewma:.0%})" if zh else f" (EWMA variant {ewma:.0%})"
        now_vol = exposure.get("volatility_now_annual")
        reduce_above = exposure.get("volatility_reduce_above") or exposure.get(
            "target_volatility_annual"
        )
        if now_vol is not None and reduce_above:
            line += (
                f" — 20 日已实现波动率 {now_vol:.1%},目标 {reduce_above:.0%};"
                "升破目标即按比例降仓"
                if zh
                else f" — 20-session realised volatility is {now_vol:.1%} against a "
                f"{reduce_above:.0%} target; above the target the position scales "
                "down one for one"
            )
        body.append(line)

    if not body:
        # A heading and a caveat with nothing between them is worse than silence.
        return []
    note = (
        "两者都不是预测:它们是把今天的价格和波动率代入事先冻结的规则。"
        "26 年实测账本 年化 6.4% / 最大回撤 −34.4%,是标尺不是建议。"
        if zh
        else "Neither is a forecast: both are today's price and volatility put through a "
        "rule frozen in advance. The 26-year ledger behind them is 6.4% annual return "
        "and a -34.4% maximum drawdown - a yardstick, not advice."
    )
    heading = "参考仓位" if zh else "Reference exposure"
    return [f"## {heading}", "", *body, "", note, ""]


def _frequency_chart(report: dict[str, Any], language: str = "en") -> str:
    """The one comparison that changes a decision, drawn instead of listed.

    Three numbers - the base rate, the rate above the long average, and the rate
    below it - and the whole point is the gap between the last two. As prose
    they read as three similar sentences; as bars the reader sees that the rule
    at the top of the page separates an 11% regime from a 35% one.

    The bar for the condition that holds today is marked, because "which of
    these am I in" is the question a reader brings and the chart should not make
    them work it out.

    Frequencies over settled history. The caption says so, because a bar chart
    of percentages is the easiest place in a report to read a probability that
    was never claimed.
    """
    conditionals = report["2_judgment"].get("conditionals") or {}
    if not conditionals.get("base_rate"):
        return ""
    zh = language == "zh"
    below_now = bool((conditionals.get("current") or {}).get("below_200d_ma"))
    entries = [
        ("base_rate", "所有交易日" if zh else "all sessions", None),
        ("above_200d_ma", "价格在 200 日均线之上" if zh else "price above its 200-session average",
         not below_now),
        ("below_200d_ma", "价格在 200 日均线之下" if zh else "price below its 200-session average",
         below_now),
        ("vol_at_least_current",
         "波动率不低于当前" if zh else "volatility at least as high as now", None),
    ]
    # A window with too little history has no frequency, and drawing a bar for
    # it would put a zero next to four real numbers.
    rows = [
        (label, conditionals[key], here)
        for key, label, here in entries
        if isinstance(conditionals.get(key), dict)
        and conditionals[key].get("event_frequency") is not None
    ]
    if not rows:
        return ""
    widest = max(float(r[1]["event_frequency"]) for r in rows) or 1.0

    bars = []
    for label, block, here in rows:
        share = float(block["event_frequency"])
        tone = "#c0392b" if share >= 0.30 else "#e67e22" if share >= 0.20 else "#2d7d46"
        # Outside _esc, or the entity is escaped into its own literal text.
        mark = ("<b> ← 当前</b>" if zh else "<b> &larr; today</b>") if here else ""
        bars.append(
            f"<tr{' class=here' if here else ''}><td class='n'>{_esc(label)}{mark}</td>"
            f"<td class='bar'><span style='width:{share / widest * 100:.0f}%;"
            f"background:{tone}'></span></td>"
            f"<td class='v'>{share:.0%}</td>"
            f"<td class='d'>n={int(block['sessions']):,}</td></tr>"
        )
    heading = "20 个交易日内回撤 5% 的历史频率" if zh else "How often a 5% drawdown followed within 20 sessions"
    caption = (
        "已结算历史上的发生频率,不是预测概率"
        if zh
        else "frequency over settled history, not a predicted probability"
    )
    return (
        f"<h2>{_esc(heading)}</h2>"
        f"<p class='hint'>{_esc(caption)}</p>"
        "<table class='strip'><tbody>" + "".join(bars) + "</tbody></table>"
    )


def _rank_strip(report: dict[str, Any], language: str = "en") -> str:
    """One bar per indicator, ordered by how risky its level is.

    Twelve rows of numbers do not answer "is anything unusual today" - the eye
    has to read and compare every one. A sorted bar answers it in a glance, and
    it is the same `risk_rank` the table already carries, oriented so that
    longer always means riskier rather than larger.

    No chart library. The page has to open from disk on a machine with no
    network, and a dependency that renders nothing offline is worse than a bar
    made of a div.
    """
    indicators = [
        i for i in report["4_dimensions"]["indicators"]
        if i.get("risk_rank") is not None and not i.get("overlaps")
    ]
    if not indicators:
        return ""
    indicators.sort(key=lambda i: i["risk_rank"], reverse=True)
    heading = "风险分位（越长越危险）" if language == "zh" else "Risk rank (longer is riskier)"
    calm = "历史上更平静的位置" if language == "zh" else "calmer than history"

    rows = []
    for item in indicators:
        rank = float(item["risk_rank"])
        # Colour carries the same number, not a second opinion about it.
        tone = "#c0392b" if rank >= 0.8 else "#e67e22" if rank >= 0.6 else "#7f8c8d"
        rows.append(
            f"<tr><td class='n'>{_esc(item['dimension'])}</td>"
            f"<td class='n'>{_esc(item['name'])}</td>"
            f"<td class='bar'><span style='width:{rank * 100:.0f}%;background:{tone}'></span></td>"
            f"<td class='v'>{rank:.0%}</td>"
            f"<td class='v'>{'' if item['value'] is None else format(item['value'], '.2f')}</td>"
            f"<td class='d'>{_esc(item['direction'])}</td></tr>"
        )
    return (
        f"<h2>{_esc(heading)}</h2>"
        f"<p class='hint'>{_esc(calm)} &larr; &rarr; "
        f"{_esc('历史上更危险的位置' if language == 'zh' else 'riskier than history')}</p>"
        "<table class='strip'><tbody>" + "".join(rows) + "</tbody></table>"
    )


def render_html(
    report: dict[str, Any],
    title: str = "Market state",
    language: str = "en",
) -> str:
    """The same seven sections as a page. Rendered from the report, not the Markdown.

    A small hand-written renderer rather than a Markdown dependency: the subset
    emitted here is headings, lists and two tables, and converting the prose a
    second time would give two documents that could drift apart.
    """
    markdown = render_markdown(report, language)
    body: list[str] = []
    rows: list[str] = []

    def flush_table() -> None:
        if not rows:
            return
        head, *rest = [r for r in rows if not set(r.replace("|", "").strip()) <= {"-", " "}]
        cells = [c.strip() for c in head.strip("|").split("|")]
        body.append("<table><thead><tr>" + "".join(f"<th>{_esc(c)}</th>" for c in cells))
        body.append("</tr></thead><tbody>")
        for line in rest:
            values = [c.strip() for c in line.strip("|").split("|")]
            body.append("<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in values) + "</tr>")
        body.append("</tbody></table>")
        rows.clear()

    strip = _rank_strip(report, language)
    frequency = _frequency_chart(report, language)
    for line in markdown.splitlines():
        if line.startswith("|"):
            rows.append(line)
            continue
        flush_table()
        if line.startswith("# "):
            # Above the prose, because the question a reader arrives with is
            # "is anything unusual today" and every paragraph delays the answer.
            body.append(f"<h1>{_esc(line[2:])}</h1>")
            body.append(frequency)
            body.append(strip)
        elif line.startswith("## "):
            body.append(f"<h2>{_esc(line[3:])}</h2>")
        elif line.startswith("- "):
            body.append(f"<li>{_esc(line[2:])}</li>")
        elif line.strip():
            body.append(f"<p>{_esc(line)}</p>")
    flush_table()

    return (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{_esc(title)}</title>"
        "<style>body{font:14px/1.5 system-ui,sans-serif;max-width:60rem;margin:2rem auto;"
        "padding:0 1rem}table{border-collapse:collapse;width:100%;margin:1rem 0}"
        "th,td{border:1px solid #ddd;padding:.35rem .5rem;text-align:left}"
        "th{background:#f6f6f6}h2{margin-top:2rem;border-bottom:1px solid #eee}"
        "li{margin:.2rem 0}"
        "table.strip td{border:0;padding:.2rem .5rem;white-space:nowrap}"
        "table.strip tr:nth-child(odd){background:#fafafa}"
        "td.bar{width:55%;border:0}"
        "td.bar span{display:block;height:.85rem;border-radius:2px;min-width:2px}"
        "td.v{text-align:right;font-variant-numeric:tabular-nums}"
        "td.n{color:#333}td.d{color:#777;font-size:.85em}"
        "p.hint{color:#888;font-size:.85em;margin:.2rem 0 .6rem}"
        "tr.here td{font-weight:600}tr.here td.n{color:#111}</style>"
        + "".join(body)
    )


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("**", "")
        .replace("`", "")
    )

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
    rows = list(inputs.controls)
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
                    "label", "structure", "contracts", "cost_usd", "cost_pct_of_notional",
                    "pnl_at_worst_move", "protection_at_worst_move", "pnl_if_flat",
                    "pnl_at_best_move", "protected_below", "uncovered_shares",
                    "coverage_ratio", "market_data_type",
                )
            }
            for row in rows
        ],
        "note": (
            "payoff is at expiry only; the reference exposure is a stated yardstick and "
            "not anyone's position"
        ),
    }


def _review(inputs: ReportInputs) -> dict[str, Any]:
    """Section 7. What would change this, and how well tested any of it is."""
    return {
        "triggers": inputs.assessment.triggers[:6],
        "next_review_reason": inputs.assessment.next_review_reason,
        "research_grade": inputs.assessment.research_grade,
        "applicable_conditions": inputs.assessment.applicable_conditions,
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


def render_markdown(report: dict[str, Any]) -> str:
    """A templated rendering. No sentence here is generated from the numbers."""
    status = report["1_data_status"]
    judgment = report["2_judgment"]
    changes = report["3_changes"]
    dimensions = report["4_dimensions"]
    instruments = report["5_instruments"]
    scenarios = report["6_scenarios"]
    review = report["7_review"]

    lines = [
        f"# Market state, session {status['session_date']}",
        "",
        "## 1. Data status",
        f"- snapshot: `{status['snapshot_id']}`",
        f"- evidence as of: {status['evidence_as_of']}",
        f"- approved uses: {', '.join(status['approved_uses']) or 'none granted'}",
        f"- {status['note']}",
    ]
    if status.get("cross_source"):
        cross = status["cross_source"]
        lines.append(
            f"- cross-source: {cross.get('compared_days')} sessions against "
            f"{', '.join(cross.get('sources', []))}, "
            f"{cross.get('days_outside_tolerance')} outside tolerance"
        )

    lines += [
        "",
        "## 2. Market reading",
        f"- state: {', '.join(judgment['state_labels'])}",
        f"- action leaning: **{judgment['action_leaning']}**",
        f"- research grade: {judgment['research_grade']}",
        f"- {judgment['leaning_reason']}",
        "",
        "**Supporting**" if judgment["supporting"] else "**Supporting** — none",
    ]
    lines += [f"- {item}" for item in judgment["supporting"]]
    if judgment["supporting_withheld"]:
        lines.append(f"- ({judgment['supporting_withheld']} further items not shown)")
    lines += ["", "**Against**" if judgment["contrary"] else "**Against** — none"]
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

    lines += ["", "## 4. Dimensions", "", _indicator_table(dimensions["indicators"])]
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
        f"Reference: ${scenarios['reference_notional_usd']:,.0f}. {scenarios['note']}"
    )
    lines.append("")
    lines.append(_scenario_table(scenarios["rows"]))

    lines += ["", "## 7. What would change this"]
    lines += [f"- {t['detail']}" for t in review["triggers"]]
    lines += [f"- {c}" for c in review["applicable_conditions"]]
    lines.append(f"- research grade: {review['research_grade']}")
    return "\n".join(lines)


def _indicator_table(indicators: list[dict[str, Any]]) -> str:
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


def _scenario_table(rows: list[dict[str, Any]]) -> str:
    header = "| candidate | structure | cost | at -20% | vs unhedged | if flat | at +10% |"
    rule = "|---|---|---|---|---|---|---|"
    out = []
    for row in rows:
        out.append(
            "| {label} | {structure} | {cost} | {worst} | {protection} | {flat} | {best} |".format(
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
    "render_markdown",
]

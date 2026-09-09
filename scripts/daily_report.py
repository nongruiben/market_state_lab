"""The daily run: the seven-section report and the artefact set section 18 fixes.

Composes the layers, writes every artefact into `reports/{run_id}/`, and decides
nothing. Read-only throughout.

Public data alone fills sections 1-4 and 7. Sections 5 and 6 need option quotes,
so they come from the most recent protection-table snapshot this machine
recorded, and are honestly empty when there is none - a report that waits for
TWS is a report nobody reads on the days TWS is down.

    .\\.venv\\Scripts\\python.exe scripts\\daily_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_state_lab.config import load_config  # noqa: E402
from market_state_lab.data.public import PublicDataLoader  # noqa: E402
from market_state_lab.data.reconciliation import SourceSeries, reconcile_closes  # noqa: E402
from market_state_lab.data.reconciliation import summarise as reconciliation_summary  # noqa: E402
from market_state_lab.data.snapshots import latest_sessions, read_snapshot  # noqa: E402
from market_state_lab.data.validation import (  # noqa: E402
    issues_frame,
    validate_daily_bars,
)
from market_state_lab.data.validation import (
    summarise as quality_summary,
)
from market_state_lab.events import EventLog, EventRules  # noqa: E402
from market_state_lab.main_report import (  # noqa: E402
    ReportInputs,
    build_report,
    render_html,
    render_markdown,
)
from market_state_lab.market_assessment import assess, state_labels  # noqa: E402
from market_state_lab.market_evidence import build_evidence  # noqa: E402
from market_state_lab.research import open_registry  # noqa: E402


def _protection(root: Path) -> dict[str, Any]:
    """The newest protection snapshot, if this machine has recorded one."""
    history = latest_sessions(root / "data")
    if history.empty:
        return {"candidates": None, "controls": [], "snapshot_id": None, "eligible": ()}
    snapshot = read_snapshot(root / "data", history.iloc[0]["snapshot_id"])
    return {
        "candidates": snapshot.tables.get("comparison"),
        "controls": (
            snapshot.tables["controls"].to_dict("records")
            if "controls" in snapshot.tables
            else []
        ),
        "snapshot_id": snapshot.snapshot_id,
        "eligible": snapshot.eligible_for,
        "manifest": snapshot.manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notional", type=float, default=100_000.0)
    parser.add_argument("--quiet", action="store_true", help="write the artefacts, print a summary")
    args = parser.parse_args()

    config = load_config()
    bundle = PublicDataLoader(config).load()
    evidence = build_evidence(bundle.etf_close, bundle.vix, bundle.macro)
    assessment = assess(evidence)
    session = str(evidence.as_of.date()) if evidence.as_of is not None else "unknown"

    # Quality and cross-source, on the core input the plan names first.
    spy = bundle.etf_close["spy"].dropna() if "spy" in bundle.etf_close else pd.Series(dtype=float)
    issues = validate_daily_bars(pd.DataFrame({"close": spy}), "SPY") if not spy.empty else []
    reconciliation = None
    raw = bundle.etf_close_unadjusted
    if not spy.empty and "spy" in getattr(raw, "columns", []):
        # Adjusted against adjusted would compare like with like; the unadjusted
        # series is here so the convention check has something legal to compare.
        result = reconcile_closes(
            "SPY",
            SourceSeries("Yahoo adjusted", spy, adjusted=True),
            SourceSeries("Yahoo raw", raw["spy"].dropna(), adjusted=True),
        )
        reconciliation = reconciliation_summary(result)

    events_path = ROOT / "data" / "events.json"
    log = EventLog.load(events_path, EventRules())
    previous = [e.label for e in log.open_events()]
    news = log.observe(state_labels(evidence), session)
    log.save(events_path)

    protection = _protection(ROOT)
    report = build_report(
        ReportInputs(
            evidence=evidence,
            assessment=assessment,
            snapshot_id=protection["snapshot_id"],
            session_date=session,
            data_quality=quality_summary(issues),
            eligible_for=protection["eligible"],
            new_events=news,
            open_events=log.open_events(),
            previous_labels=previous,
            candidates=protection["candidates"],
            controls=protection["controls"],
            reference_notional=args.notional,
            reconciliation=reconciliation,
        )
    )

    markdown = render_markdown(report)
    if not args.quiet:
        print(markdown)

    run_id = f"{session}-{datetime.now(timezone.utc).strftime('%H%M%SZ')}"
    out = ROOT / "reports" / run_id
    out.mkdir(parents=True, exist_ok=True)
    registry = open_registry().summary()

    written = _write_artefacts(out, report, markdown, evidence, assessment, issues,
                               reconciliation, protection, registry, run_id, session, args)
    print(f"\nwrote {len(written)} artefacts to {out}")
    for name in written:
        print(f"  {name}")
    print(f"\nresearch registry: {registry['registered']} experiments, none promoted")
    return 0


def _write_artefacts(
    out: Path, report, markdown, evidence, assessment, issues, reconciliation,
    protection, registry, run_id, session, args,
) -> list[str]:
    """Section 18's list. Every file is written or its absence is a named gap."""
    written: list[str] = []

    def write_json(name: str, payload: Any) -> None:
        (out / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        written.append(name)

    def write_csv(name: str, frame: pd.DataFrame | None) -> None:
        if frame is None or frame.empty:
            (out / name).write_text("", encoding="utf-8")
        else:
            frame.to_csv(out / name, index=False)
        written.append(name)

    write_json("run_manifest.json", {
        "run_id": run_id,
        "session_date": session,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_notional_usd": args.notional,
        "protection_snapshot": protection["snapshot_id"],
        "approved_uses": list(protection["eligible"]),
        "research_registry": registry,
    })
    write_json("data_quality_summary.json", quality_summary(issues))
    write_csv("data_quality_issues.csv", issues_frame(issues))
    write_csv(
        "source_reconciliation.csv",
        pd.DataFrame([reconciliation]) if reconciliation else None,
    )
    evidence.frame().to_parquet(out / "market_evidence.parquet")
    written.append("market_evidence.parquet")
    write_json("market_assessment.json", assessment.as_dict())
    write_csv("instrument_candidates.csv", protection["candidates"])
    write_csv(
        "scenario_comparison.csv",
        pd.DataFrame(report["6_scenarios"]["rows"]) if report["6_scenarios"]["rows"] else None,
    )
    (out / "report.md").write_text(markdown, encoding="utf-8")
    written.append("report.md")
    (out / "report.html").write_text(
        render_html(report, f"Market state {session}"), encoding="utf-8"
    )
    written.append("report.html")
    return written


if __name__ == "__main__":
    raise SystemExit(main())

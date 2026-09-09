"""The daily report: data status, market reading, instruments, payoff, review.

Composes the layers into the seven sections of the plan's section 13 and writes
them as Markdown and JSON. Read-only throughout, and it decides nothing.

Public data alone is enough to produce sections 1-4 and 7. Sections 5 and 6 need
option quotes, so they are filled from the most recent protection-table snapshot
if one exists and are honestly empty if not - a report that waits for TWS is a
report nobody reads on the days TWS is down.

    .\.venv\Scripts\python.exe scripts\daily_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_state_lab.config import load_config  # noqa: E402
from market_state_lab.data.public import PublicDataLoader  # noqa: E402
from market_state_lab.data.snapshots import latest_sessions, read_snapshot  # noqa: E402
from market_state_lab.events import EventLog, EventRules  # noqa: E402
from market_state_lab.main_report import (  # noqa: E402
    ReportInputs,
    build_report,
    render_markdown,
)
from market_state_lab.market_assessment import assess, state_labels  # noqa: E402
from market_state_lab.market_evidence import build_evidence  # noqa: E402
from market_state_lab.research import open_registry  # noqa: E402


def _protection(root: Path) -> tuple[pd.DataFrame | None, list[dict], str | None, tuple]:
    """The newest protection snapshot, if this machine has recorded one."""
    history = latest_sessions(root / "data")
    if history.empty:
        return None, [], None, ()
    newest = history.iloc[0]
    snapshot = read_snapshot(root / "data", newest["snapshot_id"])
    candidates = snapshot.tables.get("comparison")
    controls = (
        snapshot.tables["controls"].to_dict("records")
        if "controls" in snapshot.tables
        else []
    )
    return candidates, controls, snapshot.snapshot_id, snapshot.eligible_for


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notional", type=float, default=100_000.0)
    args = parser.parse_args()

    config = load_config()
    bundle = PublicDataLoader(config).load()
    evidence = build_evidence(bundle.etf_close, bundle.vix, bundle.macro)
    assessment = assess(evidence)

    session = str(evidence.as_of.date()) if evidence.as_of is not None else "unknown"
    events_path = ROOT / "data" / "events.json"
    log = EventLog.load(events_path, EventRules())
    previous = [e.label for e in log.open_events()]
    news = log.observe(state_labels(evidence), session)
    log.save(events_path)

    candidates, controls, snapshot_id, eligible = _protection(ROOT)

    report = build_report(
        ReportInputs(
            evidence=evidence,
            assessment=assessment,
            snapshot_id=snapshot_id,
            session_date=session,
            eligible_for=eligible,
            new_events=news,
            open_events=log.open_events(),
            previous_labels=previous,
            candidates=candidates,
            controls=controls,
            reference_notional=args.notional,
        )
    )
    markdown = render_markdown(report)
    print(markdown)

    registry = open_registry().summary()
    print()
    print(f"research registry: {registry['registered']} experiments, none promoted")
    print(f"  {registry['note']}")

    out = ROOT / "reports" / f"daily_report_{session}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(markdown, encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(
            {"built_at_utc": datetime.now(timezone.utc).isoformat(),
             "report": report, "research_registry": registry},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

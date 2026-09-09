"""One run: load, gate, describe, compare, and write the artefact set.

This is the single entry point. `cli run` and `scripts/daily_report.py` both
call it, because section 18 defines one run as producing one set of artefacts
and two paths would eventually produce two answers.

The forecasting layer that used to sit in the middle of this function is gone.
It lost to a two-line persistence rule, its drawdown edge disappeared under an
exposure-matched control, and its conclusions now live in `research.py` as
registered findings rather than as code nobody should run. What replaced it
describes rather than predicts, and the benchmark ledger it used to be scored
against survives in `evaluation.py` - that machinery is the reason the failure
was findable, so it outlived the thing it measured.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from market_state_lab.config import ensure_runtime_directories, project_path
from market_state_lab.data.fixtures import load_offline_fixture
from market_state_lab.data.health import evaluate_manifest, required_health_failures
from market_state_lab.data.public import PublicDataBundle, PublicDataLoader
from market_state_lab.data.reconciliation import unverifiable
from market_state_lab.data.snapshots import latest_sessions, read_snapshot
from market_state_lab.data.validation import issues_frame, validate_daily_bars
from market_state_lab.data.validation import summarise as quality_summary
from market_state_lab.evaluation import compare_against_benchmarks
from market_state_lab.events import EventLog, EventRules
from market_state_lab.main_report import ReportInputs, build_report, render_html, render_markdown
from market_state_lab.market_assessment import assess, state_labels
from market_state_lab.market_evidence import build_evidence
from market_state_lab.research import open_registry
from market_state_lab.timeutils import completed_market_clock


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    if frame is not None and not frame.empty:
        frame.to_parquet(path)


def _eligible_bundle(bundle: PublicDataBundle, manifest: pd.DataFrame) -> PublicDataBundle:
    """Drop the sources health rejected, rather than letting them through quietly."""
    rejected = set(
        manifest.loc[manifest["health_status"].ne("ok"), "dataset"].astype(str)
    )
    if not rejected:
        return bundle
    trimmed = {}
    for name in ("macro", "vix", "ofr", "french", "etf_close", "etf_close_unadjusted"):
        frame = getattr(bundle, name, pd.DataFrame())
        if frame is None or frame.empty:
            trimmed[name] = frame
            continue
        keep = [c for c in frame.columns if c not in rejected]
        trimmed[name] = frame[keep]
    return PublicDataBundle(
        trimmed["macro"], trimmed["vix"], trimmed["ofr"], trimmed["french"],
        trimmed["etf_close"], bundle.manifest, bundle.point_in_time_status,
        etf_close_unadjusted=trimmed["etf_close_unadjusted"],
    )


def _as_of(bundle: PublicDataBundle, session: Any) -> PublicDataBundle:
    """Trim every frame to the settled session.

    A frame carrying a row after the session being described would let the
    description use a day that had not finished, which is the same lookahead as
    a forward-looking label and just as invisible in the output.
    """
    cutoff = pd.Timestamp(session)
    trimmed = {}
    for name in ("macro", "vix", "ofr", "french", "etf_close", "etf_close_unadjusted"):
        frame = getattr(bundle, name, pd.DataFrame())
        trimmed[name] = frame.loc[frame.index <= cutoff] if frame is not None and not frame.empty else frame
    return PublicDataBundle(
        trimmed["macro"], trimmed["vix"], trimmed["ofr"], trimmed["french"],
        trimmed["etf_close"], bundle.manifest, bundle.point_in_time_status,
        etf_close_unadjusted=trimmed["etf_close_unadjusted"],
    )


# The dimensions a description cannot be made without. SPY drives both, so an
# absence here means the core input failed rather than that a corner of the
# market is quiet.
REQUIRED_DIMENSIONS = ("trend", "volatility")


def require_dimensions(evidence, required: tuple[str, ...] = REQUIRED_DIMENSIONS) -> None:
    """Fail loudly when a required dimension has no value at all.

    The guarantee the old feature-coverage gate protected, moved to what the
    description actually consumes. Its predecessor failed open: a rule was
    evaluated as passing while the feature it read was entirely absent, and the
    report said nothing. "Unknown" and "calm" are different statements, and only
    one of them may be produced silently.
    """
    missing = [d for d in required if d in evidence.missing_dimensions]
    if missing:
        raise RuntimeError(
            f"required evidence dimensions are absent, not calm: {', '.join(missing)}"
        )


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
            if "controls" in snapshot.tables else []
        ),
        "snapshot_id": snapshot.snapshot_id,
        "eligible": snapshot.eligible_for,
    }


def run_pipeline(
    config: dict[str, Any],
    with_ibkr: bool = False,
    offline: bool = False,
    notional: float = 100_000.0,
) -> dict[str, Path]:
    config = deepcopy(config)
    ensure_runtime_directories(config)
    root = Path(project_path(config, "."))
    processed = project_path(config, "data", "processed")
    reports = project_path(config, "reports")

    if offline:
        # A fixture run used to write over the live report, so reports/ could be
        # holding synthetic numbers with nothing in the filename to say so.
        reports = reports / "offline"
        processed = processed / "offline"
        reports.mkdir(parents=True, exist_ok=True)
        processed.mkdir(parents=True, exist_ok=True)
        bundle = load_offline_fixture(config)
        fixture_end = max(
            frame.index.max()
            for frame in (bundle.macro, bundle.vix, bundle.ofr, bundle.french, bundle.etf_close)
            if not frame.empty
        )
        clock = completed_market_clock(
            config, now=pd.Timestamp(fixture_end, tz="UTC") + pd.Timedelta(hours=23)
        )
    else:
        clock = completed_market_clock(config)
        bundle = PublicDataLoader(config).load()
    config["_runtime"] = clock.as_dict()

    manifest = evaluate_manifest(bundle.manifest, config, clock.market_session)
    failures = required_health_failures(manifest)
    if not failures.empty:
        details = ", ".join(
            f"{row.dataset}:{row.health_status}" for row in failures.itertuples(index=False)
        )
        raise RuntimeError(f"Required data health checks failed: {details}")
    bundle = _as_of(_eligible_bundle(bundle, manifest), clock.market_session)

    evidence = build_evidence(bundle.etf_close, bundle.vix, bundle.macro)
    require_dimensions(evidence)
    assessment = assess(evidence)
    session = str(evidence.as_of.date()) if evidence.as_of is not None else clock.market_session

    spy = bundle.etf_close["spy"].dropna() if "spy" in bundle.etf_close else pd.Series(dtype=float)
    issues = validate_daily_bars(pd.DataFrame({"close": spy}), "SPY") if not spy.empty else []
    # No second source on this path, and saying so is the only honest option.
    # One vendor's adjusted series against its own raw series is one source in
    # two conventions, not corroboration - and making it pass would take
    # declaring the raw series adjusted, which is the false declaration that
    # already cost this project a cross-source check once. Real verification
    # lives in scripts/reconcile_spy.py, where TWS is a genuinely separate feed.
    reconciliation = {
        "symbol": "SPY",
        "sources": ["public data only"],
        "compared_days": 0,
        "agreed": None,
        "note": unverifiable(
            "SPY daily closes",
            "this run used one vendor; run scripts/reconcile_spy.py with TWS up for "
            "an independent second source",
        ).detail,
    }
    benchmarks = compare_against_benchmarks(spy) if len(spy) > 300 else pd.DataFrame()

    events_path = root / "data" / ("events_offline.json" if offline else "events.json")
    log = EventLog.load(events_path, EventRules())
    previous = [e.label for e in log.open_events()]
    news = log.observe(state_labels(evidence), session)
    log.save(events_path)

    protection = _protection(root)
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
            reference_notional=notional,
            reconciliation=reconciliation,
        )
    )

    if with_ibkr:
        manifest = _append_ibkr_status(config, manifest, clock, reports)

    _write_frame(bundle.macro, processed / "macro.parquet")
    _write_frame(bundle.vix, processed / "vix.parquet")
    _write_frame(bundle.etf_close, processed / "etf_close.parquet")

    run_id = f"{session}-{datetime.now(timezone.utc).strftime('%H%M%SZ')}"
    out = reports / run_id
    out.mkdir(parents=True, exist_ok=True)
    registry = open_registry().summary()

    (out / "run_manifest.json").write_text(json.dumps({
        "run_id": run_id,
        "session_date": session,
        "offline": offline,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_notional_usd": notional,
        "protection_snapshot": protection["snapshot_id"],
        "approved_uses": list(protection["eligible"]),
        "research_registry": registry,
        "runtime": clock.as_dict(),
    }, indent=2, default=str), encoding="utf-8")
    (out / "data_quality_summary.json").write_text(
        json.dumps(quality_summary(issues), indent=2, default=str), encoding="utf-8"
    )
    issues_frame(issues).to_csv(out / "data_quality_issues.csv", index=False)
    pd.DataFrame([reconciliation] if reconciliation else []).to_csv(
        out / "source_reconciliation.csv", index=False
    )
    benchmarks.to_csv(out / "benchmark_ledger.csv", index=False)
    evidence.frame().to_parquet(out / "market_evidence.parquet")
    (out / "market_assessment.json").write_text(
        json.dumps(assessment.as_dict(), indent=2, default=str), encoding="utf-8"
    )
    candidates = protection["candidates"]
    (candidates if candidates is not None else pd.DataFrame()).to_csv(
        out / "instrument_candidates.csv", index=False
    )
    pd.DataFrame(report["6_scenarios"]["rows"]).to_csv(
        out / "scenario_comparison.csv", index=False
    )
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (out / "report.html").write_text(
        render_html(report, f"Market state {session}"), encoding="utf-8"
    )
    manifest.to_csv(out / "data_manifest.csv", index=False)

    return {name: out / name for name in sorted(p.name for p in out.iterdir())}


def _append_ibkr_status(config, manifest, clock, reports) -> pd.DataFrame:
    """A snapshot fetched after everything above, so it cannot reach an output."""
    from market_state_lab.data.ibkr import ReadOnlyIBKRClient

    symbols = [str(symbol) for symbol in config["ibkr"]["snapshot_symbols"]]
    try:
        with ReadOnlyIBKRClient(config) as client:
            contracts = [client.qualify_stock(symbol) for symbol in symbols]
            snapshot = client.quotes(contracts)
            snapshot.to_csv(reports / "ibkr_snapshot.csv", index=False)
            status = {"status": "success", "rows": len(snapshot), "error": ""}
    except Exception as exc:
        status = {"status": "failed", "rows": 0, "error": f"{type(exc).__name__}: {exc}"}
    row = pd.DataFrame([{
        "dataset": "ibkr_quotes",
        "provider": "IBKR TWS read-only",
        "earliest_date": "",
        "latest_date": pd.Timestamp.utcnow().date().isoformat(),
        "vintage_mode": "snapshot",
        **status,
    }])
    return evaluate_manifest(
        pd.concat([manifest, row], ignore_index=True), config, clock.market_session
    )

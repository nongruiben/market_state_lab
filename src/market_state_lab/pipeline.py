from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from market_state_lab.config import ensure_runtime_directories, project_path
from market_state_lab.data.ibkr import ReadOnlyIBKRClient
from market_state_lab.data.public import PublicDataLoader
from market_state_lab.features import build_features
from market_state_lab.models import fit_market_state, fit_style
from market_state_lab.reporting import write_dashboard


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    if not frame.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)


def _add_freshness(manifest: pd.DataFrame) -> pd.DataFrame:
    result = manifest.copy()
    if result.empty:
        return result
    latest = pd.to_datetime(result["latest_date"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    result["age_calendar_days"] = (today - latest).dt.days
    result["usable"] = result["status"].ne("failed") & result["rows"].gt(0)
    return result


def run_pipeline(
    config: dict[str, Any],
    with_ibkr: bool = False,
) -> dict[str, Path]:
    ensure_runtime_directories(config)
    bundle = PublicDataLoader(config).load()
    features = build_features(bundle, config)
    state = fit_market_state(features.market, config)
    style = fit_style(features.style_returns, config)

    processed = project_path(config, "data", "processed")
    reports = project_path(config, "reports")
    _write_frame(bundle.macro, processed / "macro.parquet")
    _write_frame(bundle.vix, processed / "vix.parquet")
    _write_frame(bundle.ofr, processed / "ofr_fsi.parquet")
    _write_frame(bundle.french, processed / "french_factors.parquet")
    _write_frame(bundle.etf_close, processed / "etf_close.parquet")
    _write_frame(features.market, processed / "market_features.parquet")
    _write_frame(features.style_returns, processed / "style_returns.parquet")
    _write_frame(state.history, reports / "market_state_history.parquet")
    _write_frame(style.history, reports / "style_history.parquet")

    manifest = _add_freshness(bundle.manifest)
    if with_ibkr:
        symbols = [str(symbol) for symbol in config["ibkr"]["snapshot_symbols"]]
        try:
            with ReadOnlyIBKRClient(config) as client:
                snapshot = client.delayed_snapshots(symbols)
                snapshot.to_csv(reports / "ibkr_snapshot.csv")
                ibkr_status = pd.DataFrame(
                    [{
                        "dataset": "delayed_snapshots",
                        "provider": "IBKR TWS read-only",
                        "status": "success",
                        "rows": len(snapshot),
                        "earliest_date": "",
                        "latest_date": pd.Timestamp.utcnow().date().isoformat(),
                        "error": "",
                    }]
                )
        except Exception as exc:
            ibkr_status = pd.DataFrame(
                [{
                    "dataset": "delayed_snapshots",
                    "provider": "IBKR TWS read-only",
                    "status": "failed",
                    "rows": 0,
                    "earliest_date": "",
                    "latest_date": "",
                    "error": str(exc),
                }]
            )
        manifest = _add_freshness(pd.concat([manifest, ibkr_status], ignore_index=True))

    manifest.to_csv(reports / "data_manifest.csv", index=False)
    (reports / "latest_market_state.json").write_text(
        json.dumps(state.latest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    style.latest.to_csv(reports / "latest_style_state.csv", index=False)
    write_dashboard(reports / "market_state_dashboard.html", state, style, manifest, config)
    return {
        "dashboard": reports / "market_state_dashboard.html",
        "market_state": reports / "latest_market_state.json",
        "style_state": reports / "latest_style_state.csv",
        "manifest": reports / "data_manifest.csv",
    }


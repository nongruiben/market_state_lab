from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_state_lab.config import ensure_runtime_directories, project_path
from market_state_lab.data.fixtures import load_offline_fixture
from market_state_lab.data.health import (
    eligible_datasets,
    evaluate_manifest,
    required_health_failures,
)
from market_state_lab.data.ibkr import ReadOnlyIBKRClient
from market_state_lab.data.public import PublicDataBundle, PublicDataLoader
from market_state_lab.features import build_features
from market_state_lab.models import fit_market_state, fit_style
from market_state_lab.news import run_news_pipeline
from market_state_lab.news.evaluation import evaluate_news_forward
from market_state_lab.reporting import write_dashboard
from market_state_lab.timeutils import completed_market_clock


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    if not frame.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)


def _feature_coverage(features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Per-feature non-NaN coverage over the modelled panel.

    The manifest checks whether a *source* arrived; nothing checked whether the
    *feature* built from it actually exists. That gap is how hy_oas sat at 11%
    coverage and downside_volatility_60 at 19% while every health row said ok.
    """
    settings = config["features"].get("coverage", {}) or {}
    default_minimum = float(settings.get("default_min_coverage", 0.0))
    rules = settings.get("rules", {}) or {}
    total = len(features)
    rows: list[dict[str, Any]] = []
    for column in features.columns:
        rule = rules.get(str(column), {}) or {}
        present = int(features[column].count())
        coverage = present / total if total else float("nan")
        minimum = float(rule.get("min_coverage", default_minimum))
        first_valid = features[column].first_valid_index()
        rows.append(
            {
                "feature": str(column),
                "rows": total,
                "observations": present,
                "coverage": coverage,
                "min_coverage": minimum,
                "required": bool(rule.get("required", False)),
                "first_valid_date": "" if first_valid is None else str(first_valid.date()),
                "status": "ok" if coverage >= minimum else "below_threshold",
            }
        )
    return pd.DataFrame(rows).sort_values("coverage").reset_index(drop=True)


def _eligible_bundle(bundle: PublicDataBundle, manifest: pd.DataFrame) -> PublicDataBundle:
    eligible = eligible_datasets(manifest)
    macro = bundle.macro[[column for column in bundle.macro if column in eligible]]
    # The extra CBOE indices ride inside bundle.vix but are separate manifest
    # datasets, so gating the whole frame on the "vix" row alone would let a
    # source that failed its own health check in through the side door.
    if "vix" in eligible:
        vix_columns = [column for column in bundle.vix if str(column).startswith("vix_")]
        vix_columns.extend(column for column in bundle.vix if str(column) in eligible)
        vix = bundle.vix[list(dict.fromkeys(vix_columns))]
    else:
        vix = pd.DataFrame()
    ofr = bundle.ofr if "financial_stress_index" in eligible else pd.DataFrame()
    french_columns: list[str] = []
    if "ff5" in eligible:
        french_columns.extend(column for column in ("mkt_rf", "smb", "hml", "rmw", "cma", "rf") if column in bundle.french)
    for dataset in ("momentum", "short_reversal", "long_reversal"):
        if dataset in eligible and dataset in bundle.french:
            french_columns.append(dataset)
    french = bundle.french[french_columns] if french_columns else pd.DataFrame()
    etf = bundle.etf_close[[column for column in bundle.etf_close if column in eligible]]
    return PublicDataBundle(
        macro=macro,
        vix=vix,
        ofr=ofr,
        french=french,
        etf_close=etf,
        manifest=manifest,
        point_in_time_status=bundle.point_in_time_status,
    )


def run_pipeline(
    config: dict[str, Any],
    with_ibkr: bool = False,
    offline: bool = False,
) -> dict[str, Path]:
    config = deepcopy(config)
    ensure_runtime_directories(config)
    processed = project_path(config, "data", "processed")
    reports = project_path(config, "reports")
    if offline:
        # A fixture run used to write over the live dashboard, so reports/ could
        # be holding synthetic numbers with nothing in the filename to say so.
        # Keep the two worlds in separate directories.
        reports = reports / "offline"
        processed = processed / "offline"
        reports.mkdir(parents=True, exist_ok=True)
        processed.mkdir(parents=True, exist_ok=True)
    if offline:
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
    config["_runtime"] = clock.as_dict()
    if not offline:
        bundle = PublicDataLoader(config).load()
    manifest = evaluate_manifest(bundle.manifest, config, clock.market_session)
    failures = required_health_failures(manifest)
    if not failures.empty:
        details = ", ".join(
            f"{row.dataset}:{row.health_status}" for row in failures.itertuples(index=False)
        )
        raise RuntimeError(f"Required data health checks failed: {details}")
    bundle = _eligible_bundle(bundle, manifest)
    features = build_features(bundle, config, as_of=clock.market_session)
    coverage = _feature_coverage(features.market, config)
    coverage.to_csv(reports / "feature_coverage.csv", index=False)
    short = coverage.loc[coverage["required"] & coverage["status"].eq("below_threshold")]
    if not short.empty:
        details = ", ".join(
            f"{row.feature}:{row.coverage:.1%}<{row.min_coverage:.0%}"
            for row in short.itertuples(index=False)
        )
        raise RuntimeError(f"Required feature coverage below threshold: {details}")
    state = fit_market_state(features.market, config)
    style = fit_style(features.style_returns, config)

    uses_latest_macro = not offline and str(config["data"]["fred"].get("vintage_mode", "latest")) != "point_in_time"
    uses_latest_french = not offline and not bundle.french.empty
    history_is_latest_vintage = uses_latest_macro or uses_latest_french
    if bool(config["data"].get("strict_history", False)) and history_is_latest_vintage:
        raise RuntimeError(
            "Strict historical mode rejected latest-vintage FRED/French data. "
            "Enable ALFRED and disable revision-prone French history."
        )
    state.latest.update(
        {
            "run_clock": clock.as_dict(),
            "information_date": clock.market_session,
            "run_date": clock.run_date_local,
            "history_is_latest_vintage": history_is_latest_vintage,
            "historical_backtest_eligible": not history_is_latest_vintage,
            "point_in_time_status": bundle.point_in_time_status,
        }
    )
    if bool(config.get("news", {}).get("enabled", False)):
        try:
            news = run_news_pipeline(
                config,
                fetch=bool(config["news"].get("fetch_on_run", False)),
                use_llm=bool(config["news"].get("llm_enabled", True)),
            )
            news_evaluation = evaluate_news_forward(
                news.daily_features, features.market["market_return"]
            )
            if not news_evaluation.empty:
                news_evaluation.to_csv(reports / "news_forward_evaluation.csv", index=False)
            if (
                news.quality.get("status") in {"available", "session_gap_fallback"}
                and news.quality.get("snapshot_fresh", False)
            ):
                latest_news = news.daily_features.iloc[-1].replace({np.nan: None}).to_dict()
                state.latest["news_overlay"] = {
                    "status": news.quality["status"],
                    "as_of": news.daily_features.index[-1].date().isoformat(),
                    "features": latest_news,
                    "quality": news.quality,
                    "llm": news.metadata,
                    "used_in_state_model": False,
                }
            else:
                state.latest["news_overlay"] = {
                    "status": news.quality.get("status", "stale"),
                    "quality": news.quality,
                    "llm": news.metadata,
                    "used_in_state_model": False,
                }
        except Exception as exc:
            state.latest["news_overlay"] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                "used_in_state_model": False,
            }

    _write_frame(bundle.macro, processed / "macro.parquet")
    _write_frame(bundle.vix, processed / "vix.parquet")
    _write_frame(bundle.ofr, processed / "ofr_fsi.parquet")
    _write_frame(bundle.french, processed / "french_factors.parquet")
    _write_frame(bundle.etf_close, processed / "etf_close.parquet")
    _write_frame(features.market, processed / "market_features.parquet")
    _write_frame(features.style_returns, processed / "style_returns.parquet")
    _write_frame(state.history, reports / "market_state_history.parquet")
    _write_frame(style.history, reports / "style_history.parquet")
    state.diagnostics.to_csv(reports / "model_refit_diagnostics.csv", index=False)
    state.comparison.to_csv(reports / "model_comparison.csv", index=False)
    state.decision_value.to_csv(reports / "decision_value_comparison.csv", index=False)
    state.exposure_tradeoff.to_csv(reports / "exposure_tradeoff.csv", index=False)

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
                        "vintage_mode": "snapshot",
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
                    "vintage_mode": "snapshot",
                    "error": str(exc),
                }]
            )
        manifest = evaluate_manifest(
            pd.concat([manifest, ibkr_status], ignore_index=True), config, clock.market_session
        )

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
        "model_comparison": reports / "model_comparison.csv",
        "model_diagnostics": reports / "model_refit_diagnostics.csv",
        "decision_value": reports / "decision_value_comparison.csv",
        "exposure_tradeoff": reports / "exposure_tradeoff.csv",
        "feature_coverage": reports / "feature_coverage.csv",
    }

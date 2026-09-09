from __future__ import annotations

import pandas as pd
import pytest

from market_state_lab.config import load_config
from market_state_lab.data.health import evaluate_manifest, required_health_failures
from market_state_lab.pipeline import _feature_coverage


def test_required_source_accepts_healthy_fallback_provider() -> None:
    config = load_config()
    manifest = pd.DataFrame(
        [
            {"dataset": "spy", "provider": "Stooq", "status": "failed", "rows": 0, "latest_date": ""},
            {"dataset": "spy", "provider": "Nasdaq", "status": "success", "rows": 10, "latest_date": "2026-08-27"},
        ]
    )
    evaluated = evaluate_manifest(manifest, config, "2026-08-27")
    assert required_health_failures(evaluated).empty


def test_fresh_but_truncated_source_is_rejected() -> None:
    """The exact silent failure this check exists for.

    FRED's graph CSV serves only a rolling ~3-year window for the licensed ICE
    BofA spreads, so hy_oas arrived on time every day, reported ok, and quietly
    took 89% of its own history with it.
    """
    config = load_config()
    manifest = pd.DataFrame(
        [
            {
                "dataset": "hy_oas",
                "provider": "FRED graph CSV",
                "status": "cache_fresh",
                "rows": 795,
                "earliest_date": "2023-08-29",
                "latest_date": "2026-08-27",
            },
            {
                "dataset": "yield_curve_10y2y",
                "provider": "FRED graph CSV",
                "status": "cache_fresh",
                "rows": 6955,
                "earliest_date": "2000-01-03",
                "latest_date": "2026-08-27",
            },
        ]
    )
    evaluated = evaluate_manifest(manifest, config, "2026-08-27")
    truncated = evaluated.set_index("dataset").loc["hy_oas"]
    complete = evaluated.set_index("dataset").loc["yield_curve_10y2y"]
    assert truncated["observation_freshness"] == "fresh"
    assert truncated["history_coverage"] == "truncated"
    assert truncated["health_status"] == "truncated"
    assert not truncated["model_eligible"]
    assert complete["health_status"] == "ok"
    assert complete["model_eligible"]


def test_offline_fixture_is_exempt_from_history_depth() -> None:
    config = load_config()
    manifest = pd.DataFrame(
        [
            {
                "dataset": "spy",
                "provider": "offline synthetic fixture",
                "status": "fixture",
                "rows": 1474,
                "earliest_date": "2021-01-04",
                "latest_date": "2026-08-27",
            }
        ]
    )
    evaluated = evaluate_manifest(manifest, config, "2026-08-27")
    assert evaluated.loc[0, "model_eligible"]
    assert required_health_failures(evaluated).empty


def test_feature_coverage_flags_a_required_gap() -> None:
    index = pd.date_range("2020-01-01", periods=200, freq="B")
    features = pd.DataFrame(
        {
            "market_return": 0.001,
            "vix_close": 15.0,
            "macro_hy_oas": [float("nan")] * 180 + [3.0] * 20,
        },
        index=index,
    )
    config = load_config()
    coverage = _feature_coverage(features, config).set_index("feature")
    assert coverage.loc["macro_hy_oas", "status"] == "below_threshold"
    assert coverage.loc["macro_hy_oas", "coverage"] == pytest.approx(0.10)
    assert not coverage.loc["macro_hy_oas", "required"]
    assert coverage.loc["market_return", "status"] == "ok"
    assert coverage.loc["vix_close", "status"] == "ok"

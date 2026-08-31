from __future__ import annotations

import json

import pandas as pd

from market_state_lab.config import load_config
from market_state_lab.data.health import evaluate_manifest, required_health_failures
from market_state_lab.news.pipeline import _cluster_records, _normalize_records, _validate_events


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


def test_news_window_guard_clustering_and_evidence_validation() -> None:
    payload = {
        "generated_at": "2026-08-27T12:00:00+00:00",
        "records": [
            {
                "provider_code": "DJ",
                "article_id": "1",
                "timestamp": "2026-08-27T10:00:00+00:00",
                "headline": "Central bank raises rates by 25 bps",
                "impact_score": 90,
            },
            {
                "provider_code": "BZ",
                "article_id": "2",
                "timestamp": "2026-08-27T10:05:00+00:00",
                "headline": "Central bank raises rates 25 bps",
                "impact_score": 70,
            },
            {
                "provider_code": "OLD",
                "article_id": "3",
                "timestamp": "2026-08-25T10:00:00+00:00",
                "headline": "Old story outside requested range",
            },
        ],
    }
    records, counters, _ = _normalize_records(payload, 24)
    assert len(records) == 2
    assert counters["outside_window"] == 1
    clusters = _cluster_records(records, 0.40)
    assert len(clusters) == 1
    cluster = clusters[0]
    response = {
        "events": [
            {
                "cluster_id": cluster["cluster_id"],
                "theme": "central_banks_rates",
                "summary": "Rate increase confirmed by two sources.",
                "event_status": "confirmed",
                "risk_direction": "risk_off",
                "systemic_relevance": 0.8,
                "urgency": 0.9,
                "novelty": 0.7,
                "llm_confidence": 0.9,
                "signals": {"monetary_hawkishness": 1.5},
                "evidence_ids": [item["evidence_id"] for item in cluster["evidence"]] + ["invented"],
            }
        ]
    }
    events, rejected = _validate_events(response, clusters)
    assert rejected == 0
    assert len(events) == 1
    assert events[0]["monetary_hawkishness"] == 1.0
    assert "invented" not in json.loads(events[0]["evidence_ids"])


def test_news_session_gap_fallback_is_bounded_and_explicit() -> None:
    payload = {
        "generated_at": "2026-08-31T04:00:00+00:00",
        "window_filter": {
            "fallback_used": True,
            "requested_start": "2026-08-30T04:00:00+00:00",
            "oldest_retained_at": "2026-08-28T21:00:00+00:00",
        },
        "records": [
            {
                "provider_code": "BRFG",
                "article_id": "weekend-fallback",
                "timestamp": "2026-08-28T21:00:00+00:00",
                "timestamp_parse_status": "parsed",
                "window_status": "session_gap_fallback",
                "headline": "Friday close market summary",
            },
            {
                "provider_code": "BRFG",
                "article_id": "too-old",
                "timestamp": "2026-08-26T21:00:00+00:00",
                "timestamp_parse_status": "parsed",
                "window_status": "session_gap_fallback",
                "headline": "Stale market summary",
            },
        ],
    }

    records, counters, _ = _normalize_records(payload, 24, fallback_max_age_hours=96)

    assert [record["article_id"] for record in records] == ["weekend-fallback"]
    assert counters["window_fallback_used"] is True
    assert counters["outside_window"] == 1

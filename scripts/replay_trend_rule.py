"""The first registered replay: the trend rule against the research event.

Everything below is frozen *before* the numbers are looked at. The predecessor
chose benchmarks after seeing results; this script records its criteria first
and its verdict second, so the outcome cannot be talked into existence.

Frozen criteria (2026-09-09):
  capability            risk_identification
  event                 drawdown_5pct_20_sessions (research.py, frozen)
  candidate             TREND_DAMAGED: SPY below its 200-session average,
                        run through the product's own confirmation smoothing
                        (3 sessions to open, 5 to release)
  comparator            a trailing-volatility alert through the same smoothing,
                        its percentile threshold chosen on the train window to
                        match the candidate's train-half signal frequency
  train / test          2000-01-03 .. 2013-06-30 / 2013-07-01 .. latest
  primary metric        episode recall on the test window, under the constraint
                        that confirmed signals stay within a monthly-frequency
                        budget of 12 per year
  verdict               PASSED: candidate recall above the frequency-matched
                        comparator on test, and within budget
                        EVIDENCE_OF_HARM: candidate recall at least 0.05 below
                        the comparator on test
                        otherwise INSUFFICIENT - recorded, never forgotten

A PASSED verdict promotes the experiment one rung, to `replayed_and_explainable`.
That is not enough to influence a recommendation - the ladder still requires a
frozen shadow period and forward data that did not exist when the rule froze.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_state_lab.config import load_config, project_path  # noqa: E402
from market_state_lab.data.public import PublicDataLoader  # noqa: E402
from market_state_lab.evaluation import (  # noqa: E402
    TRADING_DAYS,
    build_episodes,
    forward_drawdown_event,
    replay_metrics,
)
from market_state_lab.events import EventLog, EventRules  # noqa: E402
from market_state_lab.research import (  # noqa: E402
    EVIDENCE_OF_HARM,
    INSUFFICIENT,
    PASSED,
    PRIMARY_EVENT,
    REPLAYED,
    REQUIRED_BENCHMARKS,
    RISK_IDENTIFICATION,
    Experiment,
    Registry,
    open_registry,
    promote,
    record_outcome,
)

TRAIN_END = pd.Timestamp("2013-06-30")
SIGNAL_BUDGET_PER_YEAR = 12.0
RECALL_MARGIN = 0.05
EPISODE_GAP = 5
EXPERIMENT_NAME = "trend_rule_vs_drawdown_event_replay_v1"


def _causal_expanding_percentile(series: pd.Series, minimum: int = 252) -> pd.Series:
    clean = series.dropna()
    ranks = clean.expanding(min_periods=minimum).apply(
        lambda window: float((window[:-1] < window[-1]).mean()) if len(window) > 1 else float("nan"),
        raw=True,
    )
    return ranks.reindex(series.index)


def _signals_from_labels(
    labels: pd.Series,
    confirm: int,
    release: int,
) -> pd.Series:
    """Raw labels through the product's event machine: what the user sees.

    The reader is told about confirmed events, not daily label flickers, so the
    replay scores the confirmed standing state - otherwise it measures a signal
    the product never shows.
    """
    log = EventLog(rules=EventRules(confirm_sessions=confirm, release_sessions=release))
    signals: list[float] = []
    for session, present in labels.dropna().items():
        log.observe(["trend_damaged"] if present else [], str(session.date()))
        signals.append(float(any(e.open for e in log.open_events())))
    return pd.Series(signals, index=labels.dropna().index)


def _signal_rate(signals: pd.Series) -> float:
    """Openings per year, re-openings included: the actionable unit."""
    run_starts = int((signals.diff() == 1).sum())
    return run_starts / max(1, len(signals)) * TRADING_DAYS


def _vol_alert(vol: pd.Series, percentile: float) -> pd.Series:
    ranks = _causal_expanding_percentile(vol)
    return (ranks >= percentile).astype(float)


def _match_threshold(train_alerts: pd.Series, train_vol: pd.Series, target_rate: float) -> float:
    grid = [p / 100.0 for p in range(50, 99)]
    best, best_gap = 0.95, float("inf")
    for percentile in grid:
        alerts = _vol_alert(train_vol, percentile).reindex(train_alerts.index)
        rate = _signal_rate(_signals_from_labels(alerts.fillna(0.0).astype(bool), 3, 5))
        gap = abs(rate - target_rate)
        if gap < best_gap:
            best, best_gap = percentile, gap
    return best


def measure(bundle) -> dict[str, object]:
    spy = bundle.etf_close["spy"].dropna()
    event = forward_drawdown_event(spy, PRIMARY_EVENT.horizon_sessions, PRIMARY_EVENT.drawdown_threshold)

    # Both rules share one convention: the signal is computed at close t-1 and
    # the alert applies from session t. The comparator gets the same shift the
    # candidate has, or it would quietly act one day earlier on every event.
    below = (spy <= spy.rolling(200).mean()).shift(1)
    candidate = _signals_from_labels(below.fillna(0.0).astype(bool), 3, 5)

    returns = spy.pct_change()
    vol20 = returns.rolling(20).std() * (TRADING_DAYS ** 0.5)
    train_vol = vol20.loc[:TRAIN_END]
    train_alerts = candidate.loc[:TRAIN_END]
    target_rate = _signal_rate(train_alerts)
    matched = _match_threshold(train_alerts, train_vol, target_rate)
    comparator = _signals_from_labels(
        _vol_alert(vol20, matched).shift(1).fillna(0.0).astype(bool), 3, 5
    )

    episodes = build_episodes(event, gap_sessions=EPISODE_GAP)
    test_episodes = episodes.loc[episodes["start"] >= pd.Timestamp("2013-07-01")]
    train_episodes = episodes.loc[episodes["end"] <= TRAIN_END]

    test_event = event.loc[event.index >= pd.Timestamp("2013-07-01")]
    candidate_metrics = replay_metrics(candidate, event, test_episodes)
    comparator_metrics = replay_metrics(comparator, event, test_episodes)
    candidate_rate = _signal_rate(candidate.loc[test_event.index])
    comparator_rate = _signal_rate(comparator.loc[test_event.index])

    within_budget = candidate_rate <= SIGNAL_BUDGET_PER_YEAR
    recall_gap = candidate_metrics["episode_recall"] - comparator_metrics["episode_recall"]
    if within_budget and recall_gap > 0:
        outcome, detail = PASSED, (
            f"test recall {candidate_metrics['episode_recall']:.2f} above the "
            f"frequency-matched vol rule's {comparator_metrics['episode_recall']:.2f} "
            f"({recall_gap:+.2f}), {candidate_rate:.1f} signals/year within the "
            f"{SIGNAL_BUDGET_PER_YEAR:.0f}/year budget"
        )
    elif recall_gap < -RECALL_MARGIN:
        outcome, detail = EVIDENCE_OF_HARM, (
            f"test recall {candidate_metrics['episode_recall']:.2f} below the "
            f"frequency-matched vol rule's {comparator_metrics['episode_recall']:.2f} "
            f"({recall_gap:+.2f})"
        )
    else:
        outcome, detail = INSUFFICIENT, (
            f"test recall {candidate_metrics['episode_recall']:.2f} vs the vol rule's "
            f"{comparator_metrics['episode_recall']:.2f} ({recall_gap:+.2f}), "
            f"{candidate_rate:.1f} signals/year "
            f"{'within' if within_budget else 'over'} the "
            f"{SIGNAL_BUDGET_PER_YEAR:.0f}/year budget"
        )

    return {
        "criteria": {
            "event": PRIMARY_EVENT.name,
            "candidate": "trend_damaged (SPY below 200-session MA), 3-session confirm, 5-session release",
            "comparator": f"trailing-vol expanding percentile >= {matched:.0%} matched on train",
            "train_window": f"2000-01-03 .. {TRAIN_END.date()}",
            "test_window": "2013-07-01 .. latest",
            "primary_metric": "episode recall at <= 12 confirmed signals per year",
            "budget_per_year": SIGNAL_BUDGET_PER_YEAR,
        },
        "train": {
            "candidate_signals_per_year": round(target_rate, 2),
            "episodes": int(len(train_episodes)),
        },
        "test": {
            "episodes": int(len(test_episodes)),
            "candidate": {
                **{k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in candidate_metrics.items()},
                "signals_per_year": round(candidate_rate, 2),
            },
            "comparator": {
                **{k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in comparator_metrics.items()},
                "signals_per_year": round(comparator_rate, 2),
            },
        },
        "verdict": outcome,
        "detail": detail,
    }


def record(result: dict[str, object]) -> Experiment:
    verdict = result["verdict"]
    registry = open_registry()
    experiment = Experiment(
        name=EXPERIMENT_NAME,
        capability=RISK_IDENTIFICATION,
        hypothesis=(
            "the trend rule warns of 5% drawdowns within 20 sessions better than a "
            "frequency-matched volatility rule, within a monthly-frequency budget"
        ),
        event=PRIMARY_EVENT,
        benchmarks=list(REQUIRED_BENCHMARKS),
        frozen_on="2026-09-09",
        family="trend-rule-replays",
        family_size=1,
        primary_metric="episode recall at <= 12 confirmed signals per year",
        sacrifice_budget="not reached: risk identification, not action value",
    )
    experiment = record_outcome(experiment, verdict, result["detail"])
    if verdict == PASSED:
        experiment = promote(
            experiment, REPLAYED,
            {
                "metric": experiment.primary_metric,
                "value": result["test"]["candidate"]["episode_recall"],
                "benchmark": "frequency-matched volatility rule",
                "window": "2013-07-01 .. latest",
            },
        )
    # Idempotent re-runs replace the record rather than duplicating it.
    kept = [e for e in registry.experiments if e.name != EXPERIMENT_NAME]
    registry = Registry(kept)
    registry.register(experiment)
    path = project_path({}, "data") / "research_registry.json"
    registry.save(path)
    return experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="run on the synthetic fixture")
    args = parser.parse_args()
    config = load_config()
    if args.offline:
        from market_state_lab.data.fixtures import load_offline_fixture
        bundle = load_offline_fixture(config)
    else:
        bundle = PublicDataLoader(config).load()

    result = measure(bundle)
    experiment = record(result)
    out = project_path({}, "reports") / "research"
    out.mkdir(parents=True, exist_ok=True)
    (out / "trend_rule_replay.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame([result["test"]["candidate"], result["test"]["comparator"]]).to_csv(
        out / "trend_rule_replay_metrics.csv", index=False
    )
    print(json.dumps(result, indent=2, default=str))
    print(f"registry grade: {experiment.grade}, outcome: {experiment.outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

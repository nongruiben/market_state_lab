from __future__ import annotations

from typing import Any

import pandas as pd

from market_state_lab.timeutils import session_age


def evaluate_manifest(
    manifest: pd.DataFrame,
    config: dict[str, Any],
    market_session: str,
) -> pd.DataFrame:
    result = manifest.copy()
    if result.empty:
        return result

    health = config["data"].get("source_health", {})
    rules = health.get("rules", {})
    default_max_age = int(health.get("default_max_age_business_days", 10))
    default_min_years = float(health.get("default_min_history_years", 0.0))
    default_min_rows = int(health.get("default_min_rows", 1))
    default_min_density = float(health.get("default_min_observation_density", 0.0))
    calendar_name = str(config["project"].get("market_calendar", "XNYS"))
    ages: list[int | None] = []
    required: list[bool] = []
    maximums: list[int] = []
    minimum_years: list[float] = []
    minimum_rows: list[int] = []
    minimum_density: list[float] = []
    history_years: list[float] = []
    density: list[float] = []
    for row in result.itertuples(index=False):
        rule = rules.get(str(row.dataset), {})
        ages.append(session_age(calendar_name, getattr(row, "latest_date", None), market_session))
        required.append(bool(rule.get("required", False)))
        maximums.append(int(rule.get("max_age_business_days", default_max_age)))
        minimum_years.append(float(rule.get("min_history_years", default_min_years)))
        minimum_rows.append(int(rule.get("min_rows", default_min_rows)))
        minimum_density.append(float(rule.get("min_observation_density", default_min_density)))
        history_years.append(_history_years(getattr(row, "earliest_date", None), market_session))
        density.append(
            _observation_density(
                getattr(row, "earliest_date", None),
                getattr(row, "latest_date", None),
                int(getattr(row, "rows", 0) or 0),
            )
        )
    result["age_business_days"] = ages
    result["required"] = required
    result["max_age_business_days"] = maximums
    result["min_history_years"] = minimum_years
    result["min_rows"] = minimum_rows
    result["min_observation_density"] = minimum_density
    result["history_years"] = history_years
    result["observation_density"] = density

    fetch_ok = result["status"].ne("failed") & result["rows"].gt(0)
    fresh = result["age_business_days"].notna() & (
        result["age_business_days"] <= result["max_age_business_days"]
    )
    # A source can be perfectly fresh and still be useless: FRED's graph CSV serves
    # only a rolling three-year window for the licensed ICE BofA spread series, so
    # hy_oas arrived every day looking healthy while missing 89% of its history.
    # Recency alone never detects that; coverage has to be checked explicitly.
    deep_enough = result["rows"].ge(result["min_rows"]) & (
        result["history_years"].isna() | result["history_years"].ge(result["min_history_years"])
    )
    dense_enough = result["observation_density"].isna() | result["observation_density"].ge(
        result["min_observation_density"]
    )
    # The offline fixture spans a deliberately short synthetic window, so depth and
    # density say nothing about it. These rules exist to catch a *live* source that
    # quietly started serving less than it used to.
    synthetic = result["status"].eq("fixture")
    complete = (deep_enough & dense_enough) | synthetic

    result["fetch_status"] = fetch_ok.map({True: "ok", False: "failed"})
    result["observation_freshness"] = fresh.map({True: "fresh", False: "stale"})
    result["history_coverage"] = complete.map({True: "complete", False: "truncated"})
    result["model_eligible"] = fetch_ok & fresh & complete
    result["health_status"] = "ok"
    result.loc[fetch_ok & fresh & ~complete, "health_status"] = "truncated"
    result.loc[fetch_ok & ~fresh, "health_status"] = "stale"
    result.loc[~fetch_ok, "health_status"] = "failed"
    return result


def _history_years(earliest: Any, as_of: Any) -> float:
    earliest_ts = pd.to_datetime(earliest, errors="coerce")
    as_of_ts = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(earliest_ts) or pd.isna(as_of_ts):
        return float("nan")
    return max(0.0, (as_of_ts - earliest_ts).days / 365.25)


def _observation_density(earliest: Any, latest: Any, rows: int) -> float:
    """Rows actually delivered divided by the business days they claim to span.

    Catches interior holes that neither a row count nor a span check would see.
    """
    earliest_ts = pd.to_datetime(earliest, errors="coerce")
    latest_ts = pd.to_datetime(latest, errors="coerce")
    if pd.isna(earliest_ts) or pd.isna(latest_ts) or rows <= 0:
        return float("nan")
    expected = len(pd.date_range(earliest_ts, latest_ts, freq="B"))
    if expected <= 0:
        return float("nan")
    return min(1.0, rows / expected)


def required_health_failures(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty or "model_eligible" not in manifest:
        return pd.DataFrame()
    required_datasets = manifest.loc[manifest["required"], "dataset"].drop_duplicates()
    failures: list[pd.DataFrame] = []
    for dataset in required_datasets:
        rows = manifest.loc[manifest["dataset"].eq(dataset)]
        if not rows["model_eligible"].any():
            failures.append(rows)
    return pd.concat(failures, ignore_index=True) if failures else pd.DataFrame()


def eligible_datasets(manifest: pd.DataFrame) -> set[str]:
    if manifest.empty or "model_eligible" not in manifest:
        return set()
    return set(manifest.loc[manifest["model_eligible"], "dataset"].astype(str))

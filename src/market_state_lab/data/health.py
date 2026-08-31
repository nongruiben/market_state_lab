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
    calendar_name = str(config["project"].get("market_calendar", "XNYS"))
    ages: list[int | None] = []
    required: list[bool] = []
    maximums: list[int] = []
    for row in result.itertuples(index=False):
        rule = rules.get(str(row.dataset), {})
        ages.append(session_age(calendar_name, getattr(row, "latest_date", None), market_session))
        required.append(bool(rule.get("required", False)))
        maximums.append(int(rule.get("max_age_business_days", default_max_age)))
    result["age_business_days"] = ages
    result["required"] = required
    result["max_age_business_days"] = maximums
    fetch_ok = result["status"].ne("failed") & result["rows"].gt(0)
    fresh = result["age_business_days"].notna() & (
        result["age_business_days"] <= result["max_age_business_days"]
    )
    result["fetch_status"] = fetch_ok.map({True: "ok", False: "failed"})
    result["observation_freshness"] = fresh.map({True: "fresh", False: "stale"})
    result["model_eligible"] = fetch_ok & fresh
    result["health_status"] = "ok"
    result.loc[fetch_ok & ~fresh, "health_status"] = "stale"
    result.loc[~fetch_ok, "health_status"] = "failed"
    return result


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

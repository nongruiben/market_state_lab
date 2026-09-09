from __future__ import annotations

import importlib.util
import os
import socket
from typing import Any

import pandas as pd

from market_state_lab.config import PROJECT_ROOT
from market_state_lab.data.ibkr import installed_client_version, supported_client_version


def run_diagnostics(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for package in (
        "numpy", "pandas", "pyarrow", "yaml", "requests", "sklearn", "scipy",
        "exchange_calendars",
    ):
        installed = importlib.util.find_spec(package) is not None
        rows.append({"check": f"dependency:{package}", "status": "ok" if installed else "failed", "detail": ""})
    client_version = installed_client_version()
    client_ok = supported_client_version(client_version)
    rows.append(
        {
            "check": "dependency:ib_async",
            "status": "ok" if client_ok else "warning",
            "detail": (
                f"ib_async {client_version}"
                if client_ok
                else "optional for public runs; pip install 'ib-async>=2.0' to enable TWS reads"
            ),
        }
    )

    readonly = bool(config.get("ibkr", {}).get("readonly_required"))
    enabled = bool(config.get("ibkr", {}).get("enabled", True))
    rows.append({"check": "ibkr:readonly_required", "status": "ok" if readonly else "failed", "detail": str(readonly)})
    # Reports the two gates that actually exist. The old auto_connect_disabled
    # check read a config key nothing enforced, so it always passed and said
    # nothing; the real guarantee is that a connection needs the --with-ibkr
    # flag, and ibkr.enabled can now refuse even that.
    pipeline_source = (PROJECT_ROOT / "src" / "market_state_lab" / "pipeline.py").read_text(
        encoding="utf-8"
    )
    # The gate that actually exists: no client is constructed before the
    # with_ibkr early return, and the snapshot client sits inside `if with_ibkr:`.
    # Two constructions is correct - one cross-source, one snapshot - so the old
    # "== 1" count reported failure on the very refactor that kept the invariant.
    gated = (
        pipeline_source.count("ReadOnlyIBKRClient(") == 2
        and "if with_ibkr:" in pipeline_source
        and "if not with_ibkr" in pipeline_source
        and pipeline_source.index("if not with_ibkr")
        < pipeline_source.index("ReadOnlyIBKRClient(")
        and pipeline_source.index("ReadOnlyIBKRClient(")
        < pipeline_source.index("if with_ibkr:")
    )
    rows.append(
        {
            "check": "ibkr:connection_requires_explicit_flag",
            "status": "ok" if gated else "failed",
            "detail": f"both clients behind the with_ibkr gate; enabled={enabled}",
        }
    )

    source = (PROJECT_ROOT / "src" / "market_state_lab" / "data" / "ibkr.py").read_text(encoding="utf-8")
    forbidden = [token for token in ("place" + "Order", "cancel" + "Order", "reqOpenOrders") if token in source]
    rows.append({
        "check": "ibkr:no_order_api",
        "status": "ok" if not forbidden else "failed",
        "detail": ",".join(forbidden),
    })

    host = str(config["ibkr"]["host"])
    port = int(config["ibkr"]["port"])
    try:
        with socket.create_connection((host, port), timeout=1.0):
            port_status = "ok"
            detail = "TWS socket is open"
    except OSError as exc:
        port_status = "warning"
        detail = f"TWS not reachable now: {exc}"
    rows.append({"check": f"ibkr:socket:{host}:{port}", "status": port_status, "detail": detail})

    fred_mode = str(config["data"].get("fred", {}).get("vintage_mode", "latest"))
    has_fred_key = bool(os.getenv("FRED_API_KEY", "").strip())
    rows.append(
        {
            "check": "data:alfred_key",
            "status": "ok" if fred_mode != "point_in_time" or has_fred_key else "failed",
            "detail": f"vintage_mode={fred_mode}; FRED_API_KEY={'set' if has_fred_key else 'missing'}",
        }
    )

    manifest_path = PROJECT_ROOT / "reports" / "data_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        if {"required", "model_eligible", "dataset"}.issubset(manifest.columns):
            required = manifest.loc[manifest["required"].astype(bool)]
            failed = [
                dataset
                for dataset, group in required.groupby("dataset")
                if not group["model_eligible"].astype(bool).any()
            ]
            rows.append(
                {
                    "check": "data:required_freshness",
                    "status": "failed" if failed else "ok",
                    "detail": ",".join(failed),
                }
            )
    else:
        rows.append(
            {
                "check": "data:required_freshness",
                "status": "warning",
                "detail": "run the pipeline once to create data_manifest.csv",
            }
        )

    # The old check read market_features.parquet, an artefact the forecasting
    # pipeline used to write and nothing writes now - it passed on stale data or
    # failed on an empty file, both for the wrong reason. The evidence layer
    # needs SPY closes, so that is what this checks.
    processed_path = PROJECT_ROOT / "data" / "processed" / "etf_close.parquet"
    if processed_path.exists():
        closes = pd.read_parquet(processed_path)
        if "spy" in closes.columns:
            usable = int(closes["spy"].notna().sum())
            latest = str(closes["spy"].dropna().index.max().date()) if usable else "none"
            rows.append(
                {
                    "check": "evidence:spy_history",
                    "status": "ok" if usable >= 252 else "failed",
                    "detail": f"{usable} closes, latest {latest}",
                }
            )
        else:
            rows.append(
                {
                    "check": "evidence:spy_history",
                    "status": "failed",
                    "detail": "etf_close.parquet has no spy column",
                }
            )
    else:
        rows.append(
            {
                "check": "evidence:spy_history",
                "status": "warning",
                "detail": "run the pipeline once to create data/processed/etf_close.parquet",
            }
        )

    sidecar_value = str(config.get("news", {}).get("sidecar_script", ""))
    if sidecar_value:
        sidecar = (PROJECT_ROOT / sidecar_value).resolve()
        if sidecar.exists():
            news_source = sidecar.read_text(encoding="utf-8")
            safe = "readonly=True" in news_source and not any(
                token in news_source for token in ("placeOrder", "cancelOrder")
            )
            rows.append(
                {
                    "check": "news:sidecar_readonly",
                    "status": "ok" if safe else "failed",
                    "detail": str(sidecar),
                }
            )
    return pd.DataFrame(rows)

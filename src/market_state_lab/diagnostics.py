from __future__ import annotations

import importlib.util
import socket
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from market_state_lab.config import PROJECT_ROOT
from market_state_lab.data.ibkr import supported_ibapi_version


def run_diagnostics(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for package in ("numpy", "pandas", "pyarrow", "yaml", "requests", "sklearn", "plotly"):
        installed = importlib.util.find_spec(package) is not None
        rows.append({"check": f"dependency:{package}", "status": "ok" if installed else "failed", "detail": ""})
    try:
        ibapi_version = metadata.version("ibapi")
    except metadata.PackageNotFoundError:
        ibapi_version = None
    ibapi_status = "ok" if supported_ibapi_version(ibapi_version) else "warning"
    ibapi_detail = (
        f"official API version {ibapi_version}"
        if ibapi_status == "ok"
        else "optional for public runs; install official IBKR API 10.x with install_ibkr_api.ps1"
    )
    rows.append({"check": "dependency:ibapi", "status": ibapi_status, "detail": ibapi_detail})

    readonly = bool(config.get("ibkr", {}).get("readonly_required"))
    auto_connect = bool(config.get("ibkr", {}).get("auto_connect"))
    rows.append({"check": "ibkr:readonly_required", "status": "ok" if readonly else "failed", "detail": str(readonly)})
    rows.append({"check": "ibkr:auto_connect_disabled", "status": "ok" if not auto_connect else "failed", "detail": str(auto_connect)})

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
    return pd.DataFrame(rows)

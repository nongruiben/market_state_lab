"""P0 capability probe: find out what this TWS session can actually serve.

Read-only and deliberately small. The point is not to fetch data but to record
what is and is not available before anything is built on top of it - entitlements,
actual market-data type, option-chain reach and clock skew are all things the
plan currently assumes rather than knows.

    .\\.venv\\Scripts\\python.exe scripts\\tws_capability_probe.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_state_lab.config import load_config  # noqa: E402
from market_state_lab.data.ibkr import (  # noqa: E402
    MARKET_DATA_TYPE_NAMES,
    ReadOnlyIBKRClient,
)

PROBE_CLIENT_ID = 917  # distinct, so a running sidecar keeps its own session


def main() -> int:
    config = load_config(str(ROOT / "configs" / "settings.yml"))
    config["ibkr"]["client_id"] = PROBE_CLIENT_ID
    findings: dict[str, object] = {"probed_at_utc": datetime.now(timezone.utc).isoformat()}

    with ReadOnlyIBKRClient(config) as client:
        clock = client.server_clock()
        findings["clock"] = {k: str(v) for k, v in clock.items()}
        print(f"clock      server {clock['server_time_utc']}  skew {clock['skew_seconds']:+.2f}s")

        spy = client.qualify_stock("SPY")
        findings["spy_contract"] = {
            "con_id": spy.conId,
            "primary_exchange": spy.primaryExchange,
            "currency": spy.currency,
        }
        print(f"contract   SPY conId={spy.conId} primary={spy.primaryExchange}")

        quotes = client.quotes([spy])
        row = quotes.iloc[0]
        findings["spy_quote"] = {
            "status": row["status"],
            "bid": row["bid"],
            "ask": row["ask"],
            "last": row["last"],
            "close": row["close"],
            "requested_type": int(row["requested_market_data_type"]),
            "actual_type": row["actual_market_data_type"],
            "actual_type_name": row["actual_market_data_type_name"],
            "quote_age_seconds": row["quote_age_seconds"],
        }
        print(
            f"quote      {row['status']} bid={row['bid']} ask={row['ask']} close={row['close']} "
            f"requested={MARKET_DATA_TYPE_NAMES.get(int(row['requested_market_data_type']))} "
            f"actual={row['actual_market_data_type_name']}"
        )

        try:
            params = client.option_parameters(spy)
            print(
                params[
                    ["exchange", "trading_class", "multiplier", "expiry_count", "strike_count"]
                ].to_string(index=False)
            )
            # Widest chain, rather than assuming SMART is present or listed first.
            chain = params.loc[params["strike_count"].idxmax()]
            findings["option_chain"] = {
                "exchanges": sorted(params["exchange"].tolist()),
                "trading_class": chain["trading_class"],
                "multiplier": chain["multiplier"],
                "expiry_count": int(chain["expiry_count"]),
                "strike_count": int(chain["strike_count"]),
                "nearest_expiries": chain["expirations"][:6],
            }
            print(
                f"chain      {int(chain['expiry_count'])} expiries, "
                f"{int(chain['strike_count'])} strikes, next {chain['expirations'][:3]}"
            )

            spot = row["last"] or row["close"] or row["ask"] or row["bid"]
            if spot:
                strikes = [s for s in chain["strikes"] if 0.85 * spot <= s <= 1.0 * spot][-4:]
                expiry = next(
                    (e for e in chain["expirations"] if 25 <= _days_out(e) <= 70),
                    chain["expirations"][0],
                )
                puts = client.qualify_puts("SPY", expiry, strikes, trading_class=chain["trading_class"])
                findings["put_qualification"] = {
                    "expiry": expiry,
                    "requested_strikes": strikes,
                    "qualified": len(puts),
                }
                print(f"puts       expiry {expiry}: {len(puts)}/{len(strikes)} qualified")
                if puts:
                    pq = client.quotes(puts)
                    findings["put_quotes"] = json.loads(
                        pq[["strike", "bid", "ask", "status", "actual_market_data_type_name"]]
                        .to_json(orient="records")
                    )
                    print(pq[["strike", "bid", "ask", "status"]].to_string(index=False))
        except Exception as exc:  # a missing entitlement is a finding, not a crash
            findings["option_chain_error"] = f"{type(exc).__name__}: {exc}"
            print(f"chain      FAILED {type(exc).__name__}: {exc}")

        findings["request_log"] = json.loads(client.request_log.to_json(orient="records"))

    out = ROOT / "reports" / "tws_capability_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def _days_out(expiry: str) -> int:
    try:
        target = datetime.strptime(expiry, "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return -1
    return (target - datetime.now(timezone.utc)).days


if __name__ == "__main__":
    raise SystemExit(main())

"""Price a small set of protective puts side by side, from the live chain.

Chain -> candidates -> confirmed strikes -> qualified contracts -> quotes ->
parity check -> expiry payoff. Every stage refuses to guess: a strike the expiry
does not list, a contract TWS will not qualify, or a leg with no ask is carried
into the output as a marked row rather than dropped, because a comparison
silently missing one of its alternatives is worse than one that says what it
could not price.

The quotes are checked before they are used. A delayed feed, a frozen book and a
mis-qualified contract all return numbers that look like prices, so the call
opposite each put is fetched purely to test put-call parity - an arbitrage
identity that needs no model, no extra subscription and no second data source.
The table says whether its own inputs passed.

Read-only. Nothing here reads an account, sizes against real holdings, ranks the
candidates, or suggests that buying protection is a good idea. It answers one
arithmetic question - what does this listed put cost, and what does it pay at
expiry against a stated reference exposure - and leaves the trade-off between a
shallower strike and a cheaper one to the reader.

    .\\.venv\\Scripts\\python.exe scripts\\protection_table.py --notional 100000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_state_lab.config import load_config  # noqa: E402
from market_state_lab.data.ibkr import ReadOnlyIBKRClient  # noqa: E402
from market_state_lab.defense_tools import (  # noqa: E402
    attach_contracts,
    attach_quotes,
    plan_candidates,
    put_quotes,
    resolve_strikes,
    select_chain,
)
from market_state_lab.quote_checks import parity_verdict, put_call_parity_check  # noqa: E402
from market_state_lab.scenarios import (  # noqa: E402
    ReferenceExposure,
    protective_put_scenarios,
    summarise_candidate,
)

CLIENT_ID = 923

PLAN_COLS = [
    "bucket",
    "expiry",
    "days_to_expiry",
    "target_moneyness",
    "strike_proposed",
    "strike",
    "actual_moneyness",
    "con_id",
    "bid",
    "ask",
    "relative_spread",
    "status",
]

PARITY_COLS = [
    "expiry",
    "days_to_expiry",
    "pairs",
    "slope",
    "implied_rate",
    "implied_forward",
    "max_abs_residual",
    "worst_residual_tolerance",
    "status",
]

SUMMARY_COLS = [
    "expiry",
    "strike",
    "moneyness",
    "contracts",
    "ask",
    "spread",
    "cost_usd",
    "cost_pct_of_notional",
    "pnl_if_flat",
    "unhedged_pnl_at_worst_move",
    "pnl_at_worst_move",
    "protection_at_worst_move",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--notional", type=float, default=100_000.0)
    parser.add_argument("--coverage", type=float, default=1.0)
    parser.add_argument("--wait", type=float, default=15.0, help="seconds to let quotes stream")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the parity check; the table then reports itself as unverified",
    )
    args = parser.parse_args()

    config = load_config(str(ROOT / "configs" / "settings.yml"))
    config["ibkr"]["client_id"] = CLIENT_ID

    with ReadOnlyIBKRClient(config) as client:
        stock = client.qualify_stock(args.symbol)
        stock_quote = client.quotes([stock], wait_seconds=args.wait).iloc[0]
        spot = stock_quote["last"] or stock_quote["close"]
        if not spot:
            print(f"{args.symbol}: no usable spot ({stock_quote['status']}); stopping")
            return 1
        spot = float(spot)
        print(
            f"spot {spot} ({stock_quote['status']}, "
            f"{stock_quote['actual_market_data_type_name']})"
        )

        chain = select_chain(client.option_parameters(stock), args.symbol)
        plan = plan_candidates(args.symbol, chain, spot=spot, as_of=date.today())
        listed = {
            expiry: client.listed_strikes(
                args.symbol, expiry, "P", trading_class=chain["trading_class"]
            )
            for expiry in sorted({str(e) for e in plan["expiry"].dropna()})
        }
        plan = resolve_strikes(plan, listed)

        puts: list = []
        calls: list = []
        for expiry, group in plan.loc[plan["status"].eq("ok")].groupby("expiry"):
            strikes = [float(s) for s in group["strike"]]
            puts += client.qualify_options(
                args.symbol, str(expiry), strikes, "P", trading_class=chain["trading_class"]
            )
            if not args.no_verify:
                # Calls are fetched to test parity and for no other reason; none
                # of them reaches the payoff table.
                calls += client.qualify_options(
                    args.symbol, str(expiry), strikes, "C", trading_class=chain["trading_class"]
                )
        plan = attach_contracts(plan, puts)

        quotes = (
            client.quotes(puts + calls, wait_seconds=args.wait)
            if (puts or calls)
            else pd.DataFrame()
        )
        plan = attach_quotes(plan, quotes.loc[quotes["right"].eq("P")] if len(quotes) else quotes)
        request_log = client.request_log

    days = {
        str(row["expiry"]): row["days_to_expiry"]
        for row in plan.dropna(subset=["expiry"]).to_dict("records")
    }
    parity = (
        put_call_parity_check(quotes, days)
        if len(quotes) and not args.no_verify
        else pd.DataFrame()
    )
    verdict = parity_verdict(parity)

    print()
    print(plan.reindex(columns=PLAN_COLS).to_string(index=False))

    print(f"\n=== quote verification: put-call parity -> {verdict.upper()} ===")
    if parity.empty:
        print("  not run; these quotes have not been checked against anything")
    else:
        print(parity.reindex(columns=PARITY_COLS).to_string(index=False))
        print(
            "  slope is -e^-rT, so implied_rate is the cash rate net of the dividend\n"
            "  priced into that expiry. Residuals are judged against the quoted\n"
            "  spread, because a mid is only known to within half a spread per leg."
        )
    if verdict == "failed":
        print("\nquotes failed their own consistency check; not building a payoff table")
        return 1

    exposure = ReferenceExposure(args.symbol, spot=spot, notional_usd=args.notional)
    quoted_puts = put_quotes(plan)
    if not quoted_puts:
        print("\nnothing priced; no payoff table to build")
        return 1

    frames = [
        protective_put_scenarios(exposure, put, coverage=args.coverage) for put in quoted_puts
    ]
    summaries = [summarise_candidate(frame) for frame in frames]
    table = pd.DataFrame(summaries)
    sizing = frames[0].attrs

    print()
    print(f"reference exposure: {exposure.shares:,.1f} shares of {args.symbol} = "
          f"${exposure.notional_usd:,.0f} at {spot}")
    print(
        f"requested coverage {args.coverage:.0%}, but contracts are whole: "
        f"{sizing['contracts']} x {int(float(quoted_puts[0].multiplier))} = "
        f"{sizing['shares_covered']:,.0f} shares = {sizing['coverage_ratio']:.1%} actual, "
        f"{sizing['uncovered_shares']:,.1f} shares uncovered"
    )
    print(table.reindex(columns=SUMMARY_COLS).to_string(index=False))

    types = sorted({s["market_data_type"] for s in summaries if s["market_data_type"]})
    print(
        f"\nquotes: {', '.join(types)} - a delayed or frozen ask is not what an order "
        "would fill at."
    )
    print("payoff is at expiry only; nothing here values the put before then.")
    print("no row is ranked: a deeper strike costs less and protects later, which is")
    print("a preference, not an optimum.")

    out = ROOT / "reports" / f"protection_table_{args.symbol.lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "built_at_utc": datetime.now(timezone.utc).isoformat(),
                "symbol": args.symbol,
                "spot": spot,
                "spot_status": stock_quote["status"],
                "spot_market_data_type": stock_quote["actual_market_data_type_name"],
                "reference_notional_usd": args.notional,
                "requested_coverage": args.coverage,
                "actual_coverage_ratio": sizing["coverage_ratio"],
                "trading_class": chain["trading_class"],
                "parity_verdict": verdict,
                "parity_checks": json.loads(parity.to_json(orient="records")),
                "candidates": json.loads(plan.to_json(orient="records")),
                "comparison": summaries,
                "request_log": json.loads(request_log.to_json(orient="records")),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

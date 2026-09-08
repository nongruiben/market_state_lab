"""Compare a short list of protective puts against doing nothing and against owning less.

Chain -> candidates -> confirmed strikes -> qualified contracts -> quotes ->
staleness and parity checks -> hard screen -> expiry payoff -> at most three
candidates beside two controls.

Every stage refuses to guess. A strike the expiry does not list, a contract TWS
will not qualify, a leg with no ask, a spread too wide to cost, an open interest
that never arrived - each is carried into the output as a marked row with the
number that failed, because a comparison silently missing one of its
alternatives reads as "these are the choices".

The two controls are the point of the table. Six puts with no "do nothing" row
reads as "pick one of these", when the first question is whether to buy
protection at all, and the second is whether simply owning less would do the
same job. Neither control is free: owning less gives up the upside, and the
sale pays a spread.

Read-only. Nothing here reads an account, sizes against real holdings, ranks
the candidates, or suggests that buying protection is a good idea.

    .\\.venv\\Scripts\\python.exe scripts\\protection_table.py --notional 100000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_state_lab.config import load_config  # noqa: E402
from market_state_lab.data.ibkr import ReadOnlyIBKRClient  # noqa: E402
from market_state_lab.data.snapshots import (  # noqa: E402
    latest_sessions,
    write_snapshot,
)
from market_state_lab.defense_tools import (  # noqa: E402
    ScreenLimits,
    attach_contracts,
    attach_quotes,
    plan_candidates,
    put_quotes,
    resolve_strikes,
    screen_candidates,
    screened,
    select_chain,
    shortlist,
)
from market_state_lab.quote_checks import (  # noqa: E402
    parity_verdict,
    put_call_parity_check,
    staleness_note,
)
from market_state_lab.scenarios import (  # noqa: E402
    ReferenceExposure,
    no_protection_scenarios,
    protective_put_scenarios,
    reduced_exposure_scenarios,
    summarise_candidate,
    summarise_control,
)
from market_state_lab.timeutils import last_completed_session, market_is_open  # noqa: E402

CLIENT_ID = 923

PLAN_COLS = [
    "bucket", "expiry", "days_to_expiry", "target_moneyness", "strike_proposed",
    "strike", "actual_moneyness", "con_id", "bid", "ask", "relative_spread",
    "open_interest", "quote_qualification", "status",
]
PARITY_COLS = [
    "expiry", "days_to_expiry", "pairs", "slope", "implied_rate", "implied_forward",
    "implied_rate_stderr", "strike_span", "max_abs_residual",
    "worst_residual_tolerance", "status",
]
COMPARE_COLS = [
    "label", "structure", "contracts", "cost_usd", "cost_pct_of_notional",
    "pnl_at_worst_move", "protection_at_worst_move", "pnl_if_flat", "pnl_at_best_move",
]
DETAIL_COLS = [
    "label", "protected_below", "unprotected_drop_pct", "uncovered_shares",
    "coverage_ratio", "ask", "spread", "maintenance", "review_when",
]


def analyse(client: Any, symbol: str, args: argparse.Namespace) -> dict[str, Any]:
    """Everything for one underlying. Each is priced against its own reference."""
    stock = client.qualify_stock(symbol)

    # Pass one: a provisional spot, only to choose which strikes to ask about.
    # A few dollars either way barely moves which listed strike is nearest.
    probe = client.quotes([stock], wait_seconds=args.wait).iloc[0]
    provisional = probe["close"] or probe["last"]
    if not provisional:
        return {"symbol": symbol, "error": f"no usable spot ({probe['status']})"}

    chain = select_chain(client.option_parameters(stock), symbol)
    plan = plan_candidates(symbol, chain, spot=float(provisional), as_of=date.today())
    listed = {
        expiry: client.listed_strikes(symbol, expiry, "P", trading_class=chain["trading_class"])
        for expiry in sorted({str(e) for e in plan["expiry"].dropna()})
    }
    plan = resolve_strikes(plan, listed)

    puts: list = []
    calls: list = []
    for expiry, group in plan.loc[plan["status"].eq("ok")].groupby("expiry"):
        strikes = [float(s) for s in group["strike"]]
        puts += client.qualify_options(
            symbol, str(expiry), strikes, "P", trading_class=chain["trading_class"]
        )
        if not args.no_verify:
            # Calls are fetched to test parity and for no other reason; none of
            # them reaches the payoff table.
            calls += client.qualify_options(
                symbol, str(expiry), strikes, "C", trading_class=chain["trading_class"]
            )
    plan = attach_contracts(plan, puts)

    # Pass two: the stock rides in the same batch as the options, so both halves
    # of the table are read from one moment on one feed.
    batch = (
        client.quotes([stock] + puts + calls, wait_seconds=args.wait)
        if (puts or calls)
        else client.quotes([stock], wait_seconds=args.wait)
    )
    quotes = batch.loc[batch["sec_type"].eq("OPT")].reset_index(drop=True)
    stock_quote = batch.loc[batch["sec_type"].eq("STK")].iloc[0]
    stale = staleness_note(quotes) if len(quotes) else {"basis": "unknown", "note": "no quotes"}

    # The spot has to come from the same world as the option book. A frozen book
    # is whatever the market last settled at, so the close is its match; a live
    # book belongs with the last print. Getting this backwards is silent, and it
    # happened: SPY quoted 767 pre-market on a Tuesday while every put was still
    # frozen at Friday's 770.19 close, and the table sized across the two.
    frozen = stale["basis"] in {"frozen_last_session", "mixed"}
    spot_source = args.spot if args.spot != "auto" else ("close" if frozen else "last")
    spot = stock_quote[spot_source] or stock_quote["close"] or stock_quote["last"]
    if not spot:
        return {"symbol": symbol, "error": f"no usable spot ({stock_quote['status']})"}
    spot = float(spot)

    plan = attach_quotes(plan, quotes.loc[quotes["right"].eq("P")] if len(quotes) else quotes)
    plan = screen_candidates(plan, args.limits, stale["basis"], market_is_open())

    days = {
        str(r["expiry"]): r["days_to_expiry"]
        for r in plan.dropna(subset=["expiry"]).to_dict("records")
    }
    parity = (
        put_call_parity_check(quotes, days) if len(quotes) and not args.no_verify
        else pd.DataFrame()
    )

    exposure = ReferenceExposure(symbol, spot=spot, notional_usd=args.notional)
    horizon_days = int(
        plan.loc[plan["bucket"].eq(args.horizon), "days_to_expiry"].dropna().median() or 30
    ) if plan["bucket"].eq(args.horizon).any() else 30
    controls = [
        summarise_control(no_protection_scenarios(exposure)),
        summarise_control(
            reduced_exposure_scenarios(
                exposure,
                reduce_to=args.reduce_to,
                horizon_days=horizon_days,
                cash_rate=args.cash_rate,
            )
        ),
    ]

    survivors = screened(plan)
    summaries: list[dict[str, Any]] = []
    for put, row in zip(put_quotes(survivors), survivors.to_dict("records")):
        summary = summarise_candidate(
            protective_put_scenarios(exposure, put, coverage=args.coverage)
        )
        summaries.append({**summary, "bucket": row["bucket"],
                          "target_moneyness": row["target_moneyness"]})
    ranked = shortlist(pd.DataFrame(summaries), args.limits, args.max_candidates, args.horizon)

    return {
        "symbol": symbol, "spot": spot, "spot_source": spot_source,
        "spot_status": stock_quote["status"],
        "spot_market_data_type": stock_quote["actual_market_data_type_name"],
        "trading_class": chain["trading_class"], "staleness": stale,
        "parity_verdict": parity_verdict(parity), "parity": parity,
        "plan": plan, "candidates": ranked, "controls": controls,
        "horizon_days": horizon_days, "legs_synchronous": frozen,
    }


def report(result: dict[str, Any], args: argparse.Namespace) -> None:
    symbol = result["symbol"]
    print(f"\n{'=' * 78}\n{symbol}\n{'=' * 78}")
    if result.get("error"):
        print(f"  {result['error']}; no table for this underlying")
        return

    stale = result["staleness"]
    session, _ = last_completed_session()
    print(
        f"spot {result['spot']} from the {result['spot_source']} "
        f"({result['spot_status']}, {result['spot_market_data_type']}); "
        f"option book {stale['basis']}; last completed session {session.date()}"
    )
    print(f"  prices: {stale.get('note', stale['basis'])}")
    print()
    print(result["plan"].reindex(columns=PLAN_COLS).to_string(index=False))

    rejected = result["plan"].loc[result["plan"]["screen_failures"].notna()]
    if len(rejected):
        print("\nnot carried forward:")
        for row in rejected.to_dict("records"):
            print(f"  {row['expiry']} {row['strike']:g}P  {row['screen_failures']}")

    verdict = result["parity_verdict"]
    print(f"\nput-call parity -> {verdict.upper()}")
    if not result["parity"].empty:
        print(result["parity"].reindex(columns=PARITY_COLS).to_string(index=False))
        print("  Coherence, not currency: a book frozen days ago satisfies the identity")
        print("  exactly, because every leg is stale together. The residual test is what")
        print("  proves the quotes hang together; implied_rate is a by-product fitted over")
        print("  strike_span, and implied_rate_stderr is how little of it a short ladder")
        print("  at a near maturity actually resolves.")
        if not result["legs_synchronous"]:
            print("  Legs stream asynchronously intraday, so implied_rate is not a rate")
            print("  estimate here - 0.01 of slope is ~12% of rate at 31 days.")
    if verdict == "failed":
        print("  quotes failed their own consistency check; no payoff table")
        return

    candidates = result["candidates"]
    picked = (
        candidates.loc[candidates["shortlisted"]].to_dict("records")
        if not candidates.empty
        else []
    )
    # Built from records rather than concatenated: the controls have no strike
    # or expiry, and pandas would rather guess at the dtype of an all-empty
    # column than be told.
    table = pd.DataFrame([*picked, *result["controls"]])

    print(
        f"\nreference exposure: ${args.notional:,.0f} of {symbol} at {result['spot']}"
        f"  |  horizon {args.horizon} (~{result['horizon_days']}d)"
        f"  |  de-risk control sells to {args.reduce_to:.0%} at {args.cash_rate:.2%} cash"
    )
    if candidates.empty:
        print("  no candidate cleared the screen; the controls are the whole table")
    print(table.reindex(columns=COMPARE_COLS).to_string(index=False))
    print()
    print(table.reindex(columns=DETAIL_COLS).to_string(index=False))

    if not candidates.empty:
        dropped = candidates.loc[~candidates["shortlisted"]]
        if len(dropped):
            print("\noff the short list:")
            for row in dropped.to_dict("records"):
                print(f"  {row['label']:<24} {row['shortlist_reason']}")



def store(results: list[dict[str, Any]], args: argparse.Namespace, config: dict) -> str | None:
    """Write the run as an immutable snapshot, and say what it may be used for.

    This is where the history comes from. TWS will not sell option quotes for
    last week, so the only way to ever see how the cost of protection moved is
    to keep each day as it happens - and to keep it in a form that says which
    session it belongs to and how good it was, rather than a folder of files
    named by the day someone happened to run the script.

    Eligibility is granted here, not inherited from the data arriving. A book
    frozen at the last close describes that close perfectly and is still not a
    price anyone can trade against, so `instrument_quotes` needs every screened
    row to have qualified VALID, and `training` is never granted by one run.
    """
    usable = [r for r in results if not r.get("error")]
    if not usable:
        return None

    frames: dict[str, pd.DataFrame] = {}
    for name, key in (("candidates", "plan"), ("comparison", "candidates"), ("parity", "parity")):
        parts = [
            r[key].assign(symbol=r["symbol"])
            for r in usable
            if isinstance(r.get(key), pd.DataFrame) and not r[key].empty
        ]
        if parts:
            frames[name] = pd.concat(parts, ignore_index=True)
    frames["controls"] = pd.DataFrame(
        [{**c, "symbol": r["symbol"]} for r in usable for c in r["controls"]]
    )
    if not frames:
        return None

    sessions = {
        r["staleness"].get("last_completed_session") for r in usable
    } - {None}
    session_date = sorted(sessions)[-1] if sessions else str(last_completed_session()[0].date())

    # A single frozen post-close book: good enough to describe the session, and
    # good enough to price a contract only if nothing was flagged on the way.
    qualifications = set(frames.get("candidates", pd.DataFrame()).get(
        "quote_qualification", pd.Series(dtype=str)
    ))
    all_valid = qualifications <= {"VALID"} and bool(qualifications)
    eligible = ["day_end_analysis"] + (["instrument_quotes"] if all_valid else [])
    ineligible = {
        "intraday_observation": "post-close run on a frozen book",
        "training": "one session; a training set is granted over a series, not a run",
    }
    if not all_valid:
        ineligible["instrument_quotes"] = (
            "not every screened row qualified VALID: "
            + ", ".join(sorted(qualifications - {"VALID"}))
        )

    snapshot = write_snapshot(
        ROOT / "data",
        session_date,
        frames,
        config,
        eligible_for=tuple(eligible),
        ineligibility=ineligible,
        project_root=ROOT,
    )
    revision = snapshot.manifest.get("revision", 1)
    history = latest_sessions(ROOT / "data")
    line = f"\nsnapshot {snapshot.snapshot_id} rev {revision}"
    if snapshot.manifest.get("supersedes"):
        # A fuller look at the same close supersedes the earlier one and says
        # what moved. It never rewrites it: the first reading stays readable.
        changed = ", ".join(snapshot.manifest.get("changed_tables") or []) or "nothing"
        line += f", superseding {snapshot.manifest['supersedes']} (changed: {changed})"
    print(
        f"{line}  eligible_for={','.join(eligible)}"
        f"\n  {len(history)} session(s) recorded"
        f" ({history['session_date'].min()} to {history['session_date'].max()}),"
        f" {int(history['revisions'].sum())} observation(s)"
        " - the option-price history the feed will not sell is only ever kept forward"
    )
    return snapshot.snapshot_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="SPY,QQQ,IWM")
    parser.add_argument("--notional", type=float, default=100_000.0)
    parser.add_argument("--coverage", type=float, default=1.0)
    parser.add_argument("--horizon", default="30-60d", choices=("30-60d", "60-90d"))
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--reduce-to", type=float, default=0.8)
    parser.add_argument(
        "--cash-rate", type=float, default=0.0,
        help="annual return on the cash raised by the de-risk control; 0 understates it",
    )
    parser.add_argument("--wait", type=float, default=15.0, help="seconds to let quotes stream")
    parser.add_argument("--spot", choices=("auto", "close", "last"), default="auto")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    args.limits = ScreenLimits()

    config = load_config(str(ROOT / "configs" / "settings.yml"))
    config["ibkr"]["client_id"] = CLIENT_ID
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    results = []
    with ReadOnlyIBKRClient(config) as client:
        for symbol in symbols:
            try:
                results.append(analyse(client, symbol, args))
            except Exception as exc:  # one bad underlying must not lose the others
                results.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
        request_log = client.request_log

    for result in results:
        report(result, args)

    if len(symbols) > 1:
        print(f"\n{'=' * 78}")
        print(
            "Each underlying is priced against its own $"
            f"{args.notional:,.0f} reference exposure, so these blocks are not a "
            "ranking.\nA QQQ put is not a substitute hedge for an SPY position: "
            "comparing them as\nhedges for one portfolio needs a fixed common "
            "reference and a stated mapping,\nwhich this does not have."
        )

    snapshot_id = store(results, args, config)

    out = ROOT / "reports" / "protection_table.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "built_at_utc": datetime.now(timezone.utc).isoformat(),
                "snapshot_id": snapshot_id,
                "market_open": market_is_open(),
                "reference_notional_usd": args.notional,
                "horizon": args.horizon,
                "screen_limits": vars(args.limits),
                "reduce_to": args.reduce_to,
                "cash_rate": args.cash_rate,
                "underlyings": [
                    {
                        **{k: v for k, v in r.items()
                           if k not in {"plan", "candidates", "parity", "controls"}},
                        "plan": json.loads(r["plan"].to_json(orient="records"))
                        if isinstance(r.get("plan"), pd.DataFrame) else None,
                        "candidates": json.loads(r["candidates"].to_json(orient="records"))
                        if isinstance(r.get("candidates"), pd.DataFrame) else None,
                        "parity": json.loads(r["parity"].to_json(orient="records"))
                        if isinstance(r.get("parity"), pd.DataFrame) else None,
                        "controls": r.get("controls"),
                    }
                    for r in results
                ],
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

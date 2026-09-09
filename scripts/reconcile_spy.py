"""Cross-check the core input the plan names first: SPY daily closes, TWS against Yahoo.

Read-only, and it decides nothing. It reports where two independent sources
agree, where they do not, and what a difference would have to be worked through
before anyone concluded which one was wrong.

    .\\.venv\\Scripts\\python.exe scripts\\reconcile_spy.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_state_lab.config import load_config  # noqa: E402
from market_state_lab.data.ibkr import ReadOnlyIBKRClient  # noqa: E402
from market_state_lab.data.public import PublicDataLoader  # noqa: E402
from market_state_lab.data.reconciliation import (  # noqa: E402
    SourceSeries,
    agreement_cannot_clear,
    confirm_move,
    reconcile_closes,
    summarise,
    unverifiable,
)
from market_state_lab.data.validation import (  # noqa: E402
    REVIEW,
    issues_frame,
    validate_daily_bars,
)

CLIENT_ID = 931


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--duration", default="1 Y")
    args = parser.parse_args()
    symbol = args.symbol.upper()

    config = load_config()
    config["ibkr"]["client_id"] = CLIENT_ID
    # TWS daily TRADES bars are unadjusted; the comparison refuses to run
    # against an adjusted series, so the flag has to be right rather than
    # convenient.
    os.environ["IBKR_ALLOW_HISTORICAL"] = "1"

    with ReadOnlyIBKRClient(config) as client:
        contract = client.qualify_stock(symbol)
        bars = client.historical_daily_bars(contract, duration=args.duration, what_to_show="TRADES")
    tws = bars["close"]
    tws.index = pd.to_datetime(tws.index).tz_localize(None).normalize()
    print(f"TWS    {len(tws)} sessions, {tws.index.min().date()} to {tws.index.max().date()}")

    # The raw series, not the adjusted one. Yahoo's chart endpoint prefers
    # adjusted closes, and comparing those against unadjusted TWS trades gives a
    # difference that is entirely real and entirely about dividends - a year of
    # SPY is 1.1% of drift, wide enough that a measured tolerance absorbs it and
    # reports agreement. Declaring the convention correctly is the check.
    bundle = PublicDataLoader(config).load()
    public = bundle.etf_close_unadjusted
    column = symbol.lower()
    if column not in public.columns:
        print(f"no unadjusted second source for {symbol}; nothing it is legal to compare")
        return 1
    other = public[column].dropna()
    other.index = pd.to_datetime(other.index).normalize()
    print(f"Yahoo  {len(other)} sessions, {other.index.min().date()} to {other.index.max().date()}")

    single_source = validate_daily_bars(bars.rename_axis(None), symbol)
    result = reconcile_closes(
        symbol,
        SourceSeries("TWS", tws, adjusted=False, session="RTH"),
        SourceSeries("Yahoo", other, adjusted=False, session="RTH"),
    )
    issues = agreement_cannot_clear(result, single_source + result.issues)

    print()
    print(json.dumps(summarise(result), indent=2, default=str))

    reviews = [i for i in issues if i.severity == REVIEW]
    print(f"\nreview items: {len(reviews)}")
    for issue in reviews[:10]:
        print(f"  {issue.code:<22} {issue.subject}")
        print(f"    {issue.detail[:150]}")

    # Every review-level move is put to the second source: row 2 keeps a crash
    # that both saw, and row 9 keeps a quarantine that both share.
    for issue in [i for i in single_source if i.code == "large_price_move"][:10]:
        stamp = pd.Timestamp(issue.subject.split(" ", 1)[1])
        if stamp in tws.index and stamp in other.index:
            verdict = confirm_move(result, stamp, SourceSeries("TWS", tws),
                                   SourceSeries("Yahoo", other))
            print(f"  {verdict.code:<22} {verdict.subject}\n    {verdict.detail[:150]}")

    print()
    print(unverifiable(f"{symbol} options", "option quotes arrive from TWS alone").detail
          or "options: no independent source")

    out = ROOT / "reports" / f"reconciliation_{column}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "built_at_utc": datetime.now(timezone.utc).isoformat(),
                "summary": summarise(result),
                "issues": json.loads(issues_frame(issues).to_json(orient="records")),
                "comparison": json.loads(result.comparison.tail(10).to_json(orient="index")),
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

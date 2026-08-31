from __future__ import annotations

import argparse
import json
import sys

from market_state_lab.config import load_config
from market_state_lab.data.ibkr import ReadOnlyIBKRClient
from market_state_lab.diagnostics import run_diagnostics
from market_state_lab.news import run_news_pipeline
from market_state_lab.pipeline import run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="US market-state and style research lab")
    parser.add_argument("--config", default="configs/settings.yml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="download public data, model, and report")
    run.add_argument("--with-ibkr", action="store_true", help="add read-only delayed snapshots")
    run.add_argument("--offline", action="store_true", help="run the fixed fixture without network access")
    subparsers.add_parser("doctor", help="validate environment and read-only controls")
    check = subparsers.add_parser("ibkr-check", help="explicitly test read-only TWS data access")
    check.add_argument("--symbols", default="SPY,QQQ,IWM")
    news = subparsers.add_parser("news", help="process the TWS news sidecar into research features")
    news.add_argument("--fetch", action="store_true", help="run the guarded read-only TWS sidecar first")
    news.add_argument("--no-llm", action="store_true", help="run quality and clustering without DeepSeek")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "doctor":
        report = run_diagnostics(config)
        print(report.to_string(index=False))
        return 1 if (report["status"] == "failed").any() else 0
    if args.command == "ibkr-check":
        symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
        with ReadOnlyIBKRClient(config) as client:
            print(f"TWS server time: {client.server_clock().isoformat()}")
            print(client.delayed_snapshots(symbols).to_string())
            if not client.errors.empty:
                print(client.errors.to_string(index=False))
        return 0
    if args.command == "news":
        result = run_news_pipeline(config, fetch=args.fetch, use_llm=not args.no_llm)
        print(json.dumps({"quality": result.quality, "llm": result.metadata}, indent=2))
        return 1 if result.metadata.get("status") == "failed" else 0
    outputs = run_pipeline(config, with_ibkr=args.with_ibkr, offline=args.offline)
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

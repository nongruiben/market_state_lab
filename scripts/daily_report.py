r"""The daily run. A thin wrapper over `run_pipeline`, which is the single entry.

Section 18 defines one run as producing one set of artefacts, so there is one
function that produces them and two ways to call it. Two implementations would
eventually give two answers.

    .\.venv\Scripts\python.exe scripts\daily_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_state_lab.config import load_config  # noqa: E402
from market_state_lab.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notional", type=float, default=100_000.0)
    parser.add_argument("--offline", action="store_true", help="run the synthetic fixture")
    parser.add_argument(
        "--with-ibkr", action="store_true",
        help="use TWS as an independent second source for the cross-source check",
    )
    parser.add_argument("--show", action="store_true", help="print the report to stdout")
    args = parser.parse_args()

    written = run_pipeline(
        load_config(), with_ibkr=args.with_ibkr,
        offline=args.offline, notional=args.notional,
    )
    if args.show:
        print(written["report.md"].read_text(encoding="utf-8"))
    print(f"wrote {len(written)} artefacts to {next(iter(written.values())).parent}")
    for name in written:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

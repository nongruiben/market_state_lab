from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_state_lab.cli import main


if __name__ == "__main__":
    raise SystemExit(
        main(["--config", str(ROOT / "configs" / "settings.yml"), "ibkr-check", *sys.argv[1:]])
    )


"""Verify Odrzywołek canonical EML identities."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odrzywolek import verify_canonical_identities  # noqa: E402


def main() -> None:
    print("=== Odrzywołek canonical identity checks ===")
    errs = verify_canonical_identities()
    for key in ("exp", "ln", "euler", "mul"):
        print(f"{key}: max error = {errs[key]:.2e}")
    print("PASS" if errs["ok"] else "FAIL")


if __name__ == "__main__":
    main()

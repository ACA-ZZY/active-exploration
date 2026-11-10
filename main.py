#!/usr/bin/env python3
"""Single entry point for the five-case reliability study."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "benchmarks"))

def main() -> int:
    from run_case import main as run_case

    return run_case()

if __name__ == "__main__":
    raise SystemExit(main())

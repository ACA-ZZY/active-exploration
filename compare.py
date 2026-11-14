from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "benchmarks"))
from uncertainty_benchmark.run_block import main
if __name__ == "__main__":
    raise SystemExit(main())

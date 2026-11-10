"""Subprocess bridge to the isolated authors' gpsearch Python 3.8 runtime"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

ADAPTER_DIR = Path(__file__).resolve().parent
WORKER = ADAPTER_DIR / "gpsearch_worker.py"
DEFAULT_RUNTIME_MANIFEST = ADAPTER_DIR / "gpsearch_runtime.local.json"
EXPECTED_COMMIT = "8c5d77b3f0df803029d26828966274d6185c12e4"

class GPSearchProcessError(RuntimeError):
    """Raised when the isolated original-method worker cannot be trusted."""

def _runtime_python(
    python_executable: str | Path | None = None,
    runtime_manifest: str | Path = DEFAULT_RUNTIME_MANIFEST,
) -> Path:
    if python_executable is not None:
        candidate = Path(python_executable).resolve()
    else:
        override = os.environ.get("GPSEARCH_PYTHON")
        if override:
            candidate = Path(override).resolve()
        else:
            manifest_path = Path(runtime_manifest).resolve()
            if not manifest_path.is_file():
                raise GPSearchProcessError(
                    "No gpsearch runtime manifest and GPSEARCH_PYTHON is unset: "
                    f"{manifest_path}"
                )
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            candidate = Path(manifest["python_executable"]).resolve()
    if not candidate.is_file():
        raise GPSearchProcessError(f"gpsearch Python executable is absent: {candidate}")
    return candidate

def run_gpsearch_step(
    request: str | Path,
    response: str | Path,
    *,
    checkpoint: str | Path | None = None,
    python_executable: str | Path | None = None,
    runtime_manifest: str | Path = DEFAULT_RUNTIME_MANIFEST,
    timeout_seconds: float = 7200.0,
) -> dict[str, Any]:
    """Run one original sequential proposal in the isolated runtime."""

    runtime = _runtime_python(python_executable, runtime_manifest)
    request_path = Path(request).resolve()
    response_path = Path(response).resolve()
    command = [
        str(runtime),
        str(WORKER),
        "--request",
        str(request_path),
        "--response",
        str(response_path),
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", str(Path(checkpoint).resolve())])
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=float(timeout_seconds),
        env=environment,
    )
    if not response_path.is_file():
        raise GPSearchProcessError(
            "gpsearch worker produced no response (exit {}): stdout={!r}; stderr={!r}".format(
                completed.returncode, completed.stdout[-4000:], completed.stderr[-4000:]
            )
        )
    with response_path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    commit = report.get("third_party", {}).get("commit")
    if commit != EXPECTED_COMMIT:
        raise GPSearchProcessError(
            f"gpsearch response provenance mismatch: expected {EXPECTED_COMMIT}, got {commit}"
        )
    report["subprocess"] = {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0 or report.get("status") != "PASS":
        raise GPSearchProcessError(
            "gpsearch worker rejected the proposal; inspect {} (status={}, exit={})".format(
                response_path, report.get("status"), completed.returncode
            )
        )
    return report

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--runtime-manifest", type=Path, default=DEFAULT_RUNTIME_MANIFEST)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args(argv)
    report = run_gpsearch_step(
        args.request,
        args.response,
        checkpoint=args.checkpoint,
        python_executable=args.python,
        runtime_manifest=args.runtime_manifest,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "method": report["method"],
                "proposal": report["proposal"],
                "response": str(args.response.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

"""Atomic artefact storage and request locking for benchmark seed blocks."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Mapping

import numpy as np

def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def _temporary(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    token = next(tempfile._get_candidate_names())
    return path.with_name(f".{path.name}.{os.getpid()}.{token}.tmp")

def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    tmp = _temporary(path)
    try:
        with tmp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()

def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_bytes(
        Path(path),
        (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"),
    )

class BlockLock(AbstractContextManager):
    """Advisory process lock released automatically if a worker dies."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0)
        self.handle.write(b"0")
        self.handle.flush()
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"benchmark block is already running: {self.path}") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            try:
                if os.name == "nt":
                    import msvcrt

                    self.handle.seek(0)
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            finally:
                self.handle.close()
        return False

class BlockStore:
    def __init__(self, root: Path, request: Mapping[str, object]):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.request = dict(request)
        request_path = self.root / "request.json"
        record = {"request": self.request}
        if request_path.is_file():
            observed = json.loads(request_path.read_text(encoding="utf-8"))
            if observed != record:
                raise RuntimeError(
                    f"output directory belongs to a different request: {self.root}"
                )
        else:
            atomic_write_json(request_path, record)

    def lock(self) -> BlockLock:
        return BlockLock(self.root / ".block.lock")

    def path(self, relative: str | Path) -> Path:
        result = (self.root / relative).resolve()
        try:
            result.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"artifact escapes block root: {relative}") from exc
        return result

    def write_json(self, relative: str | Path, value: object) -> Path:
        path = self.path(relative)
        atomic_write_json(path, value)
        return path

    def read_json(self, relative: str | Path) -> dict[str, Any] | None:
        path = self.path(relative)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def save_npz(self, relative: str | Path, **arrays: object) -> Path:
        path = self.path(relative)
        tmp = _temporary(path)
        try:
            with tmp.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        return path

    def save_npz_once(self, relative: str | Path, **arrays: object) -> tuple[Path, bool]:
        path = self.path(relative)
        if path.is_file():
            saved = self.load_npz(relative)
            if saved is None or set(saved) != set(arrays) or any(
                not np.array_equal(saved[name], np.asarray(value)) for name, value in arrays.items()
            ):
                raise RuntimeError(f"artifact already contains different arrays: {relative}")
            return path, False
        self.save_npz(relative, **arrays)
        return path, True

    def load_npz(self, relative: str | Path) -> dict[str, np.ndarray] | None:
        path = self.path(relative)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as body:
            return {key: body[key] for key in body.files}

    def save_pickle(self, relative: str | Path, value: object) -> Path:
        path = self.path(relative)
        atomic_write_bytes(path, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        return path

    def load_pickle(self, relative: str | Path) -> object | None:
        path = self.path(relative)
        if not path.is_file():
            return None
        from protocol import surrogate as protocol_surrogate

        protocol_surrogate._torch_modules()
        protocol_surrogate._temporal_attention_module_class()
        return pickle.loads(path.read_bytes())

def safe_artifact_name(event_id: str) -> str:
    from urllib.parse import quote

    name = quote(str(event_id), safe="_-.")
    if not name or name in (".", "..") or len(name.encode("utf-8")) > 240:
        raise ValueError("event id cannot be represented as an artifact filename")
    return name

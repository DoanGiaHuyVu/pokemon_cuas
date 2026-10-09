"""Atomic local JSON records: replace only after flushing the complete document."""

import json
import os
import tempfile
from pathlib import Path


def _redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[hidden]")
    if isinstance(value, dict):
        return {_redact(key, secret): _redact(item, secret) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, secret) for item in value]
    return value


def atomic_json(path: Path, record: dict, *, secret: str | None = None) -> None:
    safe_record = _redact(record, secret) if secret else record
    content = json.dumps(safe_record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + "-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        # Also persist the directory entry created by the atomic rename.
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

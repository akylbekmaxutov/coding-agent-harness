from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, trace_dir: str | Path, task_id: str) -> None:
        self.dir = Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self._path = self.dir / f"{stamp}-{_slug(task_id)}.jsonl"
        self._started = time.monotonic()

    def emit(self, event: str, **fields: Any) -> None:
        """Append one line. Never raises: tracing must not fail a run."""
        record = {"t": round(time.monotonic() - self._started, 3), "event": event, **fields}
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass

    def path(self) -> Path:
        return self._path


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text)[:60]
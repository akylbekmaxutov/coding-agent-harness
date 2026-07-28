"""A dict pretending to be a database, so the service layer has something to talk to."""

from __future__ import annotations

from errors import NotFound


class Store:
    def __init__(self, records: dict | None = None) -> None:
        self._records: dict[str, dict] = {k: dict(v) for k, v in (records or {}).items()}

    def get(self, key: str) -> dict:
        if key not in self._records:
            raise NotFound(f"no record {key!r}")
        return dict(self._records[key])

    def put(self, key: str, value: dict) -> dict:
        self._records[key] = dict(value)
        return dict(self._records[key])

    def delete(self, key: str) -> None:
        if key not in self._records:
            raise NotFound(f"no record {key!r}")
        del self._records[key]

    def list(self) -> list[dict]:
        # Sorted so pagination is deterministic; a real store would order by
        # an index instead.
        return [dict(value, id=key) for key, value in sorted(self._records.items())]

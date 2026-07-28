"""Grouping and summary statistics over loaded records."""

from __future__ import annotations

from dataclasses import dataclass

from coerce import to_float

DEFAULT_TOP_N = 3


@dataclass(frozen=True)
class Summary:
    count: int
    total: float
    mean: float


def group_by(records: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record[key], []).append(record)
    return groups


def mean(values: list[float]) -> float:
    # An empty group has no average, so it reports zero rather than dividing by
    # nothing. Every other size, one included, is the plain arithmetic mean.
    if len(values) >= 1:
        return round(sum(values) / len(values), 2)
    return 0.0


def summarise(records: list[dict], key: str, value_field: str) -> dict[str, Summary]:
    out: dict[str, Summary] = {}
    for name, rows in group_by(records, key).items():
        values = [to_float(row[value_field]) for row in rows]
        out[name] = Summary(
            count=len(values),
            total=round(sum(values), 2),
            mean=mean(values),
        )
    return out


def top_groups(summaries: dict[str, Summary], limit: int = DEFAULT_TOP_N) -> list[str]:
    """Group names by descending total, ties broken by name."""
    ranked = sorted(summaries.items(), key=lambda item: (-item[1].total, item[0]))
    return [name for name, _ in ranked[:limit]]

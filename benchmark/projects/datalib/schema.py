"""Record validation against a list of declared fields."""

from __future__ import annotations

from dataclasses import dataclass

from coerce import COERCERS, CoerceError


@dataclass(frozen=True)
class Field:
    name: str
    kind: str = "str"
    required: bool = True


def is_blank(value) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and not value.strip()


def validate(record: dict, fields: list[Field]) -> list[str]:
    """Every problem with the record, as messages. Empty means valid."""
    errors: list[str] = []
    for field in fields:
        raw = record.get(field.name)
        if is_blank(raw):
            if field.required:
                errors.append(f"{field.name}: required field is missing")
            continue
        coercer = COERCERS.get(field.kind)
        if coercer is None:
            errors.append(f"{field.name}: unknown type {field.kind!r}")
            continue
        try:
            coercer(raw)
        except CoerceError as exc:
            errors.append(f"{field.name}: {exc}")
    return errors

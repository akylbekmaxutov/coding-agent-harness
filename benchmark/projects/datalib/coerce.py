"""Cell strings to Python values. Every failure names the field it came from."""

from __future__ import annotations


class CoerceError(ValueError):
    """A cell could not be read as the declared type."""


TRUE_WORDS = frozenset({"true", "t", "yes", "y", "1", "on"})
FALSE_WORDS = frozenset({"false", "f", "no", "n", "0", "off"})


def to_str(value) -> str:
    return "" if value is None else str(value).strip()


def to_int(value) -> int:
    text = to_str(value)
    try:
        return int(text)
    except ValueError:
        raise CoerceError(f"not an integer: {text!r}") from None


def to_float(value) -> float:
    text = to_str(value)
    try:
        return float(text)
    except ValueError:
        raise CoerceError(f"not a number: {text!r}") from None


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = to_str(value).lower()
    if text in TRUE_WORDS:
        return True
    if text in FALSE_WORDS:
        return False
    raise CoerceError(f"not a boolean: {text!r}")


COERCERS = {"str": to_str, "int": to_int, "float": to_float, "bool": to_bool}


def coerce_record(record: dict, spec: dict[str, str]) -> dict:
    """Apply `spec` (field -> type name) to `record`, leaving other keys alone."""
    out = dict(record)
    for name, kind in spec.items():
        if name not in record:
            continue
        coercer = COERCERS.get(kind)
        if coercer is None:
            raise CoerceError(f"{name}: unknown type {kind!r}")
        try:
            out[name] = coercer(record[name])
        except CoerceError as exc:
            raise CoerceError(f"{name}: {exc}") from None
    return out

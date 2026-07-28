"""Recover a JSON argument object from a model reply that may be fenced or prosey."""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.DOTALL)


class ArgumentError(ValueError):
    """The reply did not contain a usable JSON object."""


def strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group("body").strip() if match else text.strip()


def looks_like_object(chunk: str) -> bool:
    # Both ends, not either: a sentence that happens to open a brace is prose,
    # and calling it malformed JSON sends the caller looking for a syntax error
    # that is not there.
    chunk = chunk.strip()
    return chunk.startswith("{") and chunk.endswith("}")


def extract_object(text: str) -> dict:
    chunk = strip_fences(text or "")
    if not looks_like_object(chunk):
        raise ArgumentError(f"no JSON object found in {chunk[:40]!r}")
    try:
        return json.loads(chunk)
    except json.JSONDecodeError as exc:
        raise ArgumentError(f"invalid JSON: {exc.msg}") from None

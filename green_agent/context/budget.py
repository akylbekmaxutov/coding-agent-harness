from __future__ import annotations

import ast

from ..types import CodeSlice


def estimate_tokens(text: str) -> int:
    """Characters over four.

    An upper-bound estimate is all the ceiling needs, and a real tokenizer
    would couple the harness to one model family for no benefit here.
    """
    return max(1, (len(text) + 3) // 4)


def fit(slices: list[CodeSlice], budget_tokens: int) -> tuple[list[CodeSlice], bool]:
    """Return (kept, degraded). Order is preserved; degraded means starved."""
    kept: list[CodeSlice] = []
    used = 0
    for sl in slices:
        cost = sl.token_estimate()
        if used + cost > budget_tokens:
            break
        kept.append(sl)
        used += cost

    if kept:
        return kept, False
    if not slices:
        return [], False
    return [signature_only(slices[0])], True


def signature_only(sl: CodeSlice) -> CodeSlice:
    """Definition line(s) plus an elision marker, still valid to read."""
    lines = sl.source.splitlines()
    header = lines[:1]
    try:
        node = ast.parse(_dedent(sl.source)).body[0]
        header = lines[: (node.body[0].lineno - 1) or 1]
    except (SyntaxError, IndexError, AttributeError):
        pass
    body = "\n".join(header + ["    ...  # body elided: context budget exceeded"])
    return CodeSlice(
        path=sl.path,
        symbol=sl.symbol,
        start_line=sl.start_line,
        end_line=sl.end_line,
        source=body,
        reason=sl.reason + " (signature only)",
    )


def _dedent(source: str) -> str:
    lines = source.splitlines()
    pad = len(lines[0]) - len(lines[0].lstrip()) if lines else 0
    return "\n".join(line[pad:] if len(line) >= pad else line for line in lines)
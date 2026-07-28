from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import Config
from ..runtime.workspace import Workspace
from ..types import TestResult, ToolCall, ToolResult
from . import apply_patch, finish, read_file, run_tests

_MODULES = (read_file, apply_patch, run_tests, finish)
_TOOLS: dict[str, tuple[dict[str, Any], Callable[..., ToolResult]]] = {
    m.SCHEMA["name"]: (m.SCHEMA, m.handle) for m in _MODULES
}


@dataclass
class ToolContext:
    """Everything a handler may touch, passed explicitly rather than imported.

    Explicit context is what lets the benchmark run tasks in parallel and lets
    the trace attribute every side effect to a specific run.
    """

    workspace: Workspace
    config: Config
    target_test: str
    last_result: TestResult | None = None
    finish_summary: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


def schemas() -> list[dict[str, Any]]:
    return [schema for schema, _ in _TOOLS.values()]


def dispatch(call: ToolCall, ctx: ToolContext) -> ToolResult:
    """Invoke a tool, converting every failure into a ToolResult.

    The loop must never crash on a bad call: an unknown name or a missing
    argument is information for the model, so it comes back as ok=False with a
    message written for the model to read.
    """
    entry = _TOOLS.get(call.name)
    if entry is None:
        return ToolResult(
            call_id=call.call_id,
            ok=False,
            content=f"No tool named {call.name!r}. Available: {', '.join(_TOOLS)}.",
        )

    _, handler = entry
    accepted = set(inspect.signature(handler).parameters) - {"ctx"}
    unknown = set(call.arguments) - accepted
    arguments = {k: v for k, v in call.arguments.items() if k in accepted}

    try:
        result = handler(**arguments, ctx=ctx)
    except TypeError as exc:
        return ToolResult(
            call_id=call.call_id,
            ok=False,
            content=f"Bad arguments for {call.name}: {exc}. Expected: {sorted(accepted)}.",
        )
    except Exception as exc:  # a tool bug must not kill the run
        return ToolResult(
            call_id=call.call_id,
            ok=False,
            content=f"{call.name} failed: {type(exc).__name__}: {exc}",
            meta={"unexpected": True},
        )

    result.call_id = call.call_id
    if unknown:
        result.meta["ignored_arguments"] = sorted(unknown)
    return result
"""run_tests — the verifier, and the only source of truth in the system."""

from __future__ import annotations

from ..runtime import pytest_runner
from ..types import ToolResult

SCHEMA = {
    "name": "run_tests",
    "description": "Run the target test. Pass node_id only to run a different test.",
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "string",
                "description": "pytest node id, e.g. tests/test_cart.py::test_total",
            }
        },
    },
}


def handle(node_id: str | None = None, *, ctx) -> ToolResult:
    # Omitting node_id means the target test, not the whole suite: it is the
    # fast inner loop, and it is the safe reading when a model omits the
    # argument. The regression sweep over the full suite is the harness's job
    # at finish-verification, not something the agent decides to run.
    target = node_id or ctx.target_test
    result = pytest_runner.run(
        ctx.workspace.root, target, timeout_s=ctx.config.guards.pytest_timeout_s
    )
    ctx.last_result = result

    return ToolResult(
        call_id="",
        ok=True,           # a failing test is a successful tool call
        content=summarise(result, target, ctx.config.context.max_raw_output_chars),
        meta={
            "node_id": target,
            "passed": result.passed,
            "timed_out": result.timed_out,
            "collect_error": result.collect_error,
            "fingerprints": [f.fingerprint() for f in result.failures],
        },
    )


def summarise(result, target: str, max_chars: int) -> str:
    """What the model sees. The full output goes to the trace only."""
    if result.timed_out:
        return f"{target} timed out after {result.duration_s}s. The code probably loops forever."
    if result.passed:
        return f"{target} PASSED in {result.duration_s}s."
    if result.collect_error:
        return (
            "Collection failed - the file cannot be imported, so the last edit "
            "probably broke the syntax:\n" + result.raw_output[-max_chars:]
        )

    lines = [f"{target} FAILED in {result.duration_s}s."]
    for failure in result.failures:
        lines.append(f"\n{failure.test_id}\n  {failure.exc_type}: {failure.message}")
        for frame in failure.frames:
            marker = "  " if frame.in_repo else "  (external) "
            lines.append(f"{marker}{frame.path}:{frame.lineno} in {frame.function}")
    return "\n".join(lines)[:max_chars]
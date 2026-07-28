from __future__ import annotations

from ..types import ToolResult

SCHEMA = {
    "name": "finish",
    "description": "Declare the fix complete. The harness will verify before accepting.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Root cause and fix, 1-3 sentences."}
        },
        "required": ["summary"],
    },
}


def handle(summary: str = "", *, ctx) -> ToolResult:
    ctx.finish_summary = summary.strip()
    return ToolResult(
        call_id="",
        ok=True,
        content="Claim recorded. The harness is re-running the tests to verify it.",
        meta={"summary": ctx.finish_summary},
    )
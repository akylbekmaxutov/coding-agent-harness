from __future__ import annotations

from ..types import ToolResult

MAX_LINES = 200

SCHEMA = {
    "name": "read_file",
    "description": "Read a range of lines from a source file in the repository.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Repo-relative path."},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
    },
}


def handle(path: str, start_line: int = 1, end_line: int | None = None, *, ctx) -> ToolResult:
    from ..runtime.workspace import PathEscape

    try:
        target = ctx.workspace.resolve(path)
    except PathEscape as exc:
        return ToolResult(call_id="", ok=False, content=str(exc))

    if not target.is_file():
        return ToolResult(call_id="", ok=False, content=f"No such file: {path}")

    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, int(start_line))
    end = len(lines) if end_line is None else min(len(lines), int(end_line))
    if end < start:
        return ToolResult(call_id="", ok=False, content="end_line is before start_line.")

    truncated = end - start + 1 > MAX_LINES
    end = min(end, start + MAX_LINES - 1)

    # Absolute line numbers, because apply_patch needs correct hunk anchors and
    # a model that cannot see them will produce diffs that do not apply.
    body = "\n".join(f"{n:>5} | {lines[n - 1]}" for n in range(start, end + 1))
    note = f"\n[showing lines {start}-{end} of {len(lines)}]" if truncated else ""
    return ToolResult(call_id="", ok=True, content=f"{path}\n{body}{note}",
                      meta={"path": path, "start": start, "end": end})
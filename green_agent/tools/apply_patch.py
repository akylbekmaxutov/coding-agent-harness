from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ..runtime.workspace import PathEscape
from ..types import ToolResult

SCHEMA = {
    "name": "apply_patch",
    "description": (
        "Apply a unified diff to source files. Test files are forbidden. "
        "Use a/ and b/ prefixes and correct line numbers."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "diff": {"type": "string", "description": "Unified diff with @@ hunks."}
        },
        "required": ["diff"],
    },
}

_PATH_RE = re.compile(r"^(?:---|\+\+\+)\s+(?:[ab]/)?(?P<path>[^\t\n]+)", re.MULTILINE)

# --recount tolerates wrong @@ line counts, which models get wrong constantly
# and which say nothing about whether the edit itself is correct.
_ATTEMPTS = (
    ["--recount"],
    ["--recount", "--unidiff-zero"],
    ["--recount", "-p0"],
)


def handle(diff: str = "", *, ctx) -> ToolResult:
    if not diff.strip():
        return _reject("Empty diff.")

    diff = _normalise(diff)
    paths = changed_paths(diff)
    if not paths:
        return _reject(
            "Could not find any file headers. A unified diff needs --- a/<path> "
            "and +++ b/<path> lines followed by @@ hunks."
        )

    guard = _check_guards(paths, diff, ctx)
    if guard:
        return _reject(guard)

    root = ctx.workspace.root
    errors: list[str] = []
    for flags in _ATTEMPTS:
        check = _git_apply(root, diff, flags + ["--check"])
        if check.returncode != 0:
            errors.append(check.stderr.strip())
            continue
        applied = _git_apply(root, diff, flags)
        if applied.returncode == 0:
            return ToolResult(
                call_id="",
                ok=True,
                content=f"Patch applied to {', '.join(paths)}.",
                meta={"paths": paths, "flags": flags},
            )
        errors.append(applied.stderr.strip())

    return _reject(
        "The patch did not apply: "
        + (errors[0] or "unknown git error")
        + " Re-read the file to get exact context lines, then send a corrected diff."
    )


def changed_paths(diff: str) -> list[str]:
    seen: list[str] = []
    for match in _PATH_RE.finditer(diff):
        path = match.group("path").strip()
        if path in ("/dev/null", "") or path in seen:
            continue
        seen.append(path)
    return seen


def changed_line_count(diff: str) -> int:
    return sum(
        1
        for line in diff.splitlines()
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith(("+++", "---"))
    )


def _check_guards(paths: list[str], diff: str, ctx) -> str | None:
    guards = ctx.config.guards
    for path in paths:
        try:
            ctx.workspace.resolve(path)
        except PathEscape:
            return f"Path {path!r} is outside the workspace."
        if guards.forbid_test_file_edits and ctx.workspace.is_test_file(path):
            # A test-verified agent will otherwise patch the assertion instead
            # of the bug, and every run would "succeed".
            return (
                f"{path} is a test file and cannot be edited. Fix the source "
                "code so the existing test passes."
            )
    n = changed_line_count(diff)
    if n > guards.max_patch_lines:
        return (
            f"Patch changes {n} lines, over the {guards.max_patch_lines} line limit. "
            "Make the smallest edit that fixes the root cause."
        )
    return None


def _normalise(diff: str) -> str:
    """Trim fences and guarantee the trailing newline git insists on."""
    text = diff.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text + "\n"


def _git_apply(root: Path, diff: str, flags: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), "apply", *flags, "-"],
        input=diff,
        capture_output=True,
        text=True,
    )


def _reject(reason: str) -> ToolResult:
    return ToolResult(call_id="", ok=False, content=reason, meta={"rejected": True})
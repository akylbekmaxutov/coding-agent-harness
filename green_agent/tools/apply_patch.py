from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ..runtime.workspace import PathEscape
from ..types import ToolResult

SCHEMA = {
    "name": "apply_patch",
    "description": (
        "Apply a plain unified diff to source files. Test files are forbidden. "
        "No wrapper or envelope: the argument starts with '--- a/<path>'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "diff": {
                "type": "string",
                "description": (
                    "Unified diff. Must be exactly: '--- a/<path>' newline "
                    "'+++ b/<path>' newline '@@ -<line>,<count> +<line>,<count> @@' "
                    "newline, then body lines prefixed with a space, '-' or '+'."
                ),
            }
        },
        "required": ["diff"],
    },
}

_PATH_RE = re.compile(r"^(?:---|\+\+\+)\s+(?:[ab]/)?(?P<path>[^\t\n]+)", re.MULTILINE)
_ENVELOPE_FILE_RE = re.compile(r"^\*\*\*\s+(?:Update|Add)\s+File:\s*(?P<path>.+?)\s*$")
_ENVELOPE_MARKERS = ("*** Begin Patch", "*** End Patch", "*** Update File:", "*** Add File:")
_NUMBERED_HUNK_RE = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@")
_TARGET_RE = re.compile(r"^\+\+\+\s+(?:[ab]/)?(?P<path>[^\t\n]+)")

# git's own words for "this is not a diff" as opposed to "this diff does not
# match the file". The two need opposite advice, and telling a model to re-read
# context lines when the format is wrong sends it back to a file that was fine.
_FORMAT_ERRORS = (
    "without header",
    "corrupt patch",
    "unrecognized input",
    "no valid patches",
    "cannot be read",
)

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
    diff = renumber_bare_hunks(diff, root)
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

    return _reject("The patch did not apply: " + (errors[0] or "unknown git error")
                   + " " + _advice(errors[0] if errors else ""))


def renumber_bare_hunks(diff: str, root: Path) -> str:
    """Give line numbers to `@@` headers that have none.

    The apply_patch envelope marks a hunk with a bare `@@` and leaves the ranges
    out. git cannot parse that with or without --recount, so a semantically
    perfect one-line edit is refused with "No valid patches in input" -- a live
    run sent the same correct change seven times and never landed it.

    The ranges are recoverable: the hunk's own context and removed lines say
    where it belongs, so we find that block in the file and write the header git
    wants. A hunk we cannot place is left alone for git to reject in its own
    words, because guessing a location is how you corrupt a file.
    """
    lines = diff.splitlines()
    out: list[str] = []
    target: Path | None = None
    index = 0

    while index < len(lines):
        line = lines[index]
        match = _TARGET_RE.match(line)
        if match:
            path = match.group("path").strip()
            target = None if path == "/dev/null" else root / path
            out.append(line)
            index += 1
            continue

        if not line.startswith("@@") or _NUMBERED_HUNK_RE.match(line):
            out.append(line)
            index += 1
            continue

        body: list[str] = []
        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].startswith(("@@", "--- ", "+++ ", "*** ")):
            body.append(lines[cursor])
            cursor += 1

        out.append(_hunk_header(body, target))
        out.extend(body)
        index = cursor

    rebuilt = "\n".join(out)
    # git calls a diff with no trailing newline a corrupt patch, so the one
    # _normalise added must survive the round trip.
    return rebuilt + "\n" if diff.endswith("\n") else rebuilt


def _hunk_header(body: list[str], target: Path | None) -> str:
    """`@@ -start,old +start,new @@` for one hunk, located in the real file."""
    old = [line[1:] for line in body if (line[:1] or " ") in (" ", "-")]
    new_count = sum(1 for line in body if (line[:1] or " ") in (" ", "+"))
    start = _locate(target, old) if target is not None else 1
    if start is None:
        return "@@"
    return f"@@ -{start},{len(old)} +{start},{new_count} @@"


def _locate(target: Path, block: list[str]) -> int | None:
    """1-based line at which `block` occurs in `target`, or None."""
    if not block:
        return 1
    try:
        haystack = target.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    size = len(block)
    for start in range(len(haystack) - size + 1):
        if haystack[start:start + size] == block:
            return start + 1
    return None


def _advice(error: str) -> str:
    """What to do about it. Wrong advice costs a turn and teaches nothing."""
    if any(marker in error.lower() for marker in _FORMAT_ERRORS):
        return (
            "That is a format problem, not a context problem, so the file is "
            "fine. Send a plain unified diff and nothing else:\n"
            "--- a/<path>\n+++ b/<path>\n@@ -<line>,<count> +<line>,<count> @@\n"
            " unchanged line\n-removed line\n+added line"
        )
    return "Re-read the file to get exact context lines, then send a corrected diff."


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
    """Trim fences, unwrap the envelope, guarantee git's trailing newline."""
    text = diff.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return unwrap_envelope(text).rstrip("\n") + "\n"


def unwrap_envelope(text: str) -> str:
    """Rewrite a ``*** Begin Patch`` envelope as a plain unified diff.

    This tool is named apply_patch, which is also the name of a widely trained
    tool whose argument format is this envelope, so models reach for it without
    being asked. What it wraps is ordinary unified diff -- only the file header
    differs -- and rejecting the whole call over the header spends a turn and
    teaches the model nothing it can act on.
    """
    if not any(marker in text for marker in _ENVELOPE_MARKERS):
        return text
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        # Independently, not as a pair: replies routinely carry the closing
        # marker with no opening one, and a single stray line is enough for git
        # to report "No valid patches in input" about an otherwise fine diff.
        if stripped in ("*** Begin Patch", "*** End Patch"):
            continue
        header = _ENVELOPE_FILE_RE.match(stripped)
        if header:
            path = header.group("path")
            path = path[2:] if path.startswith(("a/", "b/")) else path
            out += [f"--- a/{path}", f"+++ b/{path}"]
            continue
        # Some replies carry both the envelope header and real ---/+++ lines.
        # Ours are already in, so drop the duplicates rather than emit four.
        if out and out[-1].startswith("+++ b/") and stripped.startswith(("--- ", "+++ ")):
            continue
        out.append(line)
    return "\n".join(out)


def _git_apply(root: Path, diff: str, flags: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), "apply", *flags, "-"],
        input=diff,
        capture_output=True,
        text=True,
    )


def _reject(reason: str) -> ToolResult:
    return ToolResult(call_id="", ok=False, content=reason, meta={"rejected": True})
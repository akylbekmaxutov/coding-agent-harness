from __future__ import annotations

import re
from pathlib import Path

from ..types import Failure, Frame

_SECTION_RE = re.compile(r"^_{3,}\s+(?P<title>.+?)\s+_{3,}$")
_BANNER_RE = re.compile(r"^=+\s+(?P<name>[A-Za-z ]+?)\s+=+$")
_FRAME_RE = re.compile(r"^(?P<path>\S.*?):(?P<lineno>\d+): in (?P<func>.+)$")
_EXC_RE = re.compile(r"^E\s+(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit))\b:?\s*(?P<msg>.*)$")
_SUMMARY_RE = re.compile(r"^(?:FAILED|ERROR)\s+(?P<node>\S+)(?:\s+-\s+(?P<msg>.*))?$")


def parse_failures(output: str, repo_root: str | Path) -> tuple[Failure, ...]:
    """Extract every failing test from a pytest report."""
    root = Path(repo_root).resolve()
    node_ids = _summary_node_ids(output)
    sections = _split_sections(output)

    failures: list[Failure] = []
    for index, (title, body) in enumerate(sections):
        frames = _parse_frames(body, root)
        exc_type, message = _parse_exception(body)
        test_id = _match_node_id(title, node_ids, index)
        failures.append(
            Failure(test_id=test_id, exc_type=exc_type, message=message, frames=frames)
        )
    return tuple(failures)


def _split_sections(output: str) -> list[tuple[str, list[str]]]:
    """Blocks under the FAILURES and ERRORS banners, as (title, lines)."""
    sections: list[tuple[str, list[str]]] = []
    collecting = False
    title: str | None = None
    body: list[str] = []

    for line in output.splitlines():
        banner = _BANNER_RE.match(line)
        if banner:
            name = banner.group("name").strip().lower()
            if title is not None:
                sections.append((title, body))
                title, body = None, []
            collecting = name in {"failures", "errors"}
            continue

        if not collecting:
            continue

        section = _SECTION_RE.match(line)
        if section:
            if title is not None:
                sections.append((title, body))
            title, body = section.group("title"), []
            continue

        if title is not None:
            body.append(line)

    if title is not None:
        sections.append((title, body))
    return sections


def _parse_frames(body: list[str], root: Path) -> tuple[Frame, ...]:
    frames: list[Frame] = []
    for line in body:
        m = _FRAME_RE.match(line)
        if not m:
            continue
        raw_path = m.group("path")
        rel, in_repo = _classify_path(raw_path, root)
        frames.append(
            Frame(
                path=rel,
                lineno=int(m.group("lineno")),
                function=m.group("func").strip(),
                in_repo=in_repo,
            )
        )
    return tuple(frames)


def _classify_path(raw_path: str, root: Path) -> tuple[str, bool]:
    """Repo-relative path plus whether it belongs to the target project.

    Frames from stdlib, site-packages and pytest's own internals are marked
    external so the slicer never wastes context on code the agent cannot edit.
    """
    if raw_path.startswith("<"):          # <frozen importlib._bootstrap>
        return raw_path, False
    candidate = Path(raw_path)
    absolute = candidate if candidate.is_absolute() else (root / candidate)
    try:
        resolved = absolute.resolve()
    except OSError:
        return raw_path, False
    if resolved.is_relative_to(root) and resolved.is_file():
        return str(resolved.relative_to(root)), True
    return raw_path, False


def _parse_exception(body: list[str]) -> tuple[str, str]:
    """Exception type and message from the `E ...` lines.

    Plain assertion failures print no exception name at --tb=short, so an
    E-line that starts with `assert` is reported as AssertionError.
    """
    e_lines = [line[1:].strip() for line in body if line.startswith("E ")]
    if not e_lines:
        return "", ""
    for line in e_lines:
        m = _EXC_RE.match("E   " + line)
        if m:
            return m.group("type"), m.group("msg").strip()
    first = e_lines[0]
    if first.startswith("assert"):
        return "AssertionError", first
    return "", first


def _summary_node_ids(output: str) -> list[str]:
    ids: list[str] = []
    for line in output.splitlines():
        m = _SUMMARY_RE.match(line.strip())
        if m:
            ids.append(m.group("node"))
    return ids


def _match_node_id(title: str, node_ids: list[str], index: int) -> str:
    """Pair a section title with its node id.

    Titles drop the file path and use dots for classes ("TestCart.test_x"),
    so we match on the node id's tail and fall back to positional order --
    sections and summary lines are emitted in the same sequence.
    """
    tail = title.replace("ERROR at setup of ", "").replace("ERROR collecting ", "").strip()
    wanted = tail.replace(".", "::")
    for node in node_ids:
        if node.endswith("::" + wanted) or node == wanted or node.endswith(tail):
            return node
    return node_ids[index] if index < len(node_ids) else tail
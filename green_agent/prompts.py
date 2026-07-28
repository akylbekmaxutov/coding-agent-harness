from __future__ import annotations

from typing import Any

from .types import CodeSlice, Failure, Step

SYSTEM_PROMPT = """\
You are a debugging agent operating inside an automated harness.

Goal: make the target failing test pass by editing SOURCE code only.

Rules:
- Emit exactly one tool call per turn.
- Start every turn with `HYPOTHESIS: <one sentence>` before the tool call.
- You may not edit test files; such patches are rejected.
- Prefer the smallest edit that fixes the root cause. Do not refactor.
- apply_patch takes a plain unified diff and nothing else -- no envelope, no
  prose, no fences. Line numbers in the @@ header are required:

      --- a/module.py
      +++ b/module.py
      @@ -12,4 +12,4 @@
       def total(self):
      -    return self.a + self.b
      +    return self.a * self.b
       # trailing context
- The code below is usually all you need. Use read_file only for a function
  that is not shown, and never twice with the same arguments.
- Once you can name the wrong line, patch it. Do not re-read to be sure.
- After a patch, call run_tests to check it.
- Call finish only once run_tests reports the target test passing.\
"""


def render_task(repo_name: str, target_test: str) -> str:
    return (
        f"Repository: {repo_name}\n"
        f"Target test: {target_test}\n"
        "Done when: this test passes, no test file was modified, and the rest "
        "of the suite still passes."
    )


def render_failure(failure: Failure | None, raw_tail: str) -> str:
    if failure is None:
        # No failure means the last run was green. Saying so beats printing raw
        # output and leaving the model to work out what it is looking at.
        return f"There is no current failure. Latest test output:\n{raw_tail}"
    lines = [
        "Current failure:",
        f"  {failure.test_id}",
        f"  {failure.exc_type}: {failure.message}",
        "  Traceback (project frames only):",
    ]
    for frame in failure.frames:
        if frame.in_repo:
            lines.append(f"    {frame.path}:{frame.lineno} in {frame.function}")
    return "\n".join(lines)


def render_slices(slices: list[CodeSlice]) -> str:
    """Code grouped by file, with absolute line numbers.

    The numbers are not decoration: apply_patch works on unified diffs, and a
    model that cannot see line numbers writes diffs that do not apply.
    """
    if not slices:
        return "No source could be extracted; use read_file."
    blocks: list[str] = ["Relevant code:"]
    for sl in slices:
        blocks.append(f"\n{sl.path}  ({sl.symbol}, lines {sl.start_line}-{sl.end_line})")
        blocks.append(f"  [{sl.reason}]")
        width = len(str(sl.end_line))
        for offset, line in enumerate(sl.source.splitlines()):
            blocks.append(f"{sl.start_line + offset:>{width}} | {line}")
    return "\n".join(blocks)


def render_history(steps: list[Step]) -> str:
    """One line per past step: the anti-loop memory, ~30 tokens each."""
    if not steps:
        return ""
    lines = ["What you have already tried:"]
    for step in steps:
        action = step.call.name if step.call else "no tool call"
        outcome = "-"
        if step.result is not None:
            outcome = step.result.content.splitlines()[0][:110] if step.result.content else "-"
            outcome = ("ok: " if step.result.ok else "rejected: ") + outcome
        hypothesis = step.hypothesis or "(no hypothesis given)"
        lines.append(f"  #{step.index} {hypothesis} -> {action}: {outcome}")
    return "\n".join(lines)


def build_messages(
    repo_name: str,
    target_test: str,
    failure: Failure | None,
    raw_tail: str,
    slices: list[CodeSlice],
    history: list[Step],
    directive: str | None = None,
) -> list[dict[str, Any]]:
    parts = [
        render_task(repo_name, target_test),
        "",
        render_failure(failure, raw_tail),
    ]
    # With no failure there is nothing to slice, and the empty-slices text tells
    # the model to go and read a file -- the opposite of the finish directive it
    # is being given in the same prompt.
    if slices or failure is not None:
        parts += ["", render_slices(slices)]
    hist = render_history(history)
    if hist:
        parts += ["", hist]
    if directive:
        parts += ["", directive]
    parts += ["", "Take one action now."]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]
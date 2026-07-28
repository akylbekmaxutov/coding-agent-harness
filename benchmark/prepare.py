"""Materialise a task: a throwaway git repo with the bug applied at HEAD.

The agent's own Workspace copies whatever it is handed, so what it gets is a
repo, not the project directory itself. Nothing here ever writes inside
`benchmark/projects/` -- the green sources are the ground truth and stay green.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from green_agent.runtime import pytest_runner

from .catalog import Task

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".git")
_GIT_ID = ("-c", "user.name=benchmark", "-c", "user.email=benchmark@localhost")


class TaskError(RuntimeError):
    """The task itself is broken: the patch will not apply, or the repo is odd."""


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise TaskError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


@contextmanager
def checkout(task: Task, apply_bug: bool = True) -> Iterator[Path]:
    """A temp git repo holding the project, optionally with the bug at HEAD.

    HEAD has to carry the bug, not the working tree: the agent's Workspace
    creates a detached worktree from HEAD, and an uncommitted patch would be
    left behind -- the agent would be handed a green repo and a task that says
    it is red.
    """
    root = Path(tempfile.mkdtemp(prefix=f"bench-{task.task_id}-"))
    try:
        shutil.rmtree(root)
        shutil.copytree(task.project_dir, root, ignore=_IGNORE)
        _git(root, "init", "-q")
        _git(root, "add", "-A")
        _git(root, *_GIT_ID, "commit", "-q", "-m", "green")
        if apply_bug:
            proc = subprocess.run(
                ["git", "-C", str(root), "apply", "--whitespace=nowarn", "-"],
                input=task.patch_text(),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise TaskError(
                    f"{task.task_id}: bug.patch does not apply to "
                    f"{task.project}: {proc.stderr.strip()}"
                )
            _git(root, "add", "-A")
            _git(root, *_GIT_ID, "commit", "-q", "-m", "bug")
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


@dataclass
class TaskCheck:
    task_id: str
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


def verify(task: Task, timeout_s: int = 60) -> TaskCheck:
    """Assert the one property every task must have, before any agent runs.

    A task is only evidence if the bug is observable: green without the patch,
    and with the patch exactly one named test failing. A task that breaks two
    tests measures something else; a task that breaks none is not a task.
    """
    problems: list[str] = []

    with checkout(task, apply_bug=False) as green:
        clean = pytest_runner.run(green, None, timeout_s)
        if not clean.passed:
            failing = ", ".join(f.test_id for f in clean.failures) or "unknown"
            problems.append(f"project is not green before the bug (failing: {failing})")

    with checkout(task, apply_bug=True) as buggy:
        broken = pytest_runner.run(buggy, None, timeout_s)
        if broken.collect_error:
            problems.append("suite does not even collect with the bug applied")
        elif broken.passed:
            problems.append("bug is not observable: the whole suite still passes")
        else:
            failed = [failure.test_id for failure in broken.failures]
            if failed != [task.target_test]:
                problems.append(
                    f"expected exactly [{task.target_test}] to fail, got {failed}"
                )

        # A node id that no longer resolves silently turns every run into an
        # "already green" refusal, which reads as a harness bug.
        targeted = pytest_runner.run(buggy, task.target_test, timeout_s)
        if targeted.collect_error:
            problems.append(f"target test {task.target_test} does not resolve")
        elif targeted.passed:
            problems.append(f"target test {task.target_test} passes in isolation")

    return TaskCheck(task.task_id, problems)

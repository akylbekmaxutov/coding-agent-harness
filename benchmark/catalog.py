"""The task catalogue: what the bugs are, and how to read them back off disk.

`BUGS` is the declarative source of truth. `write_tasks()` turns each entry into
`tasks/<task_id>/{task.json,bug.patch}` by diffing the *real* project file, so a
patch can never drift out of context with the code it patches. `load_tasks()` is
what the runner, the report and the gate use; they never see `BUGS`.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = BENCHMARK_DIR / "projects"
TASKS_DIR = BENCHMARK_DIR / "tasks"


@dataclass(frozen=True)
class BugSpec:
    """One injected bug, as a text substitution against a green project."""

    task_id: str
    project: str
    path: str                  # project-relative file the bug lives in
    target_test: str           # the one test that must fail
    shape: str                 # off-by-one, inverted boolean, ...
    traceback: str             # "at the bug" | "away from the bug"
    note: str
    old: str
    new: str

    def diff(self) -> str:
        source = PROJECTS_DIR / self.project / self.path
        before = source.read_text(encoding="utf-8")
        if self.old not in before:
            raise ValueError(f"{self.task_id}: anchor text not found in {self.path}")
        if before.count(self.old) != 1:
            raise ValueError(f"{self.task_id}: anchor text is not unique in {self.path}")
        after = before.replace(self.old, self.new)
        if after == before:
            raise ValueError(f"{self.task_id}: substitution changed nothing")
        return "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile=f"a/{self.path}",
                tofile=f"b/{self.path}",
                n=3,
            )
        )


@dataclass(frozen=True)
class Task:
    """A task as the runner sees it: metadata plus a patch that applies."""

    task_id: str
    project: str
    target_test: str
    bug: str
    note: str = ""
    shape: str = ""
    traceback: str = ""
    directory: Path = field(default=TASKS_DIR, compare=False)

    @property
    def project_dir(self) -> Path:
        return PROJECTS_DIR / self.project

    @property
    def patch_path(self) -> Path:
        return self.directory / self.bug

    def patch_text(self) -> str:
        return self.patch_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the bugs
# ---------------------------------------------------------------------------

BUGS: tuple[BugSpec, ...] = (
    BugSpec(
        task_id="backend-pagination-off-by-one",
        project="backend",
        path="pagination.py",
        target_test="tests/test_pagination.py::test_last_page_is_not_empty",
        shape="off-by-one",
        traceback="away from the bug",
        note=(
            "Ceiling division replaced by floor-plus-one, so a total that divides "
            "exactly by the page size gains an empty page on the end."
        ),
        old="    return (total + size - 1) // size",
        new="    return total // size + 1",
    ),
    BugSpec(
        task_id="backend-unknown-action-no-early-return",
        project="backend",
        path="service.py",
        target_test="tests/test_service.py::test_unknown_action_is_a_bad_request",
        shape="missing early return",
        traceback="at the bug",
        note=(
            "The guard that answers an unroutable request is gone, so an unknown "
            "action calls None and raises TypeError past the AppError handler."
        ),
        old=(
            "    # An unroutable request is the caller's mistake, and it has to be answered\n"
            "    # here: past this point there is no handler to raise anything.\n"
            "    if handler is None:\n"
            "        unknown = BadRequest(f\"unknown action {request.get('action')!r}\")\n"
            "        return {\"status\": status_for(unknown), \"body\": error_body(unknown)}\n"
        ),
        new="",
    ),
    BugSpec(
        task_id="datalib-single-row-mean",
        project="datalib",
        path="aggregate.py",
        target_test="tests/test_aggregate.py::test_group_of_one_averages_to_its_own_value",
        shape="wrong comparison operator",
        traceback="away from the bug",
        note=(
            "The empty-group guard now excludes groups of one as well, so a single "
            "row averages to 0.0 instead of its own value."
        ),
        old="    if len(values) >= 1:",
        new="    if len(values) > 1:",
    ),
    BugSpec(
        task_id="datalib-top-n-default",
        project="datalib",
        path="aggregate.py",
        target_test="tests/test_aggregate.py::test_top_groups_defaults_to_three",
        shape="wrong default value",
        traceback="away from the bug",
        note=(
            "The default limit lives in a module constant, not in the signature "
            "the slicer emits, so the wrong value is invisible in sliced context."
        ),
        old="DEFAULT_TOP_N = 3",
        new="DEFAULT_TOP_N = 1",
    ),
    BugSpec(
        task_id="agentlib-backoff-swapped-args",
        project="agentlib",
        path="retry.py",
        target_test="tests/test_retry.py::test_delay_grows_geometrically",
        shape="swapped argument order",
        traceback="away from the bug",
        note=(
            "base and attempt handed to backoff_delay the wrong way round, so the "
            "first retry waits zero seconds and the curve is wrong thereafter."
        ),
        old="    return backoff_delay(policy.base_delay, attempt, policy.factor, policy.max_delay)",
        new="    return backoff_delay(attempt, policy.base_delay, policy.factor, policy.max_delay)",
    ),
    BugSpec(
        task_id="agentlib-loose-brace-check",
        project="agentlib",
        path="jsonargs.py",
        target_test="tests/test_jsonargs.py::test_prose_that_opens_a_brace_is_not_arguments",
        shape="inverted boolean",
        traceback="away from the bug",
        note=(
            "`and` loosened to `or`, so a sentence beginning with a brace is "
            "reported as malformed JSON instead of as prose."
        ),
        old='    return chunk.startswith("{") and chunk.endswith("}")',
        new='    return chunk.startswith("{") or chunk.endswith("}")',
    ),
)


# ---------------------------------------------------------------------------
# disk
# ---------------------------------------------------------------------------


def write_tasks(tasks_dir: Path = TASKS_DIR) -> list[Path]:
    """Regenerate tasks/ from BUGS. Run this whenever a project changes."""
    written: list[Path] = []
    for spec in BUGS:
        directory = tasks_dir / spec.task_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "bug.patch").write_text(spec.diff(), encoding="utf-8")
        (directory / "task.json").write_text(
            json.dumps(
                {
                    "task_id": spec.task_id,
                    "project": spec.project,
                    "target_test": spec.target_test,
                    "bug": "bug.patch",
                    "note": spec.note,
                    "shape": spec.shape,
                    "traceback": spec.traceback,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(directory)
    return written


def load_tasks(tasks_dir: Path = TASKS_DIR, only: list[str] | None = None) -> list[Task]:
    """Every task on disk, in task_id order. `only` filters and validates ids."""
    tasks: list[Task] = []
    for path in sorted(tasks_dir.glob("*/task.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        tasks.append(
            Task(
                task_id=data["task_id"],
                project=data["project"],
                target_test=data["target_test"],
                bug=data.get("bug", "bug.patch"),
                note=data.get("note", ""),
                shape=data.get("shape", ""),
                traceback=data.get("traceback", ""),
                directory=path.parent,
            )
        )
    if only is None:
        return tasks
    known = {task.task_id for task in tasks}
    unknown = [name for name in only if name not in known]
    if unknown:
        raise KeyError(f"unknown task id(s): {', '.join(unknown)}; have {', '.join(sorted(known))}")
    wanted = set(only)
    return [task for task in tasks if task.task_id in wanted]

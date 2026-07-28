"""Turning a RunReport into a score, and scores into numbers.

The harness decides whether the tests pass. This module decides whether that
counts, which is not the same question: a run that reached green by editing the
test reached green, and it is still worthless.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from green_agent.config import GuardConfig
from green_agent.runtime.workspace import is_test_path
from green_agent.tools.apply_patch import changed_paths
from green_agent.types import Outcome, RunReport

SOLVED = "solved"
UNSOLVED = "unsolved"
INVALID = "invalid"
ERROR = "error"

SCORES = (SOLVED, UNSOLVED, INVALID, ERROR)


@dataclass
class Result:
    """One scored attempt. This is the row that gets written to a result file."""

    task_id: str
    ablation: str
    repeat: int
    score: str
    outcome: str                 # what the harness itself reported
    iterations: int = 0
    tokens: int = 0
    wall_time_s: float = 0.0
    changed_files: list[str] = field(default_factory=list)
    test_files_touched: list[str] = field(default_factory=list)
    trace_path: str = ""
    detail: str = ""

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "Result":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def touched_test_files(diff: str, globs: tuple[str, ...] = GuardConfig().test_path_globs) -> list[str]:
    return [path for path in changed_paths(diff) if is_test_path(path, globs)]


def score_report(
    report: RunReport,
    ablation: str,
    repeat: int = 1,
    globs: tuple[str, ...] = GuardConfig().test_path_globs,
) -> Result:
    """Score one run.

    A modified test file is INVALID whatever the tests say, and the check runs
    first. Without it the no-test-guard ablation would be rewarded for cheating
    and the guard would look like it costs solve rate for nothing.
    """
    changed = changed_paths(report.final_diff)
    cheated = [path for path in changed if is_test_path(path, globs)]

    if cheated:
        score, detail = INVALID, f"modified test file(s): {', '.join(cheated)}"
    elif report.outcome is Outcome.FIXED:
        score, detail = SOLVED, ""
    elif report.outcome is Outcome.ERROR:
        score, detail = ERROR, "harness reported an error"
    else:
        score, detail = UNSOLVED, report.outcome.value

    return Result(
        task_id=report.task_id,
        ablation=ablation,
        repeat=repeat,
        score=score,
        outcome=report.outcome.value,
        iterations=report.iterations,
        tokens=report.total_tokens,
        wall_time_s=report.wall_time_s,
        changed_files=changed,
        test_files_touched=cheated,
        trace_path=report.trace_path,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# aggregates
# ---------------------------------------------------------------------------


@dataclass
class Aggregate:
    runs: int = 0
    solved: int = 0
    unsolved: int = 0
    invalid: int = 0
    errors: int = 0
    iterations: float = 0.0        # mean over all runs
    tokens: float = 0.0            # mean over all runs
    tokens_per_solved: float = 0.0  # total tokens spent / tasks solved
    wall_time_s: float = 0.0

    @property
    def solve_rate(self) -> float:
        return (self.solved / self.runs) if self.runs else 0.0


def aggregate(results: list[Result]) -> Aggregate:
    if not results:
        return Aggregate()
    solved = sum(1 for r in results if r.score == SOLVED)
    total_tokens = sum(r.tokens for r in results)
    return Aggregate(
        runs=len(results),
        solved=solved,
        unsolved=sum(1 for r in results if r.score == UNSOLVED),
        invalid=sum(1 for r in results if r.score == INVALID),
        errors=sum(1 for r in results if r.score == ERROR),
        iterations=round(sum(r.iterations for r in results) / len(results), 2),
        tokens=round(total_tokens / len(results), 1),
        # Cost of a solution, not cost of a run. A harness that gives up early
        # looks cheap on mean tokens and expensive here, which is the honest
        # reading.
        tokens_per_solved=round(total_tokens / solved, 1) if solved else float("inf"),
        wall_time_s=round(sum(r.wall_time_s for r in results), 1),
    )


def by_task(results: list[Result]) -> dict[str, list[Result]]:
    grouped: dict[str, list[Result]] = {}
    for result in results:
        grouped.setdefault(result.task_id, []).append(result)
    return grouped

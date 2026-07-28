from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# --------------------------------------------------------------------------
# Test execution
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """One line of a pytest traceback, already resolved against the repo."""

    path: str          # repo-relative if in_repo, else absolute
    lineno: int
    function: str
    in_repo: bool      # False for site-packages / stdlib frames -> deprioritised


@dataclass(frozen=True)
class Failure:
    """A single failing test, parsed out of pytest's report."""

    test_id: str               # e.g. "tests/test_cart.py::test_discount"
    exc_type: str              # e.g. "AssertionError"
    message: str               # first line(s) of the exception message
    frames: tuple[Frame, ...]  # innermost last, matching pytest's order

    def fingerprint(self) -> str:
        """Stable identity of a failure, for the stagnation detector.

        Excludes messages (values change between runs) and line numbers (they
        shift as patches are applied). What remains is the test, the exception
        type and where in the project it surfaced.
        """
        site = next(
            (f"{f.path}:{f.function}" for f in reversed(self.frames) if f.in_repo),
            "",
        )
        return f"{self.test_id}|{self.exc_type}|{site}"


@dataclass(frozen=True)
class TestResult:
    passed: bool
    failures: tuple[Failure, ...]
    duration_s: float
    raw_output: str            # kept for the trace, never sent wholesale to the model
    timed_out: bool = False
    collect_error: bool = False  # syntax error / import error -> different prompt


# --------------------------------------------------------------------------
# Context construction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeSlice:
    """A contiguous chunk of source, extracted by the AST slicer."""

    path: str          # repo-relative
    symbol: str        # "ShoppingCart.apply_discount" or "<module>"
    start_line: int
    end_line: int
    source: str
    reason: str        # "traceback frame" | "callee of X" | "requested by agent"

    def token_estimate(self) -> int:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------


class Outcome(str, Enum):
    FIXED = "fixed"                    # target test green, no other test broken
    BUDGET_EXHAUSTED = "budget"        # hit max iterations / max tokens
    STAGNATED = "stagnated"            # same failure fingerprint N times
    GUARD_ABORT = "guard_abort"        # agent kept trying a forbidden edit
    ERROR = "error"                    # harness-level failure


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass
class ToolResult:
    call_id: str
    ok: bool
    content: str        # what the model sees
    meta: dict[str, Any] = field(default_factory=dict)  # what the trace sees


@dataclass
class Step:
    """One turn of the loop: model output + the tool result it produced."""

    index: int
    hypothesis: str            # forced field, see prompts.py
    call: ToolCall | None
    result: ToolResult | None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class RunReport:
    """The unit of the benchmark. One repo + one target test -> one report."""

    task_id: str
    outcome: Outcome
    iterations: int
    total_tokens: int
    wall_time_s: float
    final_diff: str
    steps: list[Step] = field(default_factory=list)
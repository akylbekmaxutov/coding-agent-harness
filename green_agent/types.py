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
    path: str          
    lineno: int
    function: str
    in_repo: bool      


@dataclass(frozen=True)
class Failure:
    """A single failing test, parsed out of pytest's report."""
    test_id: str              
    exc_type: str              
    message: str               
    frames: tuple[Frame, ...]  

    def fingerprint(self) -> str:
        """Stable identity of a failure, used by the stagnation detector.
        Deliberately excludes the message *values* (numbers change between
        runs) but keeps test id + exception type + innermost repo frame.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class TestResult:
    passed: bool
    failures: tuple[Failure, ...]
    duration_s: float
    raw_output: str            
    timed_out: bool = False
    collect_error: bool = False  


@dataclass(frozen=True)
class CodeSlice:
    """A contiguous chunk of source, extracted by the AST slicer."""
    path: str          # repo-relative
    symbol: str        
    start_line: int
    end_line: int
    source: str
    reason: str        

    def token_estimate(self) -> int:
        raise NotImplementedError


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
    content: str        
    meta: dict[str, Any] = field(default_factory=dict) 


@dataclass
class Step:
    """One turn of the loop: model output + the tool result it produced."""
    index: int
    hypothesis: str            
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
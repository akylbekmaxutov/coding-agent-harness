from __future__ import annotations

from ..config import LoopConfig
from ..types import Outcome, Step, TestResult

FORCE_ACTION_DIRECTIVE = (
    "You have now spent several turns reading code without changing anything. "
    "The relevant functions are already shown above. Send an apply_patch diff "
    "this turn, or run_tests if you believe the code is already correct."
)

REPEAT_DIRECTIVE = (
    "You already made that exact call and nothing has changed since, so its "
    "result is unchanged and is in the history above. Take a different action."
)

FINISH_DIRECTIVE = (
    "The target test passed. Call finish now with a one-sentence summary of "
    "the root cause; the harness will verify it and end the run."
)

REPLAN_DIRECTIVE = (
    "STOP. Your last attempt produced exactly the same failure as before, so "
    "the current hypothesis is wrong. Do not re-send an equivalent patch. "
    "State a different hypothesis about the root cause, and if you are unsure "
    "which function is responsible, use read_file to inspect one you have not "
    "read yet."
)


class StagnationDetector:
    """Detects the most common agent failure: confident repetition.

    The signal is the failure fingerprint (test, exception type, innermost
    project frame) -- not the message, and not line numbers, both of which
    change for reasons unrelated to progress.

    First trip does not stop the run: it injects a re-plan directive, which
    turns a wasted loop into a cheap escape. Only a second trip aborts.
    """

    def __init__(self, threshold: int) -> None:
        self.threshold = max(1, threshold)
        self._last: str | None = None
        self._repeats = 0
        self._replans = 0
        self._aborted = False

    def observe(self, result: TestResult) -> None:
        current = "|".join(sorted(f.fingerprint() for f in result.failures))
        if result.passed or not current:
            self._last, self._repeats = None, 0
            return
        if current == self._last:
            self._repeats += 1
            # Decided here, while the repeat is visible. note_replan() clears
            # the counter, so an abort computed afterwards would never fire.
            if self._repeats >= self.threshold and self._replans >= 1:
                self._aborted = True
        else:
            self._last, self._repeats = current, 0

    @property
    def should_replan(self) -> bool:
        return self._repeats >= self.threshold and not self._aborted

    @property
    def should_abort(self) -> bool:
        return self._aborted

    def note_replan(self) -> None:
        self._replans += 1
        self._repeats = 0


class RepeatDetector:
    """Catches no-progress loops that the stagnation detector cannot see.

    StagnationDetector watches test results, so an agent that never calls
    run_tests is invisible to it. The first live run failed exactly this way:
    the model diagnosed the bug correctly on turn 1, then re-read the same file
    eleven more times until the budget ran out.

    Two signals, both cheap:
      - the same call against an unchanged workspace: blocked, never
        dispatched, because it cannot return anything new;
      - a streak of read-only turns: allowed, but the prompt is told to act.

    Keying on workspace state is what makes this safe to apply to every tool:
    re-running a test after a patch is a different signature and goes through,
    while re-running it having changed nothing does not.
    """

    READ_ONLY = {"read_file"}

    def __init__(self, max_read_streak: int = 2, max_blocked: int = 3) -> None:
        self.max_read_streak = max(1, max_read_streak)
        self.max_blocked = max(1, max_blocked)
        self._seen: set[str] = set()
        self.read_streak = 0
        self.blocked = 0

    @staticmethod
    def signature(call, state: str = "") -> str:
        arguments = ",".join(f"{k}={call.arguments[k]!r}" for k in sorted(call.arguments))
        return f"{state}:{call.name}({arguments})"

    def is_repeat(self, call, state: str = "") -> bool:
        return self.signature(call, state) in self._seen

    def note(self, call, state: str = "") -> None:
        self._seen.add(self.signature(call, state))
        self.read_streak = self.read_streak + 1 if call.name in self.READ_ONLY else 0

    def note_blocked(self) -> None:
        self.blocked += 1

    @property
    def should_force_action(self) -> bool:
        return self.read_streak >= self.max_read_streak

    @property
    def exhausted_repeats(self) -> bool:
        """Enough blocked calls that the run is going nowhere.

        Without this the two no-progress detectors disagree: blocking a call
        stops the agent learning nothing new, but nothing ends the run.
        """
        return self.blocked >= self.max_blocked


class BudgetTracker:
    """Iterations and tokens. Whichever trips first ends the run.

    Tokens matter more than iterations for reporting: cost per solved task is
    what makes two harnesses comparable.
    """

    def __init__(self, cfg: LoopConfig) -> None:
        self.cfg = cfg
        self.iterations = 0
        self.tokens = 0

    def charge(self, step: Step) -> None:
        self.iterations += 1
        self.tokens += step.prompt_tokens + step.completion_tokens

    @property
    def exhausted(self) -> bool:
        return (
            self.iterations >= self.cfg.max_iterations
            or self.tokens >= self.cfg.max_total_tokens
        )

    @property
    def reason(self) -> str:
        if self.iterations >= self.cfg.max_iterations:
            return f"iteration limit ({self.cfg.max_iterations})"
        return f"token limit ({self.cfg.max_total_tokens})"


def classify_outcome(
    fixed: bool,
    budget_exhausted: bool,
    stagnated: bool,
) -> Outcome:
    if fixed:
        return Outcome.FIXED
    if stagnated:
        return Outcome.STAGNATED
    if budget_exhausted:
        return Outcome.BUDGET_EXHAUSTED
    return Outcome.ERROR
from __future__ import annotations

import time
from pathlib import Path

from .config import Config
from .context.budget import fit
from .context.slicer import SymbolIndex, slices_for_failure
from .llm import LLM, LLMTransportError
from .observability.trace import Tracer
from .policy.guards import (
    FINISH_DIRECTIVE,
    FORCE_ACTION_DIRECTIVE,
    REPEAT_DIRECTIVE,
    REPLAN_DIRECTIVE,
    BudgetTracker,
    RepeatDetector,
    StagnationDetector,
    classify_outcome,
)
from .prompts import build_messages
from .runtime import pytest_runner
from .runtime.workspace import Workspace
from .tools.registry import ToolContext, dispatch, schemas
from .types import Outcome, RunReport, Step, TestResult, ToolCall, ToolResult

_NO_ACTION = (
    "You did not call a tool. Every turn must end in exactly one tool call. "
    "Call read_file, apply_patch, run_tests or finish now."
)


class Agent:
    def __init__(self, cfg: Config, llm: LLM) -> None:
        self.cfg = cfg
        self.llm = llm

    def run(self, repo_path: str | Path, target_test: str, task_id: str = "adhoc") -> RunReport:
        started = time.monotonic()
        tracer = Tracer(self.cfg.trace_dir, task_id)
        workspace = Workspace(repo_path, test_globs=self.cfg.guards.test_path_globs)
        steps: list[Step] = []
        budget = BudgetTracker(self.cfg.loop)
        stagnation = StagnationDetector(self.cfg.loop.stagnation_threshold)
        repeats = RepeatDetector(self.cfg.loop.max_read_streak,
                                 self.cfg.loop.max_blocked_repeats)
        outcome = Outcome.ERROR
        final_diff = ""

        tracer.emit("run_start", task_id=task_id, repo=str(repo_path), target=target_test,
                    model=self.cfg.model.name, config=self.cfg.__dict__)

        try:
            workspace.setup()
            ctx = ToolContext(workspace=workspace, config=self.cfg, target_test=target_test)
            index = SymbolIndex(workspace.root)

            result = pytest_runner.run(
                workspace.root, target_test, self.cfg.guards.pytest_timeout_s
            )
            ctx.last_result = result
            tracer.emit("baseline", passed=result.passed, failures=len(result.failures),
                        collect_error=result.collect_error)

            if result.passed:
                tracer.emit("run_end", outcome="invalid_task",
                            detail="target test already passes")
                return RunReport(task_id, Outcome.ERROR, 0, 0, _elapsed(started), "", steps)

            directive: str | None = None
            fixed = False
            # The last state in which the code at least imported. Anchoring on
            # "last applied patch" would restore the very edit that broke
            # collection, which is the thing we are trying to undo.
            healthy = workspace.checkpoint("baseline")

            while not budget.exhausted:
                step, directive, fixed, healthy = self._turn(
                    ctx, index, tracer, steps, stagnation, repeats, directive, healthy
                )
                steps.append(step)
                budget.charge(step)
                if fixed:
                    break
                if stagnation.should_abort or repeats.exhausted_repeats:
                    if repeats.exhausted_repeats:
                        tracer.emit("stagnation", detail="repeated calls with no change")
                    outcome = Outcome.STAGNATED
                    break

            final_diff = workspace.diff()
            outcome = classify_outcome(
                fixed, budget.exhausted, stagnation.should_abort or repeats.exhausted_repeats
            )

        except LLMTransportError as exc:
            tracer.emit("error", kind="transport", detail=str(exc))
            outcome = Outcome.ERROR
        finally:
            tracer.emit("run_end", outcome=outcome.value, iterations=budget.iterations,
                        tokens=budget.tokens, diff_lines=len(final_diff.splitlines()))
            workspace.teardown()

        return RunReport(
            task_id=task_id,
            outcome=outcome,
            iterations=budget.iterations,
            total_tokens=budget.tokens,
            wall_time_s=_elapsed(started),
            final_diff=final_diff,
            steps=steps,
        )

    # -- one iteration -----------------------------------------------------

    def _turn(self, ctx, index, tracer, steps, stagnation, repeats, directive, healthy):
        result: TestResult = ctx.last_result
        failure = result.failures[0] if result.failures else None
        slices, degraded = fit(
            slices_for_failure(ctx.workspace.root, failure.frames if failure else (),
                               self.cfg.context, index),
            self.cfg.context.max_context_tokens,
        )
        tracer.emit("context_built", slices=[f"{s.path}:{s.symbol}" for s in slices],
                    degraded=degraded)

        messages = build_messages(
            repo_name=Path(ctx.workspace.source).name,
            target_test=ctx.target_test,
            failure=failure,
            raw_tail=result.raw_output[-self.cfg.context.max_raw_output_chars:],
            slices=slices,
            history=steps,
            directive=directive,
        )
        force_tool = bool(steps and steps[-1].call is None)
        completion = self.llm.complete(messages, schemas(), force_tool=force_tool)
        tracer.emit("model_call", hypothesis=completion.hypothesis,
                    tool=completion.tool_call.name if completion.tool_call else None,
                    parse_error=completion.parse_error,
                    dropped_calls=completion.extra_calls_dropped,
                    prompt_tokens=completion.prompt_tokens,
                    completion_tokens=completion.completion_tokens,
                    response=completion.raw)

        index_no = len(steps) + 1
        step = Step(index=index_no, hypothesis=completion.hypothesis,
                    call=completion.tool_call, result=None,
                    prompt_tokens=completion.prompt_tokens,
                    completion_tokens=completion.completion_tokens)

        # No action, or arguments we could not parse: hand the reason back and
        # spend the turn. Both are normal model behaviour, not harness faults.
        if completion.tool_call is None:
            step.result = ToolResult(call_id="", ok=False, content=_NO_ACTION)
            return step, None, False, healthy
        if completion.parse_error:
            step.result = ToolResult(call_id=completion.tool_call.call_id, ok=False,
                                     content=completion.parse_error)
            return step, None, False, healthy

        # An identical call cannot produce new information, so it is blocked
        # rather than dispatched: the turn is spent either way, but the model
        # gets told why instead of reading the same file again.
        state = ctx.workspace.state_token()
        if repeats.is_repeat(completion.tool_call, state):
            # A blocked turn is still a turn spent reading nothing new, so it
            # counts toward the streak that forces an edit.
            repeats.note(completion.tool_call, state)
            repeats.note_blocked()
            step.result = ToolResult(call_id=completion.tool_call.call_id, ok=False,
                                     content=REPEAT_DIRECTIVE)
            tracer.emit("repeat_blocked",
                        call=RepeatDetector.signature(completion.tool_call, state))
            directive = REPEAT_DIRECTIVE
            if repeats.should_force_action:
                tracer.emit("read_streak", turns=repeats.read_streak)
                directive = REPEAT_DIRECTIVE + "\n" + FORCE_ACTION_DIRECTIVE
            return step, directive, False, healthy

        repeats.note(completion.tool_call, state)
        result_obj = dispatch(completion.tool_call, ctx)
        step.result = result_obj
        tracer.emit("tool_result", tool=completion.tool_call.name, ok=result_obj.ok,
                    content=result_obj.content[:400], meta=result_obj.meta)

        next_directive: str | None = None
        fixed = False

        if completion.tool_call.name == "run_tests":
            stagnation.observe(ctx.last_result)
            if ctx.last_result.collect_error:
                ctx.workspace.restore(healthy)
                ctx.last_result = pytest_runner.run(
                    ctx.workspace.root, ctx.target_test, self.cfg.guards.pytest_timeout_s
                )
                tracer.emit("recovered", detail="collection error; workspace restored")
                next_directive = (
                    "Your last patch broke the file so it could not even be imported. "
                    "The harness reverted it. Read the function again before editing."
                )
            else:
                # The code still imports, so this is a state worth returning to.
                healthy = ctx.workspace.checkpoint(f"healthy-{step.index}")
            if stagnation.should_replan:
                stagnation.note_replan()
                tracer.emit("stagnation", detail="identical failure repeated")
                next_directive = REPLAN_DIRECTIVE
            elif ctx.last_result.passed:
                # The verifier says green but the model has not claimed it.
                # Say so, rather than letting it re-run the test to be sure.
                next_directive = FINISH_DIRECTIVE

        elif completion.tool_call.name == "finish":
            fixed = self._verify(ctx, tracer)
            if not fixed:
                next_directive = (
                    "The harness verified your claim and the tests are not green. "
                    "Keep working from the failure shown above."
                )

        if next_directive is None and repeats.should_force_action:
            tracer.emit("read_streak", turns=repeats.read_streak)
            next_directive = FORCE_ACTION_DIRECTIVE

        return step, next_directive, fixed, healthy

    # -- verification ------------------------------------------------------

    def _verify(self, ctx, tracer) -> bool:
        """finish() is a claim; this is the check. The model does not decide."""
        target = pytest_runner.run(
            ctx.workspace.root, ctx.target_test, self.cfg.guards.pytest_timeout_s
        )
        ctx.last_result = target
        if not target.passed:
            tracer.emit("verify", stage="target", passed=False)
            return False
        if not self.cfg.guards.run_full_suite_on_success:
            tracer.emit("verify", stage="target", passed=True, suite="skipped")
            return True
        suite = pytest_runner.run(ctx.workspace.root, None, self.cfg.guards.pytest_timeout_s)
        tracer.emit("verify", stage="suite", passed=suite.passed)
        if not suite.passed:
            ctx.last_result = suite
        return suite.passed


def _elapsed(started: float) -> float:
    return round(time.monotonic() - started, 3)
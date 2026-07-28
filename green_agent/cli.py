from __future__ import annotations

import argparse
import sys

from .config import Config, ContextConfig, GuardConfig, LoopConfig, ModelConfig, load_dotenv
from .llm import LLMTransportError, OpenAICompatibleLLM, ReplayLLM
from .loop import Agent
from .types import Outcome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="green-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    fix = sub.add_parser("fix", help="fix a failing test")
    fix.add_argument("--repo", required=True)
    fix.add_argument("--test", required=True, help="pytest node id")
    fix.add_argument("--model", default=None)
    fix.add_argument("--max-iterations", type=int, default=None)
    fix.add_argument("--depth", type=int, default=None, help="call-graph depth for slicing")
    fix.add_argument("--context-tokens", type=int, default=None)
    fix.add_argument("--allow-test-edits", action="store_true", help="ablation: drop the guard")
    fix.add_argument("--no-suite-check", action="store_true", help="skip the regression sweep")
    fix.add_argument("--trace-dir", default=None)
    fix.add_argument("--show-diff", action="store_true")

    replay = sub.add_parser("replay", help="re-run a recorded trace without the network")
    replay.add_argument("--trace", required=True)
    replay.add_argument("--repo", required=True)
    replay.add_argument("--test", required=True)

    return parser


def _config(args) -> Config:
    base = Config()
    model = ModelConfig(name=args.model) if getattr(args, "model", None) else base.model
    loop = LoopConfig(
        max_iterations=args.max_iterations or base.loop.max_iterations,
        max_total_tokens=base.loop.max_total_tokens,
        stagnation_threshold=base.loop.stagnation_threshold,
    )
    context = ContextConfig(
        max_context_tokens=args.context_tokens or base.context.max_context_tokens,
        callee_depth=base.context.callee_depth if args.depth is None else args.depth,
    )
    guards = GuardConfig(
        forbid_test_file_edits=not args.allow_test_edits,
        run_full_suite_on_success=not args.no_suite_check,
    )
    return Config(model=model, context=context, loop=loop, guards=guards,
                  trace_dir=args.trace_dir or base.trace_dir)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "replay":
        cfg = Config()
        report = Agent(cfg, ReplayLLM.from_trace(args.trace)).run(args.repo, args.test, "replay")
        return _report(report, show_diff=False)

    load_dotenv()
    cfg = _config(args)
    try:
        llm = OpenAICompatibleLLM(cfg.model)
    except (RuntimeError, LLMTransportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = Agent(cfg, llm).run(args.repo, args.test, task_id=args.test)
    return _report(report, show_diff=args.show_diff)


def _report(report, show_diff: bool) -> int:
    print(
        f"{report.outcome.value}  iterations={report.iterations}  "
        f"tokens={report.total_tokens}  time={report.wall_time_s}s"
    )
    for step in report.steps:
        action = step.call.name if step.call else "-"
        status = "ok" if step.result and step.result.ok else "rejected"
        print(f"  #{step.index} {action:<12} {status:<9} {step.hypothesis[:60]}")
    if show_diff and report.final_diff:
        print("\n" + report.final_diff)
    if report.outcome is Outcome.FIXED:
        return 0
    return 1 if report.outcome is not Outcome.ERROR else 2


if __name__ == "__main__":
    raise SystemExit(main())
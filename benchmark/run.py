"""Run the benchmark.

    python benchmark/run.py --verify-tasks
    python benchmark/run.py --ablation full
    python benchmark/run.py --ablation no-slicing --tasks datalib-top-n-default
    python benchmark/run.py --ablation full --repeat 3 --out results/full-x3.json

Every task runs in its own throwaway git repo with the bug committed at HEAD.
One task blowing up is recorded as an ERROR row and the sweep carries on: a
sweep that aborts halfway is worth less than a sweep with one bad row in it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from green_agent.config import Config, load_dotenv
from green_agent.llm import LLMTransportError, OpenAICompatibleLLM
from green_agent.loop import Agent

from benchmark.ablations import ABLATIONS, config_for, describe
from benchmark.catalog import BENCHMARK_DIR, load_tasks
from benchmark.prepare import TaskError, checkout, verify
from benchmark.scoring import ERROR, Result, aggregate, score_report

RESULTS_DIR = BENCHMARK_DIR / "results"
TRACE_DIR = BENCHMARK_DIR / "traces"

_MARK = {"solved": "PASS", "unsolved": "FAIL", "invalid": "INVALID", "error": "ERROR"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark/run.py")
    parser.add_argument("--tasks", nargs="+", default=None, help="task ids; default all")
    parser.add_argument("--repeat", type=int, default=1, help="attempts per task")
    parser.add_argument("--ablation", default="full", choices=sorted(ABLATIONS))
    parser.add_argument("--out", default=None, help="default results/<ablation>.json")
    parser.add_argument("--verify-tasks", action="store_true",
                        help="check every task is observable, then exit")
    parser.add_argument("--model", default=None)
    parser.add_argument("--trace-dir", default=None)
    return parser


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def verify_tasks(tasks) -> int:
    """Assert every task is observable. Nothing else runs until this passes."""
    bad = 0
    for task in tasks:
        try:
            check = verify(task)
        except TaskError as exc:
            print(f"  BROKEN  {task.task_id}\n            {exc}")
            bad += 1
            continue
        if check.ok:
            print(f"  ok      {task.task_id}  ({task.shape}; {task.traceback})")
        else:
            bad += 1
            print(f"  BROKEN  {task.task_id}")
            for problem in check.problems:
                print(f"            {problem}")
    print(f"\n{len(tasks) - bad}/{len(tasks)} tasks are observable")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------


def run_one(task, cfg: Config, llm, ablation: str, repeat: int) -> Result:
    """One attempt. Every failure mode below is a row, never an exception."""
    try:
        with checkout(task) as repo:
            report = Agent(cfg, llm).run(repo, task.target_test, task_id=task.task_id)
        return score_report(report, ablation, repeat)
    except TaskError as exc:
        return Result(task.task_id, ablation, repeat, ERROR, "task_error", detail=str(exc))
    except LLMTransportError as exc:
        return Result(task.task_id, ablation, repeat, ERROR, "transport", detail=str(exc))
    except Exception as exc:  # a crash in one task must not end the sweep
        return Result(task.task_id, ablation, repeat, ERROR, "crash",
                      detail=f"{type(exc).__name__}: {exc}")


def sweep(tasks, cfg: Config, llm, ablation: str, repeats: int) -> list[Result]:
    results: list[Result] = []
    total = len(tasks) * repeats
    index = 0
    for repeat in range(1, repeats + 1):
        for task in tasks:
            index += 1
            started = time.monotonic()
            print(f"  [{index:>2}/{total}] {task.task_id:<38} ", end="", flush=True)
            result = run_one(task, cfg, llm, ablation, repeat)
            results.append(result)
            print(
                f"{_MARK[result.score]:<8} "
                f"iters={result.iterations:<3} tokens={result.tokens:<7} "
                f"{round(time.monotonic() - started, 1)}s"
                + (f"  {result.detail}" if result.detail else "")
            )
    return results


def write_results(path: Path, ablation: str, cfg: Config, results: list[Result]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ablation": ablation,
        "overrides": describe(ablation),
        "model": cfg.model.name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "callee_depth": cfg.context.callee_depth,
            "max_context_tokens": cfg.context.max_context_tokens,
            "whole_file_context": cfg.context.whole_file_context,
            "max_iterations": cfg.loop.max_iterations,
            "repeat_detection": cfg.loop.repeat_detection,
            "forbid_test_file_edits": cfg.guards.forbid_test_file_edits,
        },
        "results": [result.to_json() for result in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        tasks = load_tasks(only=args.tasks)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not tasks:
        print("error: no tasks found; run python benchmark/make_tasks.py", file=sys.stderr)
        return 2

    if args.verify_tasks:
        return verify_tasks(tasks)

    load_dotenv()
    overrides: dict = {"trace_dir": args.trace_dir or str(TRACE_DIR / args.ablation)}
    if args.model:
        overrides["model.name"] = args.model
    cfg = config_for(args.ablation, **overrides)

    try:
        llm = OpenAICompatibleLLM(cfg.model)
    except (RuntimeError, LLMTransportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"ablation : {args.ablation}  ({describe(args.ablation)})")
    print(f"model    : {cfg.model.name}")
    print(f"tasks    : {len(tasks)} x {args.repeat}\n")

    results = sweep(tasks, cfg, llm, args.ablation, max(1, args.repeat))

    out = Path(args.out) if args.out else RESULTS_DIR / f"{args.ablation}.json"
    write_results(out, args.ablation, cfg, results)

    totals = aggregate(results)
    print(
        f"\nsolved {totals.solved}/{totals.runs}"
        f"  ({totals.solve_rate:.0%})"
        f"  invalid={totals.invalid} errors={totals.errors}"
        f"  mean_iters={totals.iterations}  mean_tokens={totals.tokens:.0f}"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

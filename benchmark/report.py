"""Read result files and print what they say.

    python benchmark/report.py results/full.json
    python benchmark/report.py results/full.json results/no-slicing.json

With one file: per-task outcomes and the aggregate. With two: the same for
each, then the delta, baseline first. A run that modified a test file is
INVALID and is never counted as solved, whatever the tests reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.scoring import INVALID, SOLVED, Aggregate, Result, aggregate, by_task

_MARK = {"solved": "PASS", "unsolved": "fail", "invalid": "INVALID", "error": "ERROR"}


def load(path: str | Path) -> tuple[str, list[Result]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("ablation", Path(path).stem), [
        Result.from_json(row) for row in data.get("results", [])
    ]


def _cell(results: list[Result]) -> str:
    """One task's outcomes across repeats, worst-first so a cheat is visible."""
    if len(results) == 1:
        return _MARK[results[0].score]
    order = {INVALID: 0, "error": 1, "unsolved": 2, SOLVED: 3}
    marks = [_MARK[r.score] for r in sorted(results, key=lambda r: order[r.score])]
    solved = sum(1 for r in results if r.score == SOLVED)
    return f"{solved}/{len(results)}  " + " ".join(marks)


def print_table(name: str, results: list[Result]) -> Aggregate:
    grouped = by_task(results)
    width = max((len(task) for task in grouped), default=10)
    print(f"\n{name}")
    print(f"  {'task':<{width}}  {'outcome':<22} {'iters':>6} {'tokens':>8}")
    print(f"  {'-' * width}  {'-' * 22} {'-' * 6} {'-' * 8}")
    for task in sorted(grouped):
        runs = grouped[task]
        iters = sum(r.iterations for r in runs) / len(runs)
        tokens = sum(r.tokens for r in runs) / len(runs)
        print(f"  {task:<{width}}  {_cell(runs):<22} {iters:>6.1f} {tokens:>8.0f}")
        for run in runs:
            if run.detail and run.score != "unsolved":
                print(f"  {'':<{width}}    {run.detail}")

    totals = aggregate(results)
    print(
        f"\n  solve rate        {totals.solved}/{totals.runs} ({totals.solve_rate:.0%})\n"
        f"  mean iterations   {totals.iterations}\n"
        f"  mean tokens       {totals.tokens:.0f}\n"
        f"  tokens per solve  {_tokens_per_solve(totals)}\n"
        f"  invalid / error   {totals.invalid} / {totals.errors}\n"
        f"  wall time         {totals.wall_time_s}s"
    )
    return totals


def _tokens_per_solve(totals: Aggregate) -> str:
    return "n/a (nothing solved)" if totals.solved == 0 else f"{totals.tokens_per_solved:.0f}"


def _signed(value: float, digits: int = 0) -> str:
    return f"{value:+.{digits}f}"


def print_delta(base_name: str, base: list[Result], other_name: str, other: list[Result]) -> None:
    a, b = aggregate(base), aggregate(other)
    print(f"\ndelta: {other_name} vs {base_name}")
    print(f"  solve rate        {a.solve_rate:.0%} -> {b.solve_rate:.0%}"
          f"  ({_signed((b.solve_rate - a.solve_rate) * 100)} points)")
    print(f"  mean iterations   {a.iterations} -> {b.iterations}"
          f"  ({_signed(b.iterations - a.iterations, 2)})")
    print(f"  mean tokens       {a.tokens:.0f} -> {b.tokens:.0f}"
          f"  ({_signed(b.tokens - a.tokens)})")
    print(f"  invalid runs      {a.invalid} -> {b.invalid}  ({_signed(b.invalid - a.invalid)})")

    base_by_task, other_by_task = by_task(base), by_task(other)
    moved = []
    for task in sorted(set(base_by_task) | set(other_by_task)):
        before = sum(1 for r in base_by_task.get(task, []) if r.score == SOLVED)
        after = sum(1 for r in other_by_task.get(task, []) if r.score == SOLVED)
        if before != after:
            moved.append(f"  {task:<38} {before} -> {after} solved")
    if moved:
        print("\n  tasks that moved")
        print("\n".join(moved))
    else:
        print("\n  no task changed outcome")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmark/report.py")
    parser.add_argument("results", nargs="+", help="one result file, or two for a delta")
    args = parser.parse_args(argv)

    if len(args.results) > 2:
        print("error: pass one result file, or two to compare", file=sys.stderr)
        return 2

    loaded = []
    for path in args.results:
        try:
            loaded.append(load(path))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: cannot read {path}: {exc}", file=sys.stderr)
            return 2

    for name, results in loaded:
        print_table(name, results)

    if len(loaded) == 2:
        (base_name, base), (other_name, other) = loaded
        print_delta(base_name, base, other_name, other)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

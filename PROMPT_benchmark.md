# Task: build the benchmark, and fix the harness with what it tells you

Read `CLAUDE.md` first. Work in phases. **After each phase run
`python scripts/initial_test.py` and commit only when it exits 0.**

## The point of this task

The benchmark is not a scoreboard. It is an instrument for finding defects in
**this harness**. Every task you run is a probe: when a run fails, the first
question is always *did the model fail, or did the harness fail to give it a
fair chance?*

**Priority order, when they conflict:**

1. **Fix defects in `green_agent/`.** Highest priority, always. A harness bug
   found and fixed is worth more than three more benchmark tasks.
2. Regression checks in `scripts/initial_test.py` for every fix.
3. More benchmark tasks.
4. More ablations.

If time runs short, ship fewer tasks and a better harness. Say so in the README.

## Scope: Python only

All three projects are Python with pytest. No TypeScript, no vitest, no second
runner. This is a deliberate scope cut — note it in the README as a limitation
("the runner boundary is pytest-shaped; a second language would need a
`TestRunner` protocol, which is sketched but not built").

## Phase 1 — three Python projects

Create `benchmark/projects/`. Each is small, self-contained, dependency-light,
runs in milliseconds, and is **fully green before any bug is injected**.

**1. `backend/`** — service-layer logic: routing, an auth/permission check,
pagination, error mapping to status codes. Plain functions plus a dict store.
No network, no database, no framework.

**2. `datalib/`** — parsing and validation: a CSV/record loader, type coercion,
a schema validator, an aggregation with grouping. Bug shapes here are naturally
different from the backend's — boundary conditions, empty inputs, type edges.

**3. `agentlib/`** — a miniature agent library, deliberately mirroring this
project's own domain: a tool registry, a retry policy with backoff, a JSON
argument parser tolerant of fenced output, a turn budget. The most interesting
of the three, because its bugs are the ones a real harness actually has.

Each project gets a `README.md` naming what it does and why it exists.

## Phase 2 — injected bugs

`benchmark/tasks/<task_id>/task.json`:

```json
{
  "task_id": "backend-pagination-offset",
  "project": "backend",
  "target_test": "tests/test_pagination.py::test_last_page_is_not_empty",
  "bug": "bug.patch",
  "note": "off-by-one: last page dropped when total % size == 0"
}
```

Rules, all learned the hard way:

- The bug is a **patch applied to a green project**, so a ground-truth fix
  provably exists.
- **Every bug must be observable.** Exactly one named test fails with the patch
  applied; the whole suite is green without it. Add `--verify-tasks` that
  asserts this for every task and fails loudly. Do this *before* running the
  agent on anything.
- Vary the shape: off-by-one, wrong comparison operator, swapped argument order,
  missing early return, wrong default value, inverted boolean. Not six
  variations of one mistake.
- Vary the difficulty: at least two where the traceback points **away** from the
  bug (a plain assertion in the test, fault two calls deep). That is what the
  call-graph slicer exists for, and it is the case most likely to expose a
  context-construction defect.
- **6 tasks total, two per project.** More tasks is not more insight.

## Phase 3 — runner and report

`benchmark/run.py`

- `--tasks <ids>` (default all), `--repeat N` (default 1),
  `--ablation <name>`, `--out results/<name>.json`, `--verify-tasks`.
- One `RunReport` per task plus its trace path. One task crashing must never
  abort the sweep.
- Live one-line-per-task progress.

`benchmark/report.py`

- Solve rate, mean iterations, mean tokens, tokens per solved task, per-task
  outcome. Given two result files, print the **delta**.
- A run that modified a test file is scored **INVALID, not solved**, regardless
  of what the tests say. Without this the `no-test-guard` ablation rewards
  cheating and makes the guard look pointless.

Ablations, each a config change and nothing else: `full`, `no-slicing`
(whole-file context), `depth-1`, `no-repeat-detection`, `no-test-guard`.
Run `full` plus at least `no-slicing`.

## Phase 4 — harness triage (the real work)

For every task that does not reach `fixed`, open the trace and classify it.
Keep `benchmark/FINDINGS.md` with one entry per investigation:

```
### backend-pagination-offset (2 of 3 runs failed)
Symptom:   agent patched the wrong function twice, then hit the budget
Evidence:  context_built shows the buggy function was never sliced
Cause:     HARNESS — callee_depth=2 stops one hop short via a helper module
Fix:       resolve `from x import y` through re-exports in slicer.py
Check:     suite_slicer "resolves through a re-exported helper"
```

Classify honestly as **HARNESS**, **MODEL**, or **TASK** (the task itself was
unfair or ambiguous). Signals that it is the harness:

- the buggy function never appears in a `context_built` event → slicing gap
- `apply_patch` rejected repeatedly with the same reason → diff format or
  guard message problem
- the agent loops without progress and nothing stops it → policy gap
- a run reported `fixed` while the diff touched a test file → verification gap
- `collect_error` recovery restored the wrong state → checkpoint gap
- the model states the right root cause and still fails → prompt or tool gap

**Every HARNESS finding gets: a fix in `green_agent/`, a regression check in
`scripts/initial_test.py`, and a line in the README.** Then re-run the affected
tasks and record the before/after in `FINDINGS.md`.

Expect to find real defects here. The first live run of this harness failed
because the model diagnosed the bug on turn 1 and then re-read the same file
eleven times — a policy gap, not a model weakness. That fix is already in;
there will be others.

## Phase 5 — wire into the gate

Add `suite_benchmark()` to `scripts/initial_test.py`, registered in `SUITES`,
running offline under `ReplayLLM` with no API key. It must check: every
`task.json` parses; every bug patch applies cleanly; every task fails exactly
its named test with the bug and passes without it; report arithmetic is correct
on synthetic results; a test-file modification scores `INVALID`.

Scenario scripts must outlast their iteration cap, and every check must assert
one specific outcome — never "not the good one".

## Phase 6 — README

Add **Results**: the table, the ablation delta, and one honest paragraph on what
the numbers do and do not support (small N, one model, small repos). Add a
**What the benchmark found** section summarising the harness defects it exposed
and how they were fixed — for this submission that section matters more than the
solve rate. Then fill in **Tradeoffs**, **With more time**, **Hours**, and
re-read the whole file for claims that no longer match the code.

## Definition of done

- `python scripts/initial_test.py` exits 0 with the benchmark suite included.
- `python benchmark/run.py --verify-tasks` passes for all tasks.
- `python benchmark/run.py --ablation full` writes a result file.
- `python benchmark/report.py results/full.json results/no-slicing.json` prints
  a delta.
- `benchmark/FINDINGS.md` has at least one investigated failure.
- README Results and "What the benchmark found" are written and accurate.

## Non-goals

No web UI, no multi-agent orchestration, no `bash` tool, no embeddings, no
parallel trajectories, no second model provider, no second language runner. If
something was cut for time, one README sentence saying what and why — a
deliberate, explained omission beats a half-finished feature.
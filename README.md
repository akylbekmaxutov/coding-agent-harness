# green-agent

An AI coding agent that fixes **one failing test**, and a benchmark whose job is
to find defects in the agent's harness.

The success criterion is machine-checkable and non-negotiable:

- the target test passes, **and**
- no test file was modified, **and**
- the rest of the suite still passes.

The model proposes; the harness verifies. `finish()` is a claim, not an ending.

Everything here except the model is the submission: how the problem was framed,
what the harness does with a model that is confidently wrong, and what happened
when it was measured.

---

## Quick start

```bash
git clone <this repo> && cd coding-agent-harness

uv venv                                 # Python >= 3.11
source .venv/bin/activate
uv pip install httpx pytest

python scripts/initial_test.py          # 291 offline checks, ~21s, no key needed
echo 'OPENAI_KEY=sk-...' > .env         # only needed for live runs

python -m green_agent.cli fix --repo demo_repo \
  --test "tests/test_cart.py::test_discount_can_lose_free_shipping" --show-diff
```

That last command should print `fixed` in three or four iterations. Full
instructions — including running against your own repository — are under
[Usage](#usage); the benchmark is under
[Running the benchmark](#running-the-benchmark).

Model: `gpt-5.6-luna`, `reasoning_effort="none"`. That last part is a design
constraint rather than a detail: there is no hidden deliberation to lean on, so
every bit of decomposition has to live in the harness.

---

## Results

Six tasks, three repeats each, `gpt-5.6-luna`. A run that modified a test file
is scored **INVALID**, never solved, whatever the tests reported.

| ablation | knob moved | solved | mean iters | mean tokens | tokens / solve |
|---|---|---|---|---|---|
| `full` | shipped defaults | **18/18 (100%)** | 3.94 | 4,711 | 4,711 |
| `no-slicing` | `context.whole_file_context=True` | 3/18 (17%) | 9.56 | 15,659 | 93,956 |
| `depth-1` | `context.callee_depth=1` | 8/18 (44%) | 8.17 | 10,556 | 23,751 |
| `no-repeat-detection` | `loop.repeat_detection=False` | 16/18 (89%) | 4.83 | 5,988 | 6,736 |
| `no-test-guard` | `guards.forbid_test_file_edits=False` | 17/18 (94%) | 4.22 | 5,158 | 5,462 |

Per task on `full`, every task 3/3:

| task | shape | traceback | mean iters |
|---|---|---|---|
| `agentlib-backoff-swapped-args` | swapped argument order | away | 3.0 |
| `agentlib-loose-brace-check` | inverted boolean | away | 3.0 |
| `backend-pagination-off-by-one` | off-by-one | away | 3.0 |
| `backend-unknown-action-no-early-return` | missing early return | at the bug | 6.3 |
| `datalib-single-row-mean` | wrong comparison operator | away | 3.0 |
| `datalib-top-n-default` | wrong default value | away | 5.3 |

### The ablation that matters

`no-slicing` costs 83 points of solve rate and triples the token bill. The
per-task breakdown says why: the only task it still solves is
`backend-unknown-action-no-early-return`, the one whose bug raises a `TypeError`
so the traceback names `service.py`. For the other five the failure is a plain
assertion, the traceback names only the test file, and dumping that whole file
does not contain the bug. The agent still has `read_file` and could in principle
read its way out; it mostly does not. **The call-graph slicer is not a token
optimisation, it is the mechanism by which the buggy function reaches the model
at all.**

`depth-1` separates the two kinds of task cleanly, which is what it was built to
do. `datalib-top-n-default` keeps 3/3 because its bug is a module constant that
no slice contains at any depth, so `read_file` finds it either way; the four
"traceback points away" tasks drop to 1/3, because they need
test → caller → callee and depth 1 removes exactly that hop.

### What these numbers do not support

Small N, one model, small repos, one language. 18 runs per ablation is enough to
separate 100% from 17%; it is nowhere near enough to call 94% different from
100%. Specifically:

- **`no-test-guard` produced zero INVALID runs.** This model did not take the
  offer to edit the test even when the guard was removed. That is a fact about
  this model on these six bugs, not evidence the guard is unnecessary — the
  guard is cheap and the failure it prevents is total, so it stays on. The
  INVALID rule exists so that if the model ever does cheat, the scoreboard says
  so instead of rewarding it.
- **`no-repeat-detection` at 89% is inside the noise.** Its two failures were
  both the residual issue described below, not read loops.
- The repos are a few hundred lines each. A slicer that resolves calls naively
  is fine at this size and would not be at 100k lines.
- Every bug was injected by me, so difficulty is my judgement, not a sample of
  anything. Two of the six are "traceback points at the bug" easy; four are not.
- 100% on `full` mostly means the tasks stopped being able to distinguish
  harness variants once the defects below were fixed. That is a reason to add
  harder tasks, not a reason to be pleased.

---

## What the benchmark found

This is the part that matters more than the solve rate. Full write-ups with
trace evidence and before/after numbers are in
[`benchmark/FINDINGS.md`](benchmark/FINDINGS.md).

**Six of the seven investigated failures were the harness, not the model.** In
most of them the model named the correct root cause on turn 1 and then could not
get the fix through. The first full sweep scored 16/18 at 6.72 iterations and
8,165 tokens per run; the final one scores 18/18 at 3.94 and 4,711 — with no
change to the model, the prompting strategy, or the tasks.

Progress was not monotonic, and the shape of that is worth stating: an
intermediate sweep dropped to 13/18. Fixing the patch-format defects let edits
actually land, which exposed a stale-context defect that had been hidden behind
them — while patches were being rejected, the agent never got far enough to be
shown its own out-of-date code.

1. **The tool's own name primed a format it rejected.** The tool is called
   `apply_patch`, which is also the name of a widely trained tool whose argument
   format is a `*** Begin Patch` envelope. Models emit that envelope unprompted.
   The harness threw those calls away — and the rejection said "re-read the file
   to get exact context lines", when the context lines were never the problem.
   *Fix:* `unwrap_envelope()` rewrites the envelope as a unified diff, and
   `_advice()` tells a format error apart from a context error.

2. **The harness contradicted itself.** `apply_patch` told a model whose diff was
   refused to re-read the file; `RepeatDetector` then blocked exactly that
   re-read, because the workspace had not changed. A run died of stagnation
   having made no edit at all. *Fix:* a rejection counts as new information and
   moves the state the repeat guard keys on.

3. **A bare `@@` hunk header could not be consumed.** The envelope marks hunks
   with `@@` and no line ranges; `git apply` cannot parse that with or without
   `--recount`. One run sent the same semantically perfect one-line fix seven
   times and never landed it. *Fix:* `renumber_bare_hunks()` finds the hunk's
   own context lines in the real file and writes the header git wants. A hunk
   that cannot be located is left alone, because guessing a position is how a
   file gets corrupted. The system prompt and tool schema now also show the
   required format in full; previously the only guidance was "use a/ and b/
   prefixes".

4. **Directives evaporated on degenerate turns.** The loop computes a directive
   for the next turn, and the no-tool-call branch returned `None` for it. So the
   harness said "the test passed, call finish" once, the model returned an empty
   completion, and it was never told again — then spent five turns and the whole
   budget. *Fix:* the incoming directive is carried across no-action and
   parse-error turns, with the reason appended rather than substituted.

5. **The model was shown its own unfixed code.** `SymbolIndex` was built once per
   run, and every `Symbol` holds the file text it was parsed from — so after the
   first patch, every slice in every later prompt showed pre-patch source. A run
   applied the correct fix on turn 1, read back the old code, concluded *"the
   prior attempted fix did not persist"*, and re-sent the same patch until the
   budget ran out. This was the worst one. *Fix:* `refresh_if_changed()` rebuilds
   the index when the workspace token moves. The regression check asserts the
   real prompt text, because a check on internal state would not have caught it.

6. **A third no-progress detector was missing.** A run applied five different
   patches and never called `run_tests` once. Both existing detectors are blind
   to that: `StagnationDetector` watches test results that are never taken, and
   `RepeatDetector` watches identical calls when no two were the same. *Fix:*
   `VerificationTracker` counts edits since the last test run and the prompt
   insists on verification. This one fix took `full` from 17/18 to 18/18 and cut
   mean iterations from 6.3 to 3.9.

**The benchmark also found a defect in the gate.** A new scenario asserted its
first patch applied, and it had not: `demo_repo/cart.py` had no trailing
newline, so `difflib` ran the last removed line and the first added line
together and `git` refused the result. `NOOP_A` and `NOOP_B` had *never*
applied, which means the stagnation scenario built on them had been reaching
`STAGNATED` through rejected patches rather than through repeated failure
fingerprints — passing, and testing nothing. This is precisely the "a check that
can pass vacuously is not evidence" lesson, still live in the file that records
it. `_diff()` now asserts that every line of the diff it generates is a
well-formed diff line.

### One failure that was not the harness

`backend-unknown-action-no-early-return` still hits the budget roughly one run in
three across ablations. The model reaches the right diagnosis, patches, verifies
green, and then returns empty completions instead of calling `finish` — with the
finish directive present in the prompt on every one of those turns. Classified
**MODEL** and left alone. The obvious harness-side move, accepting a
verified-green workspace as solved without waiting for `finish`, is a design
change rather than a defect fix, and it would weaken the rule that the agent has
to claim its own result.

---

## Design

**The verifier is the only source of truth.** `finish()` records a claim; the
harness then re-runs the target test and the full suite itself. A model cannot
talk its way to a green run.

**Test files are not editable.** A test-verified agent will otherwise patch the
assertion instead of the bug and succeed every time. This is the one guard whose
absence would invalidate every number in the table, so the benchmark scores any
run that touches a test file as INVALID using the *same predicate* the guard uses
(`is_test_path`) — if the scorer and the guard could disagree, the
`no-test-guard` ablation would be grading itself.

**Context is built, not dumped.** The traceback names frames; the AST slicer
takes the enclosing symbols and walks the call graph outward `callee_depth` hops.
Depth is 2 rather than 1 because a plain `assert` failure yields a traceback with
only the test frame, so reaching the buggy function takes test → caller → callee.
The `no-slicing` ablation is the evidence that this is load-bearing.

**Mutations go through unified diffs only**, checked with `git apply --check`
before anything is written. No whole-file writes, no `bash` tool. Everything the
agent touches lives in a throwaway git worktree; the caller's checkout is never
modified.

**Bad model output never raises.** A malformed call becomes a
`ToolResult(ok=False)` whose text is written to be read by the model. Only an
unusable endpoint raises.

**Every knob lives in `config.py`**, because every knob is a potential ablation.
The five ablations are dotted config overrides and nothing else — anything that
cannot be expressed that way is a different harness, and comparing it to `full`
would not mean anything. A typo'd override raises rather than silently running
the default and reporting a delta of zero.

**Three no-progress detectors, because they see different things.**
`StagnationDetector` watches test results; `RepeatDetector` watches repeated calls
against an unchanged workspace; `VerificationTracker` watches edits that were
never tested. Each exists because a live run failed in a way the others could not
see.

---

## Layout

```
green_agent/
  types.py          records every module agrees on; Failure.fingerprint()
  config.py         every knob; .env loading; from_env_and_overrides()
  llm.py            provider client, retries, param adaptation, ReplayLLM
  prompts.py        prompt rebuilt each turn; forced HYPOTHESIS line
  loop.py           orchestrator; the least clever file, kept that way
  cli.py            `fix` and `replay`; exit 0 fixed / 1 not fixed / 2 error
  runtime/          workspace.py (git worktree, path jail), pytest_runner.py
  context/          traceback_parse.py, slicer.py (AST), budget.py
  tools/            registry.py + read_file, apply_patch, run_tests, finish
  policy/           guards.py: stagnation, repeats, verification, budget
  observability/    trace.py: one JSONL per run
demo_repo/          5-test cart with one real bug
benchmark/
  projects/         backend/, datalib/, agentlib/ -- green before injection
  tasks/            6 x {task.json, bug.patch}
  catalog.py        the bug specs; write_tasks() / load_tasks()
  make_tasks.py     regenerate tasks/ from the specs
  prepare.py        throwaway repo with the bug at HEAD; --verify-tasks logic
  ablations.py      name -> the one config knob it moves
  scoring.py        RunReport -> Result; the INVALID rule; aggregates
  run.py            the sweep
  report.py         tables and deltas
  results/          committed result files
  FINDINGS.md       one entry per investigated failure
scripts/
  initial_test.py   the standing gate; every module has a suite here
```

## Usage

### Install

```bash
uv venv
source .venv/bin/activate
uv pip install httpx pytest
```

Python 3.11 or newer. The only runtime dependencies are `httpx` (the provider
client) and `pytest` (the verifier). Every command below assumes that environment
is activated — activate it once per shell and just call `python`. (The venv cannot
be named `green_agent`: that directory is the package itself.)
`uv pip install -e .` also works and gives you a `green-agent` console script; the
examples below use `python -m green_agent.cli` so that nothing has to be installed.

**Run every command from the repository root.** Without an install, the
`green_agent` package is imported from the working directory, so anywhere else
gives you `No module named 'green_agent'`. With `uv pip install -e .` that goes
away, but `load_dotenv()` still reads `./.env`, so a live run from elsewhere
fails with a missing-API-key error even though the file exists — export the key
instead if you need to run from another directory. (`scripts/initial_test.py`
resolves everything from its own location and works from any directory.)

### The API key

Put it in `.env` at the repo root, or export it. A real environment variable
wins over the file.

```bash
echo 'OPENAI_KEY=sk-...' > .env
# or
export OPENAI_KEY=sk-...
```

`.env` is gitignored. Nothing offline needs a key — `scripts/initial_test.py`
runs the entire suite, benchmark included, with no network access at all.
Check the key works:

```bash
python scripts/initial_test.py --live    # one real model call
```

### Fixing a failing test

```bash
python -m green_agent.cli fix \
  --repo demo_repo \
  --test "tests/test_cart.py::test_discount_can_lose_free_shipping" \
  --show-diff
```

`--repo` is a path to a directory. `--test` is a pytest node id **relative to
that directory's root**. Your checkout is never modified: the agent works in a
throwaway git worktree and the diff you see is what it produced there.

Output is one line per turn, then the final diff:

```
fixed  iterations=3  tokens=3699  time=6.962s
  #1 apply_patch  ok        `shipping_fee` incorrectly checks the undiscounted subtotal,
  #2 run_tests    ok        The existing shipping-fee change should make the discounted
  #3 finish       ok        The prior patch fixed shipping eligibility by basing the fre
```

The first column is the outcome, then the tool called each turn, whether it was
accepted, and the model's hypothesis for that turn. Exit codes are meant for
scripting:

| exit | meaning |
|---|---|
| `0` | `fixed` — target test passes, no test file touched, suite still green |
| `1` | not fixed — `budget` or `stagnated` |
| `2` | harness error — bad config, no API key, unusable endpoint |

`fixed` is the only outcome the harness will report after independently re-running
the target test *and* the full suite itself. The model calling `finish()` is a
claim that gets checked, not an ending.

### Running against your own repository

```bash
python -m green_agent.cli fix \
  --repo /path/to/your/project \
  --test "tests/test_orders.py::test_totals_include_tax" \
  --show-diff
```

Four things the repository has to satisfy:

1. **Python and pytest.** The runner invokes `python -m pytest --tb=short -q` and
   the traceback parser reads that format specifically. There is no support for
   any other language or test runner — see [Tradeoffs](#tradeoffs).
2. **The target test currently fails.** If it already passes the run is refused
   with exit 2 rather than reporting a success it did not earn.
3. **The rest of the suite is green.** At `finish` the harness re-runs the whole
   suite and rejects the claim if anything else is red. On a repository with
   pre-existing failures, every `finish` is refused and the run burns its budget
   — pass `--no-suite-check` there.
4. **The node id resolves from the repository root.** pytest runs with the repo
   root as its working directory, so imports must work from there — a root
   `conftest.py`, or an installed package.

Three things that will actually catch you out:

**Git repositories are read at HEAD, not from your working tree.** If `.git`
exists, the workspace is created with `git worktree add --detach <tmp> HEAD`, so
uncommitted changes are invisible to the agent. If you have just written the
failing test, or hand-edited a bug in to try it, **commit first** — otherwise the
agent is handed a repository that does not match the one you are looking at. A
plain directory with no `.git` is copied instead, which is why `demo_repo`
behaves the way you would expect.

**Dependencies must be importable by the interpreter you launch.** The runner
uses `sys.executable`, so pytest runs inside the activated environment, not your
project's. Install the target's dependencies there first:

```bash
uv pip install -e /path/to/your/project      # or -r its requirements.txt
```

**Your tests must match the guard's globs.** Files matching `tests/**`,
`test_*.py`, `*_test.py` or `conftest.py` cannot be patched. If your tests live
somewhere else (`spec/`, `testing/`), add that pattern to
`GuardConfig.test_path_globs` in `green_agent/config.py` — otherwise the agent is
free to patch the assertion instead of the bug, and every run "succeeds".

### Flags

```
fix   --repo PATH            directory to work on                    (required)
      --test NODE_ID         pytest node id, repo-relative           (required)
      --show-diff            print the final diff
      --model NAME           default gpt-5.6-luna
      --max-iterations N     default 12
      --depth N              call-graph hops for slicing, default 2
      --context-tokens N     context ceiling, default 12000
      --no-suite-check       skip the full-suite regression sweep at finish
      --allow-test-edits     drop the test-file guard
      --trace-dir PATH       default .green_agent/traces

replay --trace FILE --repo PATH --test NODE_ID
```

`--allow-test-edits` exists so the guard can be ablated and measured. It is not a
convenience flag: with it on, "the test passes" stops meaning anything.

Knobs without a CLI flag live in `green_agent/config.py` and are edited there —
`pytest_timeout_s` (60s per pytest invocation), `max_patch_lines` (120),
`max_total_tokens`, `stagnation_threshold`, `max_unverified_edits`. Everything is
in that one file by design, because every knob is a potential ablation.

### Traces

Every run writes one JSONL file to `.green_agent/traces/`. It holds the full
model response for each turn, so it is both the audit log and a replay fixture.

```bash
python -c "
import json, sys
for line in open(sys.argv[1]):
    o = json.loads(line); e = o.pop('event')
    o.pop('response', None); o.pop('config', None)
    print(f'{e:<16} {str(o)[:150]}')
" .green_agent/traces/<file>.jsonl
```

When a run fails, `context_built` is the line to read first: it lists the symbols
that were put in front of the model. If the buggy function is not in that list,
the model never had a chance and the fault is the harness's — that single
distinction is what every entry in [`benchmark/FINDINGS.md`](benchmark/FINDINGS.md)
turned on.

Replay a recorded run with no API calls, to test a harness change against the
exact model output that exposed a bug:

```bash
python -m green_agent.cli replay \
  --trace .green_agent/traces/<file>.jsonl --repo demo_repo --test <node_id>
```

### The gate

```bash
python scripts/initial_test.py             # everything; exit 0 required
python scripts/initial_test.py --list      # suite names
python scripts/initial_test.py --only slicer
```

291 checks in about 21 seconds, no network and no API key. Adding a module means
adding a `suite_<name>()` and registering it in `SUITES`. Never commit with the
runner red.

### Troubleshooting

| symptom | cause |
|---|---|
| `No module named 'green_agent'` | not running from the repository root |
| `No API key found` even with a `.env` | same — `load_dotenv()` reads `./.env` |
| exit 2, "target test already passes" | the bug is uncommitted; the worktree is built from HEAD |
| `ModuleNotFoundError` inside the agent's test runs | target repo's dependencies not installed in the activated environment |
| every `finish` rejected, run hits budget | another test in the suite is already failing — use `--no-suite-check` |
| test run reported as timed out | suite slower than `pytest_timeout_s` (60s) in `config.py` |
| agent patches the test and "succeeds" | your tests do not match `test_path_globs` |

---

## Running the benchmark

Six tasks across three projects, five ablations, and a triage loop. The point is
not the scoreboard: it is an instrument for finding defects in the harness.

### Always verify the tasks first

```bash
python benchmark/run.py --verify-tasks
```

This asserts, for every task, that the project is green *without* the bug patch
and that exactly the one named test fails *with* it. Exit 0 means every task is
observable. Nothing else should be run until this passes — a benchmark whose
tasks were never valid produces numbers about nothing. It needs no API key.

### Run a sweep

```bash
python benchmark/run.py --ablation full --repeat 3
```

Live progress is one line per attempt; results land in
`benchmark/results/<ablation>.json` and traces in
`benchmark/traces/<ablation>/`. One task crashing is recorded as an `ERROR` row
and the sweep carries on.

```
  [ 1/18] agentlib-backoff-swapped-args          PASS     iters=3  tokens=3303   6.3s
  [ 2/18] agentlib-loose-brace-check             PASS     iters=3  tokens=3611   6.1s
  ...
solved 18/18  (100%)  invalid=0 errors=0  mean_iters=3.94  mean_tokens=4711
```

| flag | effect |
|---|---|
| `--ablation NAME` | `full`, `no-slicing`, `depth-1`, `no-repeat-detection`, `no-test-guard` |
| `--repeat N` | attempts per task, default 1 |
| `--tasks ID [ID ...]` | subset; an unknown id is an error listing the valid ones |
| `--out PATH` | default `benchmark/results/<ablation>.json` |
| `--model NAME` | override the model |
| `--verify-tasks` | verify and exit |

A full sweep of 18 runs takes about two minutes and roughly 85k tokens. To
reproduce the whole results table:

```bash
for a in full no-slicing depth-1 no-repeat-detection no-test-guard; do
  python benchmark/run.py --ablation "$a" --repeat 3
done
```

### Read the results

```bash
python benchmark/report.py benchmark/results/full.json
python benchmark/report.py benchmark/results/full.json \
                                     benchmark/results/no-slicing.json
```

One file prints per-task outcomes and the aggregate; two prints both plus the
delta, baseline first, and the list of tasks that changed outcome. Passing more
than two is an error rather than a silent truncation.

A run that modified a test file is scored **INVALID**, never solved, regardless
of what the tests reported. That rule is what stops the `no-test-guard` ablation
from being rewarded for cheating, and it uses the same `is_test_path` predicate
the guard itself uses — if the scorer and the guard could disagree, the ablation
would be grading itself.

### Add your own task

1. Put a green, dependency-light project under `benchmark/projects/<name>/`, with
   a root `conftest.py` so `tests/` can import the flat modules. Confirm it is
   green: `cd benchmark/projects/<name> && python -m pytest -q`.
2. Add a `BugSpec` to `BUGS` in [`benchmark/catalog.py`](benchmark/catalog.py) —
   a task id, the file, the target test, the bug shape, and the `old` → `new`
   text substitution.
3. Regenerate and verify:

```bash
python benchmark/make_tasks.py
python benchmark/run.py --verify-tasks
```

Patches are diffed from the file on disk, never hand-written, so a patch cannot
drift out of context with the code it patches. A spec whose anchor text no longer
matches — or matches more than once — fails loudly in `make_tasks.py` instead of
quietly producing a task that never applies. Run `make_tasks.py` after any edit
to a project.

The one constraint worth knowing before pointing this at a real-world repository:
`--verify-tasks` demands a fully green suite without the patch and exactly one
failing test with it. Repositories with flaky or pre-existing failures will not
satisfy that, which is a real limitation of how a task is defined here rather
than something a flag can work around.

### The benchmark in the gate

`suite_benchmark` runs the entire benchmark path offline under `ReplayLLM` — no
network, no API key — including a real agent run over a real task, plus the
report arithmetic and the INVALID rule on synthetic results. The gate therefore
covers the benchmark, not just the agent.

```bash
python scripts/initial_test.py --only benchmark
```

---

## Tradeoffs

**Python and pytest only.** Deliberate scope cut. The runner boundary is
pytest-shaped: `pytest_runner.run()` returns a `TestResult` and
`traceback_parse` reads pytest's `--tb=short` output. A second language would
need a `TestRunner` protocol — `run(cwd, node_id, timeout) -> TestResult` plus a
failure parser per runner — which is sketched by the existing shape but not
built. Everything above that boundary (workspace, slicer, policy, scoring) is
language-agnostic already; the AST slicer is not.

**Six tasks, not sixteen.** More tasks would have bought a smoother solve-rate
curve. Six tasks plus seven investigated failures bought six harness fixes. The
brief's priority order says a harness bug found and fixed beats three more tasks,
and on the evidence that was right — the first three tasks alone surfaced four
distinct defects.

**Bugs are injected as patches against green projects**, so a ground-truth fix
provably exists and `--verify-tasks` can assert that exactly one named test
fails. The cost is that these are single-line, single-site bugs. Real bugs are
often neither.

**The slicer resolves calls naively** — names within a module, `from x import y`
targets, and unique method names project-wide. Where it cannot resolve
confidently it returns nothing rather than guessing. That is right at this scale
and would not survive a large codebase. It also mis-resolves `ROUTES.get(...)` to
`Store.get` in the backend project, which costs a little context and nothing else.

**`renumber_bare_hunks` reconstructs information the model should have sent.**
This is the harness being generous to a known model failure mode. It is bounded:
a hunk whose context cannot be located verbatim in the file is left alone for git
to reject. I would rather the tool accept a correct edit in a wrong wrapper than
spend a turn teaching format.

**One trace file per run, JSONL, with full model responses.** It makes traces
large and makes `replay` exact. Every finding in `FINDINGS.md` came out of these
files, which is the justification.

## With more time

- **Module-level constants are invisible to the slicer.** It emits functions and
  classes, so a constant referenced by a sliced function is never shown —
  `datalib-top-n-default` exists to probe this and the agent only solves it via
  `read_file`. The fix is to include the module-level assignments that a sliced
  symbol actually references. Known, unfixed, and the clearest remaining
  context-construction gap.
- **Harder tasks.** `full` at 100% has stopped discriminating. Multi-site bugs,
  bugs where the natural first hypothesis is wrong, and bugs whose fix breaks a
  different test would all separate harness variants that these six cannot.
- **The residual `finish` flakiness.** Worth measuring whether a stricter
  `tool_choice` on the turn after a green test run removes it, rather than
  changing what counts as done.
- **Repeats are sequential and re-run the model at temperature 0**, so variance
  comes only from the endpoint. Running tasks in parallel would make sweeps cheap
  enough to raise N to where 89% vs 100% means something.
- **`--repeat` re-runs whole sweeps, not failures.** A `--rerun-failures` mode
  would make the triage loop faster.
- Statistical treatment of the deltas. Right now the report prints point
  estimates and I have described their limits in prose; confidence intervals
  would be better than prose.

## Hours

Roughly 9 hours: ~1 reading the existing harness and planning the task set, ~2 on
the three projects and the six bugs, ~2 on the runner, scoring and report, ~3 on
the triage loop — running sweeps, reading traces, fixing the harness and writing
regression checks — and ~1 on the write-up. The triage loop was the largest
single block and produced all six harness fixes, which is the outcome the brief
asked for.

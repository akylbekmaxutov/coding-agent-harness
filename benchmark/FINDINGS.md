# Findings

One entry per investigated failure. Every run that did not reach `fixed` was
opened in its trace and classified **HARNESS**, **MODEL** or **TASK**. Every
HARNESS finding has a fix in `green_agent/`, a regression check in
`scripts/initial_test.py`, and a line in the README.

The headline: **six of the seven investigations were the harness, not the
model.** The model named the correct root cause on turn 1 in most failing runs
and then could not get that fix through the harness.

Sweep history on `full` (6 tasks x 3 repeats), in order:

| after fixing | solved | mean iters | mean tokens |
|---|---|---|---|
| (first sweep, envelope + rejection fixes in) | 16/18 | 6.72 | 8,165 |
| bare hunks, stray End Patch, prompt format | 13/18 | 6.78 | 9,375 |
| stale SymbolIndex | 17/18 | 6.28 | 8,372 |
| unverified-edit directive | **18/18** | **3.94** | **4,711** |

The dip to 13/18 is real and worth keeping visible: making patches land exposed
the stale-context defect that had been masked behind them. While edits were
being rejected the agent never got far enough to be shown its own out-of-date
code, so the worse-looking sweep was the more informative one.

Traces referenced below are under `benchmark/traces/<ablation>/`.

---

### backend-unknown-action-no-early-return (first live run, 1 of 1 failed)

```
Symptom:   STAGNATED after 5 turns with an empty diff; three read_file calls
           blocked in a row
Evidence:  model_call #3 sent a correct edit wrapped in "*** Begin Patch /
           *** Update File: / *** End Patch"; apply_patch rejected it with
           "patch fragment without header" and the advice "Re-read the file to
           get exact context lines"
Cause:     HARNESS -- two defects in one turn.
           (a) The tool is named apply_patch, which is also the name of a
               widely trained tool whose argument format is that envelope, so
               the tool's own name primes a format it then refuses. What the
               envelope wrapped was a valid unified hunk.
           (b) The rejection told the model to re-read the file; RepeatDetector
               then blocked exactly that re-read, because the workspace had not
               changed. The harness gave an instruction and punished obeying it.
Fix:       unwrap_envelope() in tools/apply_patch.py rewrites the envelope as a
           unified diff; _advice() distinguishes a format error from a context
           error and stops blaming the file. RepeatDetector.note_rejection()
           counts rejections, and loop.py folds that count into the state the
           repeat guard keys on -- a rejection is new information even though
           nothing on disk moved.
Check:     suite_tools "envelope rewritten as a unified diff", "enveloped patch
           applies", "format error is named as a format error", "format error
           does not blame the file's context lines";
           suite_loop "a rejection unblocks the re-read it asked for",
           "the second identical read is still blocked"
Before:    stagnated, 5 iterations, 0 lines changed
After:     budget, 12 iterations -- the patch landed, which exposed the next one
```

---

### backend-unknown-action-no-early-return (second run, 1 of 1 failed)

```
Symptom:   BUDGET at 12 iterations. run_tests reported PASSED on turn 7 and the
           run spent five more turns without ever calling finish
Evidence:  turn 7 tool_result "PASSED in 0.126s"; turns 8, 10 and 12 returned no
           tool call at all; turns 9 and 11 re-ran the same test
Cause:     HARNESS -- directives were dropped by degenerate turns. The loop
           computes a directive for the next turn, and the no-tool-call branch
           of _turn returned None for it. So the harness said "call finish now"
           once, the model fumbled that turn, and the harness never said it
           again. The strongest signal it has evaporated exactly when the model
           was floundering.
           Contributing: with the target test green there is no failure to
           slice, and the empty-slices text says "No source could be extracted;
           use read_file" -- the opposite of the finish directive, in the same
           prompt.
Fix:       loop.py carries the incoming directive across no-action and
           parse-error turns, appending the reason rather than replacing it.
           prompts.py drops the slices block entirely when there is no failure
           and says so plainly.
Check:     suite_loop "finish directive survives one dead turn", "finish
           directive survives two dead turns", "the no-action reason is added,
           not substituted", "a green run stops telling the model to read files"
Before:    budget, 12 iterations, 13713 tokens
After:     fixed, 5 iterations, 5406 tokens
```

---

### datalib-top-n-default (2 of 3 failed)

```
Symptom:   BUDGET at 12 iterations. The model named the exact root cause on turn
           1 -- "DEFAULT_TOP_N is set to 1, change it to 3" -- and then failed
           to land it seven times
Evidence:  seven apply_patch calls, every one rejected with "No valid patches in
           input". The diffs were semantically perfect one-line changes
Cause:     HARNESS -- two more format artefacts the harness could not consume.
           (a) A closing "*** End Patch" marker with no opening one. The
               envelope unwrapper above bailed out unless it saw "*** Begin
               Patch", so a single stray trailing line made git call an
               otherwise fine diff invalid.
           (b) A bare "@@" hunk header with no line ranges -- the envelope's own
               convention. git cannot parse that with or without --recount.
Fix:       unwrap_envelope() strips Begin and End markers independently.
           renumber_bare_hunks() reconstructs the ranges: the hunk's context and
           removed lines say where it belongs, so the block is located in the
           real file and the header git wants is written. A hunk that cannot be
           located is left alone for git to reject in its own words -- guessing
           a position is how a file gets corrupted.
           Also: the system prompt and the tool schema now show the required
           format in full. Neither did before; the only guidance was "use a/ and
           b/ prefixes".
Check:     suite_tools "bare @@ hunk with a stray End Patch applies", "bare hunk
           edited the right line", "bare hunk changed nothing else",
           "renumbering finds the real line", "an unplaceable hunk is left for
           git to reject", "a numbered hunk is not rewritten"
Before:    0/3 solved, 12 iterations, ~14.9k tokens
After:     3/3 solved, mean 5.3 iterations, ~5.8k tokens
```

---

### agentlib-loose-brace-check (2 of 3 failed) -- the worst one

```
Symptom:   BUDGET at 12 iterations, with the correct fix applied on turn 1
Evidence:  turn 1 tool_result "Patch applied to jsonargs.py"; turn 2 hypothesis
           "The prior attempted fix did not persist or was incomplete"; the same
           patch re-sent repeatedly, then failing because it was already applied
Cause:     HARNESS -- the context was stale. SymbolIndex is built once in
           Agent.run() and every Symbol holds the file text it was parsed from,
           so the index is a snapshot. After a patch, every slice in every later
           prompt showed pre-patch source, and the AST spans used to map
           traceback line numbers were stale too. The model was shown its own
           unfixed code and drew the only reasonable conclusion: that its edit
           had not landed.
           Reproduced directly outside the agent: patch cart.py, re-slice with
           the existing index, and the slices still show the old line.
Fix:       SymbolIndex.refresh_if_changed(state) rebuilds when the workspace
           token moves; loop.py checks it once per turn and traces
           "index_refreshed". Keyed on the workspace token alone, not on the
           repeat-detection state, so a rejection does not force a rebuild.
Check:     suite_slicer "a changed workspace rebuilds", "an unchanged workspace
           does not rebuild", "slices follow the file after a patch", "stale
           source is gone";
           suite_loop "the prompt after a patch shows the patched line", "the
           replaced line is gone from the prompt" -- asserted against the real
           prompt text, because a check on internal state would not have caught
           this
Before:    13/18 across the sweep, mean 6.78 iterations
After:     17/18, mean 6.28 iterations
```

---

### agentlib-loose-brace-check (1 of 3 still failed after the above)

```
Symptom:   BUDGET at 12 iterations, five different patches applied, run_tests
           never called once
Evidence:  five apply_patch calls all ok=True, each inventing a more elaborate
           predicate; no run_tests event anywhere in the trace
Cause:     HARNESS -- a policy gap, and the third one of its kind. The two
           existing no-progress detectors are both blind to it: StagnationDetector
           watches test results, which never change because the tests are never
           taken; RepeatDetector watches identical calls, and no two of these
           patches were the same. The system prompt says "After a patch, call
           run_tests to check it" and nothing enforced it.
Fix:       VerificationTracker in policy/guards.py counts edits since the last
           test run; VERIFY_DIRECTIVE is injected while that count is over
           LoopConfig.max_unverified_edits (default 1). Placed above the
           read-streak nudge, which is the less specific complaint.
Check:     suite_loop "an unverified edit is called out next turn", "it keeps
           being called out while unverified", "running the tests clears it",
           "the directive did not prevent the fix"
Before:    17/18, mean 6.28 iterations, 8372 tokens
After:     18/18, mean 3.94 iterations, 4711 tokens
```

---

### The benchmark found a defect in the gate itself

```
Symptom:   A new suite_loop scenario asserted its first patch applied, and it
           had not
Evidence:  NOOP_A rejected with "patch failed: cart.py:31". demo_repo/cart.py
           had no trailing newline, so difflib runs the last removed line and
           the first added line together on one physical line and git refuses
           the result
Cause:     HARNESS (test infrastructure). NOOP_A and NOOP_B had never applied.
           The stagnation scenario built on them was reaching STAGNATED through
           rejected patches rather than through repeated failure fingerprints --
           passing, and testing nothing. Exactly the "a check that can pass
           vacuously is not evidence" lesson, still live in the file that
           records it.
Fix:       Trailing newlines on the demo fixtures, and _diff() now asserts every
           line of the diff it generates is a well-formed diff line, so this
           class of silent breakage cannot come back.
Check:     the assertion inside _diff() itself, plus "the first patch really
           applied" in the unverified-edits scenario
```

---

### backend-unknown-action-no-early-return (residual, ~1 in 3 across ablations)

```
Symptom:   BUDGET at 12 iterations on the easiest task in the set, intermittently
Evidence:  the model reaches the right diagnosis, patches, verifies green, and
           then produces empty completions instead of calling finish. The finish
           directive is present in the prompt on every one of those turns
Cause:     MODEL, as far as the trace shows. The harness now says the right
           thing, keeps saying it across dead turns, and force_tool is set after
           a no-action turn. The remaining variance is the model returning
           finish_reason="stop" with empty content at reasoning_effort="none".
Fix:       none. Recorded rather than papered over. The obvious harness-side
           move -- accept a verified-green workspace as solved without waiting
           for finish -- would be a design change, not a defect fix, and it
           would weaken the rule that the agent must claim its result. Noted in
           the README under "With more time".
```

---

### no-slicing: the tasks that fail are the tasks the design predicts

```
Symptom:   3/18, versus 18/18 on full
Evidence:  the only task solved is backend-unknown-action-no-early-return, the
           one whose bug raises a TypeError so the traceback names service.py.
           For the other five the failure is a plain assertion, the traceback
           names only the test file, and a whole-file dump of the test file
           does not contain the bug
Cause:     Not a defect. This is the ablation measuring what it was built to
           measure: the call-graph expansion, not the file dump, is what puts
           the buggy function in front of the model.
Note:      the agent still has read_file and could in principle recover. It
           mostly does not -- it stagnates or exhausts the budget instead --
           which says the slicer is not a convenience, it is the mechanism.
```

---

### depth-1: separates the two kinds of task, as intended

```
Symptom:   8/18. datalib-top-n-default keeps 3/3; the four "traceback points
           away" tasks drop to 1/3
Evidence:  per-task table in the delta output
Cause:     Not a defect. top-n-default's bug is a module constant that no slice
           ever contains at any depth, so the model finds it with read_file and
           the depth knob is irrelevant to it. The other four need test ->
           caller -> callee, which is precisely the hop depth 1 removes.
Note:      a real gap the benchmark exposed but which was left unfixed: the
           slicer emits functions and classes only, so a module-level constant
           referenced by a sliced function is never shown. See the README.
```

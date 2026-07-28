# agentlib

A miniature version of the library this harness is built out of.

    registry.py   name -> handler plus schema, with required-argument checking
    retry.py      RetryPolicy, capped geometric backoff, run_with_retries
    jsonargs.py   pull a JSON object out of a fenced or prosey model reply
    budget.py     turn and token budget, whichever trips first

## Why it exists

It is the most interesting of the three projects because its bugs are the ones
a real harness actually has. `green_agent` has a retry policy, a fenced-JSON
argument parser and a budget of its own; the mistakes available here — helper
arguments passed in the wrong order, a brace check loosened from `and` to `or`
— are mistakes that were plausibly one keystroke away in the harness itself.

Running the agent against a small copy of its own domain also keeps the
benchmark honest about difficulty: these are not toy arithmetic bugs, they are
the kind where the failing assertion is two calls away from the wrong line.

Every test passes before a bug is injected. `conftest.py` exists only to put the
project root on `sys.path`.

# datalib

Parsing and validation over delimited text. Four stages, each usable alone:

    loader.py     delimited text -> list of string records, with field-count checks
    coerce.py     cell strings -> int / float / bool / str, errors naming the field
    schema.py     declared fields -> a list of validation messages
    aggregate.py  group_by, summarise (count / total / mean), top_groups

## Why it exists

The backend project fails on control flow; this one fails on values. Its bug
shapes are the ones that only show up at the edges — a group of exactly one, an
empty input, a default that lives in a module constant rather than in the
function that uses it. Those are worth having in a benchmark because they are
invisible in the middle of the range: a test with three rows per group passes
either way, and only the one-row case tells you anything.

Every test passes before a bug is injected. `conftest.py` exists only to put the
project root on `sys.path`.

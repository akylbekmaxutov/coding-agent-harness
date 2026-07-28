# backend

Service-layer logic with no framework, no network and no database: a request is
a `dict`, a response is a `dict` with a status, and the store is a dictionary.

    errors.py       AppError subclasses and the one place status codes are decided
    auth.py         role -> rights, and the `require` gate
    pagination.py   page_count / window / paginate over a materialised list
    store.py        the dict pretending to be a database
    service.py      the route table and the try/except that maps errors to statuses

## Why it exists

It is the ordinary shape of application code, and its bugs are the ordinary
shape too: arithmetic on page boundaries, a guard that stops guarding, a
permission check on the wrong side of a branch. None of them raise where they
are written, which is what makes them a fair test of context construction —
the traceback names the caller, and the harness has to reach the callee.

Every test passes before a bug is injected. `conftest.py` exists only to put
the project root on `sys.path`.

"""Regenerate benchmark/tasks/ from the declarative specs in catalog.py.

    python benchmark/make_tasks.py

Run this after editing a project under benchmark/projects/. Patch context is
diffed from the file on disk, so a stale patch is impossible; a spec whose
anchor text no longer matches fails loudly here rather than silently producing
a task that never applies.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.catalog import BUGS, write_tasks


def main() -> int:
    for directory in write_tasks():
        print(f"wrote {directory.relative_to(Path.cwd()) if directory.is_relative_to(Path.cwd()) else directory}")
    print(f"{len(BUGS)} tasks regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

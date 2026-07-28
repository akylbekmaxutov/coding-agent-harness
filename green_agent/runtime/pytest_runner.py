from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from ..types import TestResult

# 0 all passed, 1 tests failed, 2 interrupted (usually a collection error),
# 3 internal error, 4 usage error, 5 nothing collected.
_COLLECT_FAILURE_CODES = {2, 3, 4, 5}


def run(cwd: str | Path, node_id: str | None = None, timeout_s: int = 60) -> TestResult:
    cmd = [sys.executable, "-m", "pytest", "--tb=long", "-q", "--color=no", "-p", "no:cacheprovider"]
    if node_id:
        cmd.append(node_id)

    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTEST_ADDOPTS": "",
        "NO_COLOR": "1",
    }

    started = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        output, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        # Kill the whole process group: an agent patch that spawns or forks
        # would otherwise leave orphans behind and hang the next run.
        timed_out = True
        _kill_group(proc)
        output, _ = proc.communicate()
        output = (output or "") + f"\n[harness] killed after {timeout_s}s timeout"

    duration = time.monotonic() - started
    rc = proc.returncode

    return TestResult(
        passed=(rc == 0 and not timed_out),
        failures=(),  # populated by context.traceback_parse in the next step
        duration_s=round(duration, 3),
        raw_output=output or "",
        timed_out=timed_out,
        collect_error=(rc in _COLLECT_FAILURE_CODES and not timed_out),
    )


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
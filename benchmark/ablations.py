"""Ablations: a name, and the one config knob it moves.

Each entry is a dict of dotted overrides and nothing else. Anything that cannot
be expressed as a config override is not an ablation -- it is a different
harness, and comparing it to `full` would not mean anything.
"""

from __future__ import annotations

from green_agent.config import Config

ABLATIONS: dict[str, dict] = {
    # The shipped configuration. Every other row is this minus one thing.
    "full": {},
    # Whole-file context from the same traceback entry points: measures the
    # AST slicer and its call-graph expansion.
    "no-slicing": {"context.whole_file_context": True},
    # One hop instead of two: measures reaching a callee the traceback never
    # named, which is the whole reason the depth knob exists.
    "depth-1": {"context.callee_depth": 1},
    # No blocking of identical calls, no force-an-edit directive: measures the
    # policy that ended the first live run's read loop.
    "no-repeat-detection": {"loop.repeat_detection": False},
    # The agent may edit test files: measures the guard. Runs that take the
    # offer are scored INVALID, never solved.
    "no-test-guard": {"guards.forbid_test_file_edits": False},
}


def config_for(name: str, **extra) -> Config:
    if name not in ABLATIONS:
        raise KeyError(f"unknown ablation {name!r}; have {', '.join(ABLATIONS)}")
    return Config.from_env_and_overrides(**{**ABLATIONS[name], **extra})


def describe(name: str) -> str:
    overrides = ABLATIONS[name]
    if not overrides:
        return "shipped defaults"
    return ", ".join(f"{key}={value}" for key, value in sorted(overrides.items()))

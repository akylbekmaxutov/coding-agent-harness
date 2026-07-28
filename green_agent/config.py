from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    name: str = "gpt-5.6-luna"
    reasoning_effort: str | None = "none"
    temperature: float = 0.0
    max_output_tokens: int = 2048
    timeout_s: int = 120
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_KEY"
    max_retries: int = 4


@dataclass(frozen=True)
class ContextConfig:
    max_context_tokens: int = 12_000
    # Depth 2, not 1: a plain `assert` failure yields a traceback with only
    # the test frame, so reaching the buggy function takes test -> caller ->
    # callee. Measured on demo_repo; an ablation axis for the benchmark.
    callee_depth: int = 2
    include_external_frames: bool = False
    max_raw_output_chars: int = 4_000
    # The no-slicing baseline: same entry points, whole files instead of
    # symbols, so the ablation moves AST extraction and nothing else.
    whole_file_context: bool = False


@dataclass(frozen=True)
class LoopConfig:
    max_iterations: int = 12
    max_total_tokens: int = 150_000
    stagnation_threshold: int = 2
    # Consecutive read-only turns before the prompt insists on an edit.
    max_read_streak: int = 2
    # Blocked identical calls tolerated before the run is called stagnant.
    max_blocked_repeats: int = 3
    # Turns off RepeatDetector entirely -- both the blocking of identical calls
    # and the force-an-edit directive. An ablation axis, never a fix.
    repeat_detection: bool = True
    # Successful patches tolerated before the prompt insists on a test run.
    max_unverified_edits: int = 1


@dataclass(frozen=True)
class GuardConfig:
    forbid_test_file_edits: bool = True
    test_path_globs: tuple[str, ...] = ("tests/**", "test_*.py", "*_test.py", "conftest.py")
    run_full_suite_on_success: bool = True
    max_patch_lines: int = 120
    pytest_timeout_s: int = 60


@dataclass(frozen=True)
class Config:
    model: ModelConfig = ModelConfig()
    context: ContextConfig = ContextConfig()
    loop: LoopConfig = LoopConfig()
    guards: GuardConfig = GuardConfig()
    trace_dir: str = ".green_agent/traces"

    @classmethod
    def from_env_and_overrides(cls, dotenv_path: str | Path = ".env", **overrides) -> "Config":
        """Defaults plus dotted overrides: `context.callee_depth=1`.

        An ablation is exactly this and nothing else, which is what keeps two
        result files comparable: one named knob moves, the rest of the run is
        byte-identical. An unknown section or field is an error, not a silent
        no-op -- a typo'd ablation that quietly runs the default config would
        report a delta of zero and look like a finding.
        """
        load_dotenv(dotenv_path)
        sections: dict[str, object] = {
            "model": ModelConfig(),
            "context": ContextConfig(),
            "loop": LoopConfig(),
            "guards": GuardConfig(),
        }
        top: dict[str, object] = {}
        for dotted, value in overrides.items():
            section, _, name = dotted.partition(".")
            if not name:
                if section not in {f for f in cls.__dataclass_fields__} - set(sections):
                    raise KeyError(f"unknown config field {dotted!r}")
                top[section] = value
                continue
            if section not in sections:
                raise KeyError(f"unknown config section {section!r} in {dotted!r}")
            if name not in type(sections[section]).__dataclass_fields__:
                raise KeyError(f"unknown field {name!r} in config section {section!r}")
            sections[section] = replace(sections[section], **{name: value})
        return cls(**sections, **top)


def load_dotenv(path: str | Path = ".env", override: bool = False) -> dict[str, str]:
    """Read a .env file into os.environ. Real env vars win unless override."""
    p = Path(path)
    if not p.is_file():
        return {}
    applied: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def resolve_api_key(cfg: ModelConfig, dotenv_path: str | Path = ".env") -> str:
    """Process env, then .env, then a readable error."""
    key = os.environ.get(cfg.api_key_env, "")
    if not key:
        load_dotenv(dotenv_path)
        key = os.environ.get(cfg.api_key_env, "")
    if not key:
        raise RuntimeError(
            f"No API key found. Set ${cfg.api_key_env} in the environment or in "
            f"{dotenv_path} (see .env.example)."
        )
    return key
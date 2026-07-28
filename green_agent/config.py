from __future__ import annotations

import os
from dataclasses import dataclass
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


@dataclass(frozen=True)
class LoopConfig:
    max_iterations: int = 12
    max_total_tokens: int = 150_000
    stagnation_threshold: int = 2


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
    def from_env_and_overrides(cls, **overrides) -> "Config":
        raise NotImplementedError


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
from __future__ import annotations

import json
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .config import ModelConfig, resolve_api_key
from .types import ToolCall

_HYPOTHESIS_RE = re.compile(r"^\s*HYPOTHESIS\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# Providers rename parameters between model generations. Renames are applied
# first; anything else unsupported is dropped so the run continues.
_PARAM_RENAMES = {"max_tokens": "max_completion_tokens"}
_KNOWN_PARAMS = (
    "max_tokens", "max_completion_tokens", "reasoning_effort",
    "parallel_tool_calls", "temperature", "tool_choice",
)


@dataclass
class Completion:
    text: str
    tool_call: ToolCall | None
    prompt_tokens: int
    completion_tokens: int
    raw: dict[str, Any] = field(default_factory=dict)
    parse_error: str | None = None
    extra_calls_dropped: int = 0

    @property
    def hypothesis(self) -> str:
        m = _HYPOTHESIS_RE.search(self.text or "")
        return m.group(1).strip() if m else ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLM(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        force_tool: bool = False,
    ) -> Completion: ...


class LLMTransportError(RuntimeError):
    """Endpoint unusable after retries. Distinct from bad model output."""


class OpenAICompatibleLLM:
    def __init__(
        self,
        cfg: ModelConfig,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.cfg = cfg
        self._api_key = (api_key or "") if client is not None else (api_key or resolve_api_key(cfg))
        self._client = client or httpx.Client(
            base_url=cfg.base_url.rstrip("/"),
            timeout=cfg.timeout_s,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        self._dropped: set[str] = set()

    def complete(self, messages, tools, force_tool: bool = False) -> Completion:
        payload = self._build_payload(messages, tools)
        if force_tool and tools:
            # Used only after a turn that produced prose and no action: the
            # model has already shown it will not choose on its own.
            payload["tool_choice"] = "required"
        return normalise_response(self._post_with_retries(payload))

    def close(self) -> None:
        self._client.close()

    def _build_payload(self, messages, tools) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.cfg.name,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_output_tokens,
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = False
        if self.cfg.reasoning_effort:
            payload["reasoning_effort"] = self.cfg.reasoning_effort
        for param in self._dropped:
            if param in _PARAM_RENAMES:
                payload[_PARAM_RENAMES[param]] = payload.pop(param, None)
            else:
                payload.pop(param, None)
        return payload

    def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_err = ""
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self._client.post("/chat/completions", json=payload)
            except httpx.RequestError as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                self._sleep(attempt)
                continue

            if resp.status_code == 200:
                return resp.json()

            body = resp.text[:500]

            if resp.status_code == 400 and self._adapt(body, payload):
                continue

            last_err = f"HTTP {resp.status_code}: {body}"
            if resp.status_code in _RETRY_STATUS:
                self._sleep(attempt, resp.headers.get("retry-after"))
                continue
            break

        raise LLMTransportError(f"Model call failed after retries. {last_err}")

    def _adapt(self, body: str, payload: dict[str, Any]) -> bool:
        """Rename or drop a parameter the server rejected. True means retry."""
        param = _rejected_param(body)
        if not param or param in self._dropped:
            return False
        self._dropped.add(param)
        if param in _PARAM_RENAMES:
            payload[_PARAM_RENAMES[param]] = payload.pop(param, None)
        else:
            payload.pop(param, None)
        return True

    @property
    def degraded_params(self) -> set[str]:
        """Parameters this endpoint refused. Belongs in the run trace."""
        return set(self._dropped)

    def _sleep(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after:
            try:
                time.sleep(min(float(retry_after), 30.0))
                return
            except ValueError:
                pass
        time.sleep(min(2 ** attempt, 16) * (0.5 + random.random() / 2))


def _rejected_param(body: str) -> str | None:
    try:
        param = ((json.loads(body).get("error") or {}).get("param") or "").strip()
    except (json.JSONDecodeError, AttributeError):
        param = ""
    if param:
        return param
    return next((p for p in _KNOWN_PARAMS if p in body), None)


def normalise_response(data: dict[str, Any]) -> Completion:
    """Provider response -> Completion. Shared by live and replay clients."""
    choices = data.get("choices") or []
    message = (choices[0].get("message") if choices else {}) or {}
    usage = data.get("usage") or {}

    text = message.get("content") or ""
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))

    raw_calls = message.get("tool_calls") or []
    tool_call: ToolCall | None = None
    parse_error: str | None = None

    if raw_calls:
        first = raw_calls[0]
        fn = first.get("function") or {}
        name = fn.get("name") or "<missing>"
        args, parse_error = _parse_arguments(name, fn.get("arguments"))
        tool_call = ToolCall(
            name=name,
            arguments=args,
            call_id=first.get("id") or f"call_{uuid.uuid4().hex[:8]}",
        )

    return Completion(
        text=text,
        tool_call=tool_call,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        raw=data,
        parse_error=parse_error,
        extra_calls_dropped=max(0, len(raw_calls) - 1),
    )


def _parse_arguments(name: str, args_text: Any) -> tuple[dict[str, Any], str | None]:
    """Returns ({}, reason). The reason is written to be read by the model."""
    if isinstance(args_text, dict):
        return args_text, None
    if args_text in (None, ""):
        return {}, None
    if not isinstance(args_text, str):
        return {}, f"Arguments for {name} had unexpected type {type(args_text).__name__}."

    try:
        parsed = json.loads(args_text)
    except json.JSONDecodeError as exc:
        stripped = args_text.strip().strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {}, (
                f"Could not parse arguments for {name} as JSON ({exc.msg} at "
                f"char {exc.pos}). Re-send the call with valid JSON arguments."
            )
    if not isinstance(parsed, dict):
        return {}, f"Arguments for {name} must be a JSON object, got {type(parsed).__name__}."
    return parsed, None


class ReplayLLM:
    """Serves recorded completions so runs are deterministic and key-free."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._queue = list(events)
        self.calls: list[dict[str, Any]] = []

    @classmethod
    def from_trace(cls, path: str) -> "ReplayLLM":
        events = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                if obj.get("event") == "model_call" and "response" in obj:
                    events.append(obj["response"])
        return cls(events)

    def complete(self, messages, tools, force_tool: bool = False) -> Completion:
        self.calls.append({"messages": messages, "tools": tools, "force_tool": force_tool})
        if not self._queue:
            raise LLMTransportError("Replay exhausted: trace has no further model calls.")
        return normalise_response(self._queue.pop(0))
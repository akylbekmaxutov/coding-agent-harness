"""Integration checks for the harness. Run this after every new module.

    python scripts/initial_test.py              # all offline suites
    python scripts/initial_test.py --only llm   # one suite
    python scripts/initial_test.py --list       # what exists
    python scripts/initial_test.py --live       # one real model call

Offline suites use fakes and temp dirs: no network, no API key, no state left
behind. Exit code is 0 only if every check passes, so this is CI-ready.

To add a suite: write `def suite_<name>() -> None`, call `check(...)` inside,
and register it in SUITES at the bottom.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from green_agent.config import GuardConfig, ModelConfig, load_dotenv, resolve_api_key
from green_agent.llm import LLMTransportError, OpenAICompatibleLLM, ReplayLLM
from green_agent.runtime import pytest_runner
from green_agent.runtime.workspace import PathEscape, Workspace

DEMO = Path(__file__).resolve().parents[1] / "demo_repo"

_results: list[tuple[str, bool]] = []
_GREEN, _RED, _DIM, _OFF = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


def check(label: str, cond: bool, detail: str = "") -> None:
    ok = bool(cond)
    _results.append((label, ok))
    mark = f"{_GREEN}ok{_OFF}" if ok else f"{_RED}FAIL{_OFF}"
    line = f"  [{mark}] {label}"
    if detail and not ok:
        line += f"\n        {_DIM}{detail}{_OFF}"
    print(line)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

TOOLS = [{"name": "run_tests", "parameters": {"type": "object", "properties": {}}}]


def _response(content="", tool_calls=None, prompt=11, completion=7):
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [{"message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


def _mock_llm(handler, cfg: ModelConfig | None = None) -> OpenAICompatibleLLM:
    cfg = cfg or ModelConfig()
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=cfg.base_url,
        headers={"Authorization": "Bearer test"},
    )
    return OpenAICompatibleLLM(cfg, api_key="test", client=client)


# ---------------------------------------------------------------------------
# suites
# ---------------------------------------------------------------------------


def suite_config() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, ".env")
        Path(path).write_text(
            '# comment\nOPENAI_KEY="sk-from-file"\nexport OTHER_VAR=plain\nMALFORMED\n'
        )
        os.environ.pop("OPENAI_KEY", None)
        os.environ.pop("OTHER_VAR", None)

        load_dotenv(path)
        check("quoted value unquoted", os.environ.get("OPENAI_KEY") == "sk-from-file")
        check("export prefix stripped", os.environ.get("OTHER_VAR") == "plain")
        check("malformed line ignored", "MALFORMED" not in os.environ)

        os.environ["OPENAI_KEY"] = "sk-from-shell"
        load_dotenv(path)
        check("real env wins over .env", os.environ["OPENAI_KEY"] == "sk-from-shell")
        check("resolve_api_key finds it", resolve_api_key(ModelConfig(), path) == "sk-from-shell")

        os.environ.pop("OPENAI_KEY", None)
        try:
            resolve_api_key(ModelConfig(), os.path.join(d, "missing.env"))
            check("missing key raises a readable error", False)
        except RuntimeError as exc:
            check("missing key raises a readable error", "OPENAI_KEY" in str(exc))


def suite_llm() -> None:
    seen: dict = {}

    def capture(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_response("HYPOTHESIS: x"))

    _mock_llm(capture).complete([{"role": "user", "content": "hi"}], TOOLS)
    check("model name sent", seen.get("model") == "gpt-5.6-luna", seen.get("model"))
    check("reasoning_effort=none sent", seen.get("reasoning_effort") == "none")
    check("temperature pinned to 0", seen.get("temperature") == 0.0)
    check("parallel tool calls disabled", seen.get("parallel_tool_calls") is False)
    check("tool_choice is auto, not required", seen.get("tool_choice") == "auto")
    check(
        "tools wrapped in function envelope",
        seen["tools"][0]["type"] == "function"
        and seen["tools"][0]["function"]["name"] == "run_tests",
    )

    calls = [{"id": "call_1", "function": {"name": "apply_patch", "arguments": '{"diff": "--- a/x.py"}'}}]
    c = _mock_llm(lambda r: httpx.Response(200, json=_response("HYPOTHESIS: off-by-one", calls))).complete([], TOOLS)
    check("tool name normalised", c.tool_call and c.tool_call.name == "apply_patch")
    check("arguments decoded to dict", c.tool_call.arguments == {"diff": "--- a/x.py"})
    check("call_id preserved", c.tool_call.call_id == "call_1")
    check("hypothesis extracted", c.hypothesis == "off-by-one", repr(c.hypothesis))
    check("token usage captured", c.total_tokens == 18)

    many = [
        {"id": "a", "function": {"name": "read_file", "arguments": "{}"}},
        {"id": "b", "function": {"name": "apply_patch", "arguments": "{}"}},
        {"id": "c", "function": {"name": "finish", "arguments": "{}"}},
    ]
    c = _mock_llm(lambda r: httpx.Response(200, json=_response("", many))).complete([], TOOLS)
    check("only first call kept", c.tool_call.name == "read_file")
    check("dropped calls counted for the trace", c.extra_calls_dropped == 2)

    bad = [{"id": "x", "function": {"name": "apply_patch", "arguments": "{diff: unquoted}"}}]
    c = _mock_llm(lambda r: httpx.Response(200, json=_response("", bad))).complete([], TOOLS)
    check("malformed args do not raise", c.parse_error is not None)
    check("parse error is model-readable", "valid JSON" in (c.parse_error or ""))
    check("arguments default to empty dict", c.tool_call.arguments == {})

    fenced = [{"id": "x", "function": {"name": "read_file", "arguments": '```json\n{"path":"a.py"}\n```'}}]
    c = _mock_llm(lambda r: httpx.Response(200, json=_response("", fenced))).complete([], TOOLS)
    check("markdown-fenced arguments recovered", c.tool_call.arguments == {"path": "a.py"})

    n = {"i": 0}

    def flaky(request):
        n["i"] += 1
        return (
            httpx.Response(503, text="upstream busy")
            if n["i"] < 3
            else httpx.Response(200, json=_response("ok"))
        )

    c = _mock_llm(flaky, ModelConfig(max_retries=5)).complete([], TOOLS)
    check("retries transient 5xx then succeeds", c.text == "ok" and n["i"] == 3)

    m = {"i": 0}

    def unauthorised(request):
        m["i"] += 1
        return httpx.Response(401, text="invalid key")

    try:
        _mock_llm(unauthorised).complete([], TOOLS)
        check("401 raises transport error", False)
    except LLMTransportError:
        check("401 raises transport error", True)
        check("401 not retried", m["i"] == 1, f"attempts={m['i']}")

    sent: list[bool] = []

    def rejects_reasoning(request):
        body = json.loads(request.content)
        sent.append("reasoning_effort" in body)
        if "reasoning_effort" in body:
            return httpx.Response(400, text="Unrecognized request argument: reasoning_effort")
        return httpx.Response(200, json=_response("ok"))

    c = _mock_llm(rejects_reasoning).complete([], TOOLS)
    check("unknown reasoning_effort dropped, not fatal", c.text == "ok" and sent == [True, False])

    bodies: list[dict] = []

    def rejects_max_tokens(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "max_tokens" in body:
            return httpx.Response(400, json={"error": {
                "message": "Unsupported parameter: 'max_tokens' is not supported with this "
                           "model. Use 'max_completion_tokens' instead.",
                "type": "invalid_request_error", "param": "max_tokens",
                "code": "unsupported_parameter"}})
        return httpx.Response(200, json=_response("ok"))

    llm = _mock_llm(rejects_max_tokens)
    c = llm.complete([], TOOLS)
    check("max_tokens rename recovers the call", c.text == "ok")
    check("renamed to max_completion_tokens",
          bodies[-1].get("max_completion_tokens") == 2048 and "max_tokens" not in bodies[-1])
    check("rename is remembered for later calls",
          llm.complete([], TOOLS).text == "ok" and len(bodies) == 3
          and "max_tokens" not in bodies[-1])
    check("degraded params exposed for the trace", llm.degraded_params == {"max_tokens"})

    seen_temp: list[dict] = []

    def rejects_temperature(request):
        body = json.loads(request.content)
        seen_temp.append(body)
        if "temperature" in body:
            return httpx.Response(400, json={"error": {
                "message": "Unsupported value: 'temperature' does not support 0.0 with this "
                           "model. Only the default (1) is supported.",
                "param": "temperature", "code": "unsupported_value"}})
        return httpx.Response(200, json=_response("ok"))

    llm = _mock_llm(rejects_temperature)
    check("unsupported temperature dropped, not fatal", llm.complete([], TOOLS).text == "ok")
    check("temperature loss is recorded", llm.degraded_params == {"temperature"})


def suite_replay() -> None:
    r = ReplayLLM([_response("HYPOTHESIS: a"), _response("HYPOTHESIS: b")])
    a, b = r.complete([], []), r.complete([], [])
    check("replay returns recorded turns in order", (a.hypothesis, b.hypothesis) == ("a", "b"))
    check("replay records what it was asked", len(r.calls) == 2)
    try:
        r.complete([], [])
        check("exhausted replay fails loudly", False)
    except LLMTransportError:
        check("exhausted replay fails loudly", True)


def suite_workspace() -> None:
    with Workspace(DEMO, test_globs=GuardConfig().test_path_globs) as ws:
        check("workspace materialised", (ws.root / "cart.py").is_file())
        check("source repo untouched by setup", (DEMO / "cart.py").is_file())
        check("baseline commit recorded", bool(ws.baseline))

        check("test file recognised", ws.is_test_file("tests/test_cart.py"))
        check("conftest recognised", ws.is_test_file("conftest.py"))
        check("source file not a test file", not ws.is_test_file("cart.py"))

        check("relative path resolves inside", ws.resolve("cart.py").is_file())
        for attack in ("../../etc/passwd", "/etc/passwd", "tests/../../outside.py"):
            try:
                ws.resolve(attack)
                check(f"path escape refused: {attack}", False)
            except PathEscape:
                check(f"path escape refused: {attack}", True)

        target = ws.resolve("cart.py")
        original = target.read_text()
        mark = ws.checkpoint("before-edit")

        target.write_text(original.replace("TAX_RATE = 0.20", "TAX_RATE = 0.99"))
        check("edit visible in diff", "TAX_RATE = 0.99" in ws.diff())
        check("changed file listed", ws.changed_files() == ["cart.py"], ws.changed_files())
        check("original still untouched on disk",
              "TAX_RATE = 0.20" in (DEMO / "cart.py").read_text())

        (ws.root / "junk.py").write_text("# stray file\n")
        check("untracked file appears in diff", "junk.py" in ws.diff())

        ws.restore(mark)
        check("restore reverts the edit", target.read_text() == original)
        check("restore removes untracked files", not (ws.root / "junk.py").exists())
        check("restore leaves a clean diff", ws.diff().strip() == "")

        root = ws.root
    check("teardown removes the workspace", not root.exists())


def suite_pytest_runner() -> None:
    with Workspace(DEMO) as ws:
        whole = pytest_runner.run(ws.root, timeout_s=60)
        check("suite reports failure", whole.passed is False)
        check("not misreported as a collection error", whole.collect_error is False)
        check("output captured", "test_discount_can_lose_free_shipping" in whole.raw_output)
        check("duration measured", whole.duration_s > 0)

        node = "tests/test_cart.py::test_discount_can_lose_free_shipping"
        one = pytest_runner.run(ws.root, node, timeout_s=60)
        check("target test fails on its own", one.passed is False)
        check("single test is faster than the suite", one.duration_s <= whole.duration_s + 0.5)

        green = pytest_runner.run(ws.root, "tests/test_cart.py::test_subtotal", timeout_s=60)
        check("passing test reports passed", green.passed is True)
        check("passing test has no failures", green.failures == ())

        ws.resolve("cart.py").write_text("def broken(:\n")
        broken = pytest_runner.run(ws.root, node, timeout_s=60)
        check("syntax error flagged as collection error", broken.collect_error is True)
        check("collection error is not a pass", broken.passed is False)

        ws.resolve("cart.py").write_text("while True:\n    pass\n")
        hung = pytest_runner.run(ws.root, node, timeout_s=3)
        check("infinite loop is killed", hung.timed_out is True)
        check("timeout is not a pass", hung.passed is False)
        check("timeout noted in output", "timeout" in hung.raw_output)


# ---------------------------------------------------------------------------
# live
# ---------------------------------------------------------------------------


def live() -> int:
    load_dotenv()
    cfg = ModelConfig()
    print(f"model    : {cfg.name}\nendpoint : {cfg.base_url}\nkey var  : ${cfg.api_key_env}\n")
    try:
        llm = OpenAICompatibleLLM(cfg)
        c = llm.complete(
            [{"role": "user", "content": "Reply with exactly: HYPOTHESIS: the harness is wired up"}],
            [],
        )
    except (RuntimeError, LLMTransportError) as exc:
        print(f"{_RED}live call failed{_OFF}: {exc}")
        return 1
    print(f"text       : {c.text!r}")
    print(f"hypothesis : {c.hypothesis!r}")
    print(f"tokens     : {c.prompt_tokens} in / {c.completion_tokens} out")
    return 0


SUITES = {
    "config": suite_config,
    "llm": suite_llm,
    "replay": suite_replay,
    "workspace": suite_workspace,
    "pytest_runner": suite_pytest_runner,
    # "traceback_parse": suite_traceback_parse,   # next
    # "slicer": suite_slicer,
}


def main(argv: list[str]) -> int:
    if "--live" in argv:
        return live()
    if "--list" in argv:
        print("\n".join(SUITES))
        return 0

    selected = SUITES
    if "--only" in argv:
        name = argv[argv.index("--only") + 1]
        if name not in SUITES:
            print(f"unknown suite {name!r}; try --list")
            return 2
        selected = {name: SUITES[name]}

    for name, fn in selected.items():
        print(f"\n{name}")
        fn()

    total = len(_results)
    failed = [label for label, ok in _results if not ok]
    print(f"\n{total - len(failed)}/{total} checks passed")
    for label in failed:
        print(f"  {_RED}failed{_OFF}: {label}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
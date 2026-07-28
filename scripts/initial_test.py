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

import difflib
import json
import os
import pathlib
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from green_agent.config import GuardConfig, ModelConfig, load_dotenv, resolve_api_key
from green_agent.llm import LLMTransportError, OpenAICompatibleLLM, ReplayLLM
from green_agent.config import ContextConfig
from green_agent.context.budget import estimate_tokens, fit, signature_only
from green_agent.context.slicer import SymbolIndex, slices_for_failure
from green_agent.context.traceback_parse import parse_failures
from green_agent.runtime import pytest_runner
from green_agent.config import Config, GuardConfig as _GC, LoopConfig
from green_agent.llm import ReplayLLM
from green_agent.loop import Agent
from green_agent.runtime.workspace import PathEscape, Workspace
from green_agent.tools.apply_patch import (
    changed_line_count,
    changed_paths,
    renumber_bare_hunks,
    unwrap_envelope,
)
from green_agent.tools.registry import ToolContext, dispatch, schemas
from green_agent.types import Frame, ToolCall

DEMO = Path(__file__).resolve().parents[1] / "demo_repo"
TASKS = Path(__file__).resolve().parents[1] / "benchmark" / "tasks"

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


def suite_traceback_parse() -> None:
    with Workspace(DEMO) as ws:
        node = "tests/test_cart.py::test_discount_can_lose_free_shipping"
        result = pytest_runner.run(ws.root, node, timeout_s=60)

        check("one failure parsed", len(result.failures) == 1, str(len(result.failures)))
        f = result.failures[0]
        check("node id recovered", f.test_id == node, f.test_id)
        check("assertion classified", f.exc_type == "AssertionError", f.exc_type)
        check("message captured", "56.16" in f.message and "61.16" in f.message, f.message)
        check("frame located in repo", f.frames[-1].in_repo is True)
        check("frame function named", f.frames[-1].function.startswith("test_discount"))
        check("frame path is repo-relative", f.frames[-1].path == "tests/test_cart.py")

        second = pytest_runner.run(ws.root, node, timeout_s=60)
        check("fingerprint stable across runs",
              second.failures[0].fingerprint() == f.fingerprint())

        source = ws.resolve("cart.py")
        source.write_text("\n" * 5 + source.read_text())
        shifted = pytest_runner.run(ws.root, node, timeout_s=60)
        check("fingerprint survives line shifts",
              shifted.failures[0].fingerprint() == f.fingerprint())

        whole = pytest_runner.run(ws.root, timeout_s=60)
        check("passing tests produce no failure records", len(whole.failures) == 1)

    # Synthetic reports: shapes the demo repo cannot produce on demand.
    deep = parse_failures(
        "=================================== FAILURES ===================================\n"
        "_________________________________ test_raises __________________________________\n"
        "tests/test_a.py:10: in test_raises\n"
        "    assert outer(0) == 1\n"
        "/usr/lib/python3.12/site-packages/thing.py:40: in helper\n"
        "    return inner(x)\n"
        "lib.py:2: in inner\n"
        "    return 10 / x\n"
        "E   ZeroDivisionError: division by zero\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/test_a.py::test_raises - ZeroDivisionError: division by zero\n",
        DEMO,
    )
    check("multi-frame traceback parsed", len(deep) == 1 and len(deep[0].frames) == 3)
    check("raised exception typed", deep[0].exc_type == "ZeroDivisionError")
    check("site-packages frame marked external", deep[0].frames[1].in_repo is False)
    check("missing repo file marked external", deep[0].frames[2].in_repo is False)

    classes = parse_failures(
        "=================================== FAILURES ===================================\n"
        "____________________________ TestThing.test_method _____________________________\n"
        "tests/test_b.py:5: in test_method\n"
        "E   assert 1 == 2\n"
        "________________________________ test_param[2] _________________________________\n"
        "tests/test_b.py:9: in test_param\n"
        "E   assert 2 == 1\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/test_b.py::TestThing::test_method - assert 1 == 2\n"
        "FAILED tests/test_b.py::test_param[2] - assert 2 == 1\n",
        DEMO,
    )
    check("two failures separated", len(classes) == 2)
    check("class method node id rebuilt",
          classes[0].test_id == "tests/test_b.py::TestThing::test_method", classes[0].test_id)
    check("parametrised node id kept",
          classes[1].test_id == "tests/test_b.py::test_param[2]", classes[1].test_id)
    check("distinct tests have distinct fingerprints",
          classes[0].fingerprint() != classes[1].fingerprint())

    fixture = parse_failures(
        "==================================== ERRORS ====================================\n"
        "_____________________ ERROR at setup of test_uses_fixture ______________________\n"
        "tests/test_c.py:5: in broken_fixture\n"
        "    raise RuntimeError(\"setup exploded\")\n"
        "E   RuntimeError: setup exploded\n"
        "=========================== short test summary info ============================\n"
        "ERROR tests/test_c.py::test_uses_fixture - RuntimeError: setup exploded\n",
        DEMO,
    )
    check("fixture error captured", len(fixture) == 1 and fixture[0].exc_type == "RuntimeError")
    check("fixture error keeps the test node id",
          fixture[0].test_id == "tests/test_c.py::test_uses_fixture", fixture[0].test_id)

    check("empty report yields nothing", parse_failures("4 passed in 0.02s", DEMO) == ())


def _demo_failure(ws):
    node = "tests/test_cart.py::test_discount_can_lose_free_shipping"
    return pytest_runner.run(ws.root, node, timeout_s=60).failures[0]


def suite_slicer() -> None:
    with Workspace(DEMO) as ws:
        index = SymbolIndex(ws.root)
        check("functions indexed", "make_cart" in index.functions)
        check("methods indexed", "shipping_fee" in index.methods)
        check("imports recorded", index.imports["tests/test_cart.py"].get("Cart") == "cart")

        failure = _demo_failure(ws)
        slices = slices_for_failure(ws.root, failure.frames, ContextConfig(), index)
        names = {s.symbol for s in slices}

        check("traceback frame sliced", "test_discount_can_lose_free_shipping" in names)
        check("callee across modules resolved", "Cart.total" in names)
        check("buggy function reached at depth 2", "Cart.shipping_fee" in names, str(names))
        check("class body not emitted whole", "Cart" not in names, str(names))
        check("no duplicate slices", len(names) == len(slices))

        buggy = next(s for s in slices if s.symbol == "Cart.shipping_fee")
        check("slice is a whole function", buggy.source.lstrip().startswith("def shipping_fee"))
        check("slice ends inside the function",
              buggy.source.rstrip().endswith("return SHIPPING_FEE"), buggy.source[-40:])
        check("slice carries real line numbers",
              ws.resolve("cart.py").read_text().splitlines()[buggy.start_line - 1].strip()
              == "def shipping_fee(self, discount_percent: float = 0.0) -> float:")
        check("reason recorded for the trace", buggy.reason.startswith("called by"))

        shallow = slices_for_failure(ws.root, failure.frames, ContextConfig(callee_depth=1), index)
        check("depth 1 misses the bug (ablation axis)",
              "Cart.shipping_fee" not in {s.symbol for s in shallow})
        check("depth 1 is cheaper", sum(s.token_estimate() for s in shallow)
              < sum(s.token_estimate() for s in slices))

        whole_files = len(ws.resolve("cart.py").read_text()) + len(
            ws.resolve("tests/test_cart.py").read_text())
        sliced = sum(len(s.source) for s in slices)
        check("slicing beats dumping both files", sliced < whole_files,
              f"{sliced} vs {whole_files}")

        # Every Symbol keeps the text it was parsed from, so the index is a
        # snapshot. A run that patches and then looks again must not be shown
        # the code it already replaced.
        index = SymbolIndex(ws.root, state=ws.state_token())
        source = ws.resolve("cart.py")
        original = source.read_text()
        # Edit inside a function body, so it is something a slice can carry at
        # all: a module constant's value never appears in one.
        source.write_text(original.replace("return SHIPPING_FEE", "return SHIPPING_FEE * 2"))
        check("edit is on disk", "SHIPPING_FEE * 2" in source.read_text())
        check("an unchanged workspace does not rebuild",
              index.refresh_if_changed(index.state) is False)
        check("a changed workspace rebuilds", index.refresh_if_changed(ws.state_token()) is True)

        body = "\n".join(
            s.source for s in slices_for_failure(ws.root, failure.frames, ContextConfig(), index))
        check("slices follow the file after a patch", "return SHIPPING_FEE * 2" in body,
              body[-300:])
        check("stale source is gone", "return SHIPPING_FEE\n" not in body)
        source.write_text(original)
        index.refresh_if_changed(ws.state_token())

        external = (Frame(path="/usr/lib/python3.12/json/decoder.py", lineno=1,
                          function="decode", in_repo=False),)
        check("external frames contribute nothing",
              slices_for_failure(ws.root, external, ContextConfig(), index) == [])


def suite_budget() -> None:
    with Workspace(DEMO) as ws:
        slices = slices_for_failure(ws.root, _demo_failure(ws).frames, ContextConfig())
        total = sum(s.token_estimate() for s in slices)

        kept, degraded = fit(slices, 10_000)
        check("everything fits under a generous ceiling", kept == slices and not degraded)

        kept, degraded = fit(slices, total - slices[-1].token_estimate())
        check("least important slice dropped first", len(kept) == len(slices) - 1)
        check("dropping is not degradation", degraded is False)
        check("kept slices stay within budget",
              sum(s.token_estimate() for s in kept) <= total - slices[-1].token_estimate())

        kept, degraded = fit(slices, 5)
        check("starvation flagged", degraded is True)
        check("starved context keeps one signature", len(kept) == 1)
        check("signature has no body",
              "context budget exceeded" in kept[0].source and "return" not in kept[0].source)
        check("signature keeps the def line", kept[0].source.lstrip().startswith("def "))

        check("empty input is not degraded", fit([], 100) == ([], False))
        check("token estimate scales with text",
              estimate_tokens("x" * 400) > estimate_tokens("x" * 40))
        check("signature_only preserves location",
              signature_only(slices[0]).start_line == slices[0].start_line)


NODE = "tests/test_cart.py::test_discount_can_lose_free_shipping"


def _diff(path: str, transform) -> str:
    """Build a unified diff against the real file in demo_repo."""
    return _diff_in(DEMO, path, transform)


def _diff_in(root: Path, path: str, transform) -> str:
    """Build a unified diff against the real file under `root`.

    Hard-coding diff context makes a check depend on the exact bytes of a
    fixture, and a patch that silently fails to apply turns every downstream
    check into a vacuous pass. Generating it means the scenario always happens.
    """
    old = (root / path).read_text()
    new = transform(old)
    assert new != old, f"transform did not change {path}"
    diff = "".join(difflib.unified_diff(
        old.splitlines(True), new.splitlines(True),
        fromfile=f"a/{path}", tofile=f"b/{path}", n=3))
    # A source file with no trailing newline makes difflib run the last removed
    # line and the first added line together on one physical line. git rejects
    # the result, so every scenario built on it passes for the wrong reason --
    # the stagnation scenario ran for months on patches that never applied.
    for line in diff.splitlines()[2:]:
        assert line[:1] in (" ", "-", "+", "@", "\\"), (
            f"malformed diff line for {path}: {line!r}")
    return diff


REAL_FIX = _diff("cart.py", lambda t: t.replace(
    "if self.subtotal() >= FREE_SHIPPING_THRESHOLD:",
    "if self.apply_discount(discount_percent) >= FREE_SHIPPING_THRESHOLD:"))

CHEAT = _diff("tests/test_cart.py", lambda t: t.replace("== 61.16", "== 56.16"))

BREAKS_SYNTAX = _diff("cart.py", lambda t: "def oops(:\n" + t)

# Valid patches that change something harmless, so the workspace state moves
# but the failure does not. Used to exercise fingerprint-based stagnation.
# Anchored on position, not content: any string I pick may not exist in the
# file, and a transform that silently matches nothing breaks the scenario.
# Distant regions so their diff contexts cannot overlap.
NOOP_A = _diff("cart.py", lambda t: t.rstrip("\n") + "\n# noop A\n")
NOOP_B = _diff("cart.py", lambda t: "# noop B\n" + t)


def _envelope(diff: str) -> str:
    """Re-wrap a unified diff the way models trained on apply_patch emit it.

    Built from the diff rather than typed out, so it cannot drift away from
    REAL_FIX and quietly stop exercising the real transform.
    """
    path = changed_paths(diff)[0]
    body = [l for l in diff.splitlines() if not l.startswith(("--- ", "+++ "))]
    return "\n".join(["*** Begin Patch", f"*** Update File: {path}", *body, "*** End Patch"])


def _ctx(ws) -> ToolContext:
    cfg = Config()
    return ToolContext(workspace=ws, config=cfg, target_test=NODE)


def _call(ctx, name, **arguments):
    return dispatch(ToolCall(name, arguments, f"call_{name}"), ctx)


def suite_tools() -> None:
    names = [s["name"] for s in schemas()]
    check("exactly four tools exposed", names == ["read_file", "apply_patch", "run_tests", "finish"],
          str(names))
    check("every schema declares parameters", all("parameters" in s for s in schemas()))

    with Workspace(DEMO, test_globs=Config().guards.test_path_globs) as ws:
        ctx = _ctx(ws)

        # --- dispatch robustness -----------------------------------------
        bad = _call(ctx, "delete_everything")
        check("unknown tool refused", bad.ok is False and "No tool named" in bad.content)
        check("refusal lists real tools", "apply_patch" in bad.content)
        check("call_id echoed back", bad.call_id == "call_delete_everything")

        extra = _call(ctx, "run_tests", node_id=NODE, nonsense=1)
        check("unexpected argument ignored, not fatal", extra.ok is True)
        check("ignored argument recorded", extra.meta.get("ignored_arguments") == ["nonsense"])

        # --- read_file ----------------------------------------------------
        # Locate the target line instead of hard-coding it: the check must
        # test read_file, not the exact formatting of demo_repo/cart.py.
        source_lines = ws.resolve("cart.py").read_text().splitlines()
        defn = next(i + 1 for i, line in enumerate(source_lines)
                    if line.strip().startswith("def shipping_fee"))
        read = _call(ctx, "read_file", path="cart.py", start_line=defn, end_line=defn + 5)
        check("read_file returns the range", read.ok and "def shipping_fee" in read.content,
              read.content[:120])
        check("read_file numbers lines absolutely", f"{defn:>5} |" in read.content,
              read.content[:60])
        check("read_file refuses escapes", _call(ctx, "read_file", path="../../etc/passwd").ok is False)
        check("read_file reports missing files",
              "No such file" in _call(ctx, "read_file", path="nope.py").content)
        check("inverted range refused",
              _call(ctx, "read_file", path="cart.py", start_line=30, end_line=5).ok is False)

        # --- guards -------------------------------------------------------
        cheat = _call(ctx, "apply_patch", diff=CHEAT)
        check("editing a test file is refused", cheat.ok is False)
        check("refusal explains what to do instead", "Fix the source" in cheat.content)
        check("test file is untouched on disk",
              "61.16" in ws.resolve("tests/test_cart.py").read_text())

        escape = _call(ctx, "apply_patch", diff=REAL_FIX.replace("a/cart.py", "a/../../evil.py")
                                                        .replace("b/cart.py", "b/../../evil.py"))
        check("patch outside the workspace refused", escape.ok is False)

        huge = "--- a/cart.py\n+++ b/cart.py\n@@ -1,1 +1,1 @@\n" + "+x\n" * 200
        check("oversized patch refused", _call(ctx, "apply_patch", diff=huge).ok is False)

        check("empty diff refused", _call(ctx, "apply_patch", diff="   ").ok is False)
        junk = _call(ctx, "apply_patch", diff="please change the shipping logic")
        check("prose instead of a diff refused", junk.ok is False)
        check("refusal explains the required format", "unified diff" in junk.content)

        stale = _call(ctx, "apply_patch", diff=REAL_FIX.replace(
            "if self.subtotal() >= FREE_SHIPPING_THRESHOLD:", "if self.nonexistent_line():"))
        check("diff with wrong context refused", stale.ok is False)
        check("rejection tells the model to re-read", "Re-read the file" in stale.content)
        check("nothing changed after a rejected patch", ws.diff().strip() == "")

        # --- the happy path -----------------------------------------------
        before = _call(ctx, "run_tests")
        check("run_tests reports the failure", before.meta["passed"] is False)
        check("failing test is still a successful call", before.ok is True)
        check("summary names the exception", "AssertionError" in before.content)
        check("fingerprints exposed for the policy layer", len(before.meta["fingerprints"]) == 1)
        check("omitted node_id means the target test", before.meta["node_id"] == NODE)

        applied = _call(ctx, "apply_patch", diff=REAL_FIX)
        check("valid patch applies", applied.ok is True, applied.content)
        check("patch recorded which files changed", applied.meta["paths"] == ["cart.py"])
        check("diff reflects the edit", "apply_discount(discount_percent)" in ws.diff())

        after = _call(ctx, "run_tests")
        check("target test now passes", after.meta["passed"] is True)
        check("no regressions in the suite",
              pytest_runner.run(ws.root, None, timeout_s=60).passed is True)

        fenced = _call(ctx, "apply_patch", diff="```diff\n" + REAL_FIX + "```")
        check("markdown-fenced diff is handled", "did not apply" in fenced.content
              or fenced.ok is True)

        done = _call(ctx, "finish", summary="shipping_fee ignored the discount")
        check("finish records the claim", ctx.finish_summary.startswith("shipping_fee"))
        check("finish does not end the run by itself", "verify" in done.content)

    # --- diff formats the model actually emits ----------------------------
    # A live run lost a correct fix here: the model wrapped a valid hunk in the
    # *** Begin Patch envelope -- the format its own tool name primes -- and the
    # harness rejected it with advice about context lines that were never wrong.
    envelope = _envelope(REAL_FIX)
    unwrapped = unwrap_envelope(envelope)
    check("envelope rewritten as a unified diff",
          unwrapped.startswith("--- a/cart.py\n+++ b/cart.py\n@@"), unwrapped[:90])
    check("envelope markers removed", "*** " not in unwrapped)
    check("envelope keeps every changed line",
          changed_line_count(unwrapped) == changed_line_count(REAL_FIX))
    check("a plain diff is left alone", unwrap_envelope(REAL_FIX) == REAL_FIX)

    with Workspace(DEMO, test_globs=Config().guards.test_path_globs) as ws:
        ctx = _ctx(ws)
        applied = _call(ctx, "apply_patch", diff=envelope)
        check("enveloped patch applies", applied.ok is True, applied.content[:160])
        check("enveloped patch edits the real file",
              "apply_discount(discount_percent)" in ws.resolve("cart.py").read_text())
        check("enveloped patch fixes the test",
              _call(ctx, "run_tests").meta["passed"] is True)

        # Both artefacts came off one trace: a closing envelope marker with no
        # opening one, and a bare `@@`. The model sent this exact edit seven
        # times and the harness refused every one of them.
        bare = ("--- a/cart.py\n+++ b/cart.py\n@@\n"
                "-TAX_RATE = 0.20\n+TAX_RATE = 0.25\n*** End Patch\n")
        landed = _call(ctx, "apply_patch", diff=bare)
        check("bare @@ hunk with a stray End Patch applies", landed.ok is True,
              landed.content[:200])
        check("bare hunk edited the right line",
              "TAX_RATE = 0.25" in ws.resolve("cart.py").read_text())
        check("bare hunk changed nothing else",
              len(ws.changed_files()) == 1, str(ws.changed_files()))

        numbered = renumber_bare_hunks(
            "--- a/cart.py\n+++ b/cart.py\n@@\n-TAX_RATE = 0.25\n+TAX_RATE = 0.20\n",
            ws.root)
        check("renumbering finds the real line",
              "@@ -3,1 +3,1 @@" in numbered, numbered)
        check("an unplaceable hunk is left for git to reject",
              "@@\n" in renumber_bare_hunks(
                  "--- a/cart.py\n+++ b/cart.py\n@@\n-NOT_IN_THE_FILE = 1\n+X = 2\n", ws.root))
        check("a numbered hunk is not rewritten",
              renumber_bare_hunks(REAL_FIX, ws.root).strip() == REAL_FIX.strip())

        no_hunk = "\n".join(l for l in REAL_FIX.splitlines() if not l.startswith("@@"))
        bad = _call(ctx, "apply_patch", diff=no_hunk)
        check("malformed diff refused", bad.ok is False)
        check("format error is named as a format error",
              "format problem" in bad.content, bad.content[:200])
        check("format error does not blame the file's context lines",
              "Re-read the file" not in bad.content, bad.content[:200])

    check("diff helper counts changed lines", changed_line_count(REAL_FIX) == 2)
    check("diff helper extracts paths", changed_paths(REAL_FIX) == ["cart.py"])


def _turn(text, name=None, arguments=None, prompt=900, completion=120):
    message = {"role": "assistant", "content": text}
    if name:
        message["tool_calls"] = [
            {"id": "c", "function": {"name": name, "arguments": json.dumps(arguments or {})}}
        ]
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


_last_replay: ReplayLLM | None = None


def _run(script, cfg=None, target=NODE, repo=None):
    global _last_replay
    with tempfile.TemporaryDirectory() as trace_dir:
        cfg = cfg or Config()
        cfg = Config(model=cfg.model, context=cfg.context, loop=cfg.loop,
                     guards=cfg.guards, trace_dir=trace_dir)
        _last_replay = ReplayLLM(script)
        report = Agent(cfg, _last_replay).run(repo or DEMO, target, task_id="check")
        # A scenario that runs out of scripted turns reports a transport error,
        # which reads as a policy failure but is a fixture failure. Name it.
        if report.outcome.value == "error" and report.iterations > 0 and not _last_replay._queue:
            check("scenario script outlasts the iteration cap", False,
                  f"replay exhausted after {report.iterations} turns; "
                  f"script had {len(script)}")
        traces = sorted(pathlib.Path(trace_dir).glob("*.jsonl"))
        events = [json.loads(line) for line in traces[0].read_text().splitlines()] if traces else []
        return report, events


def suite_loop() -> None:
    from green_agent.types import Outcome

    # --- happy path ------------------------------------------------------
    report, events = _run([
        _turn("HYPOTHESIS: shipping_fee ignores the discount", "apply_patch", {"diff": REAL_FIX}),
        _turn("HYPOTHESIS: verify", "run_tests", {}),
        _turn("HYPOTHESIS: done", "finish", {"summary": "threshold checked pre-discount"}),
    ])
    check("run reports FIXED", report.outcome is Outcome.FIXED, report.outcome.value)
    check("three iterations used", report.iterations == 3, str(report.iterations))
    check("tokens accumulated", report.total_tokens == 3060, str(report.total_tokens))
    check("diff contains the real fix", "apply_discount(discount_percent)" in report.final_diff)
    check("diff touches only source", "tests/" not in report.final_diff)
    check("hypotheses captured per step", all(s.hypothesis for s in report.steps))

    kinds = [e["event"] for e in events]
    check("trace opens and closes", kinds[0] == "run_start" and kinds[-1] == "run_end")
    check("baseline recorded before any edit", "baseline" in kinds)
    check("context slices traced", any(e["event"] == "context_built" for e in events))
    check("verification traced", any(e["event"] == "verify" for e in events))
    check("trace is replayable",
          len(ReplayLLM([e["response"] for e in events if e["event"] == "model_call"])._queue) == 3)

    # --- the model must be shown its own edits ---------------------------
    # The index was built once per run and cached file text, so every prompt
    # after the first patch showed pre-patch source. A live run applied the
    # correct fix on turn 1, read back its own unfixed code, decided the edit
    # had not persisted, and re-sent the same patch until the budget ran out.
    report, events = _run([
        _turn("HYPOTHESIS: shipping_fee ignores the discount", "apply_patch", {"diff": REAL_FIX}),
        _turn("HYPOTHESIS: verify", "run_tests", {}),
        _turn("HYPOTHESIS: done", "finish", {"summary": "threshold checked pre-discount"}),
    ])
    prompts = [c["messages"][-1]["content"] for c in _last_replay.calls]
    check("the first prompt shows the unfixed line",
          "if self.subtotal() >= FREE_SHIPPING_THRESHOLD:" in prompts[0], prompts[0][:200])
    check("the prompt after a patch shows the patched line",
          "self.apply_discount(discount_percent) >= FREE_SHIPPING_THRESHOLD" in prompts[1],
          prompts[1][:600])
    check("the replaced line is gone from the prompt",
          "if self.subtotal() >= FREE_SHIPPING_THRESHOLD:" not in prompts[1])
    check("the rebuild is traced", any(e["event"] == "index_refreshed" for e in events))

    # --- edits that are never verified ------------------------------------
    # A live run applied five different patches and never once ran the tests.
    # Neither existing detector can see that: the results never change because
    # they are never taken, and no two calls are identical.
    report, events = _run([
        _turn("HYPOTHESIS: tweak one", "apply_patch", {"diff": NOOP_A}),
        _turn("HYPOTHESIS: tweak two", "apply_patch", {"diff": NOOP_B}),
        _turn("HYPOTHESIS: the real fix", "apply_patch", {"diff": REAL_FIX}),
        _turn("HYPOTHESIS: check", "run_tests", {}),
        _turn("HYPOTHESIS: done", "finish", {"summary": "threshold checked pre-discount"}),
    ], cfg=Config(loop=LoopConfig(max_iterations=6)))
    prompts = [c["messages"][-1]["content"] for c in _last_replay.calls]
    check("the first patch really applied", report.steps[0].result.ok is True)
    check("an unverified edit is called out next turn",
          "Call run_tests now" in prompts[1], prompts[1][-220:])
    check("it keeps being called out while unverified",
          "Call run_tests now" in prompts[2], prompts[2][-220:])
    check("unverified edits are traced",
          any(e["event"] == "unverified_edit" for e in events))
    check("running the tests clears it",
          "Call run_tests now" not in prompts[4], prompts[4][-220:])
    check("the directive did not prevent the fix", report.outcome is Outcome.FIXED,
          report.outcome.value)

    # --- the model claims success without fixing anything -----------------
    report, _ = _run([
        _turn("HYPOTHESIS: probably fine", "finish", {"summary": "looks correct to me"}),
        _turn("HYPOTHESIS: still not fixed", "run_tests", {}),
        _turn("HYPOTHESIS: giving up", "finish", {"summary": "again"}),
    ], cfg=Config(loop=LoopConfig(max_iterations=3)))
    check("unverified finish is refused",
          report.outcome is Outcome.BUDGET_EXHAUSTED, report.outcome.value)
    check("finish claims were rejected, not accepted",
          all(s.result.ok for s in report.steps if s.call.name == "finish"))
    check("no changes were made", report.final_diff.strip() == "")

    # --- the model tries to edit the test --------------------------------
    report, _ = _run([
        _turn("HYPOTHESIS: the test expects the wrong number", "apply_patch", {"diff": CHEAT}),
        _turn("HYPOTHESIS: try again", "apply_patch", {"diff": CHEAT}),
        _turn("HYPOTHESIS: verify", "run_tests", {}),
    ], cfg=Config(loop=LoopConfig(max_iterations=3)))
    check("cheating never reaches FIXED", report.outcome is not Outcome.FIXED)
    check("test file never modified", "test_cart.py" not in report.final_diff, report.final_diff[:80])
    check("rejections recorded as failed steps",
          all(s.result.ok is False for s in report.steps if s.call.name == "apply_patch"))

    # --- patching without progress: same failure fingerprint --------------
    # The script must outlast max_iterations: if replay runs dry the loop
    # reports a transport error, which looks like a policy failure but is a
    # fixture failure. Either no-progress detector may fire first, and both
    # must end the run the same way.
    report, events = _run([
        _turn("HYPOTHESIS: check first", "run_tests", {}),
        _turn("HYPOTHESIS: tweak one", "apply_patch", {"diff": NOOP_A}),
        _turn("HYPOTHESIS: check", "run_tests", {}),
        _turn("HYPOTHESIS: tweak two", "apply_patch", {"diff": NOOP_B}),
        _turn("HYPOTHESIS: check again", "run_tests", {}),
    ] + [_turn("HYPOTHESIS: once more", "run_tests", {}) for _ in range(8)],
        cfg=Config(loop=LoopConfig(max_iterations=8, stagnation_threshold=1)))
    check("stagnation stops the run", report.outcome is Outcome.STAGNATED, report.outcome.value)
    check("stagnation stops early", report.iterations < 8, str(report.iterations))
    check("stagnation traced", any(e["event"] == "stagnation" for e in events))
    check("re-plan was offered before aborting",
          any(e["event"] == "stagnation" for e in events) and report.iterations >= 4,
          str(report.iterations))

    # --- a patch that breaks the file ------------------------------------
    report, events = _run([
        _turn("HYPOTHESIS: rewrite the module", "apply_patch", {"diff": BREAKS_SYNTAX}),
        _turn("HYPOTHESIS: check it", "run_tests", {}),
        _turn("HYPOTHESIS: now fix properly", "apply_patch", {"diff": REAL_FIX}),
        _turn("HYPOTHESIS: verify", "run_tests", {}),
        _turn("HYPOTHESIS: done", "finish", {"summary": "fixed"}),
    ], cfg=Config(loop=LoopConfig(max_iterations=6)))
    check("syntax-breaking patch actually applied", report.steps[0].result.ok is True,
          report.steps[0].result.content[:120])
    check("collection error detected",
          any(e.get("meta", {}).get("collect_error") for e in events))
    check("collection error is recovered", any(e["event"] == "recovered" for e in events))
    check("agent still finishes after recovery", report.outcome is Outcome.FIXED,
          report.outcome.value)
    check("broken code did not survive", "def oops" not in report.final_diff)

    # --- reading forever without editing (the first live-run failure) ----
    report, events = _run(
        [_turn("HYPOTHESIS: let me look again", "read_file", {"path": "cart.py"})
         for _ in range(6)],
        cfg=Config(loop=LoopConfig(max_iterations=6)),
    )
    blocked = [e for e in events if e["event"] == "repeat_blocked"]
    check("identical read_file is blocked", len(blocked) >= 3, str(len(blocked)))
    check("a read loop ends the run", report.outcome is Outcome.STAGNATED, report.outcome.value)
    check("first read still went through", report.steps[0].result.ok is True)
    check("repeat is rejected, not dispatched", report.steps[1].result.ok is False)
    check("rejection tells the model to move on",
          "different action" in report.steps[1].result.content)
    check("read-only streak is flagged", any(e["event"] == "read_streak" for e in events))

    # --- a rejection is new information ----------------------------------
    # apply_patch tells a model whose diff was refused to re-read the file, and
    # the repeat detector used to block exactly that re-read, because nothing on
    # disk had changed. A live run died of stagnation here having made no edit.
    report, events = _run([
        _turn("HYPOTHESIS: look at it", "read_file", {"path": "cart.py"}),
        _turn("HYPOTHESIS: look again", "read_file", {"path": "cart.py"}),
        _turn("HYPOTHESIS: try a fix", "apply_patch", {"diff": "change the shipping logic"}),
        _turn("HYPOTHESIS: re-read as instructed", "read_file", {"path": "cart.py"}),
        _turn("HYPOTHESIS: now patch properly", "apply_patch", {"diff": REAL_FIX}),
        _turn("HYPOTHESIS: verify", "run_tests", {}),
        _turn("HYPOTHESIS: done", "finish", {"summary": "threshold checked pre-discount"}),
    ], cfg=Config(loop=LoopConfig(max_iterations=8)))
    check("the second identical read is still blocked", report.steps[1].result.ok is False)
    check("the patch was actually rejected", report.steps[2].result.ok is False,
          report.steps[2].result.content[:120])
    check("a rejection unblocks the re-read it asked for",
          report.steps[3].result.ok is True, report.steps[3].result.content[:120])
    check("re-read content is the file, not a refusal",
          "cart.py" in report.steps[3].result.content)
    check("the run recovers and fixes the bug", report.outcome is Outcome.FIXED,
          report.outcome.value)
    check("only one call was blocked",
          len([e for e in events if e["event"] == "repeat_blocked"]) == 1,
          str([e["event"] for e in events].count("repeat_blocked")))

    varied = _run(
        [_turn("HYPOTHESIS: read one", "read_file", {"path": "cart.py", "start_line": 1}),
         _turn("HYPOTHESIS: read two", "read_file", {"path": "cart.py", "start_line": 20}),
         _turn("HYPOTHESIS: now patch", "apply_patch", {"diff": REAL_FIX}),
         _turn("HYPOTHESIS: verify", "run_tests", {}),
         _turn("HYPOTHESIS: done", "finish", {"summary": "fixed"})],
        cfg=Config(loop=LoopConfig(max_iterations=6)),
    )[0]
    check("different ranges are not treated as repeats", varied.outcome.value == "fixed",
          varied.outcome.value)

    repeated_runs = _run(
        [_turn("HYPOTHESIS: patch", "apply_patch", {"diff": REAL_FIX}),
         _turn("HYPOTHESIS: check", "run_tests", {}),
         _turn("HYPOTHESIS: check again", "run_tests", {}),
         _turn("HYPOTHESIS: done", "finish", {"summary": "fixed"})],
        cfg=Config(loop=LoopConfig(max_iterations=5)),
    )[0]
    check("re-running tests after a patch is allowed",
          repeated_runs.outcome.value == "fixed", repeated_runs.outcome.value)

    # --- green test, but the model does not claim it ---------------------
    report, events = _run([
        _turn("HYPOTHESIS: patch", "apply_patch", {"diff": REAL_FIX}),
        _turn("HYPOTHESIS: check", "run_tests", {}),
        _turn("HYPOTHESIS: I think we are done", "finish", {"summary": "fixed"}),
    ])
    prompts = [c["messages"][-1]["content"] for c in _last_replay.calls]
    check("passing test triggers a finish directive",
          "Call finish now" in prompts[-1], prompts[-1][-160:])
    check("directive did not delay the fix", report.outcome is Outcome.FIXED)

    # --- a directive survives a turn that did nothing ---------------------
    # A live run went green on turn 7, returned an empty completion on turn 8,
    # and was never told again that it was done: the directive was recomputed
    # as None by the no-action branch. It then spent the whole budget.
    report, events = _run([
        _turn("HYPOTHESIS: patch", "apply_patch", {"diff": REAL_FIX}),
        _turn("HYPOTHESIS: check", "run_tests", {}),
        _turn("HYPOTHESIS: nothing to say"),
        _turn("HYPOTHESIS: still nothing"),
        _turn("HYPOTHESIS: done", "finish", {"summary": "threshold checked pre-discount"}),
    ], cfg=Config(loop=LoopConfig(max_iterations=6)))
    prompts = [c["messages"][-1]["content"] for c in _last_replay.calls]
    check("finish directive is issued once the test passes",
          "Call finish now" in prompts[2], prompts[2][-200:])
    check("finish directive survives one dead turn",
          "Call finish now" in prompts[3], prompts[3][-200:])
    check("finish directive survives two dead turns",
          "Call finish now" in prompts[4], prompts[4][-200:])
    check("the no-action reason is added, not substituted",
          "exactly one tool call" in prompts[3])
    check("the run still ends fixed", report.outcome is Outcome.FIXED, report.outcome.value)
    check("a green run stops telling the model to read files",
          "use read_file" not in prompts[4], prompts[4][:300])

    # --- degenerate model behaviour --------------------------------------
    report, _ = _run([_turn("HYPOTHESIS: thinking out loud, no action") for _ in range(2)],
                     cfg=Config(loop=LoopConfig(max_iterations=2)))
    check("turns without a tool call are survived", report.outcome is Outcome.BUDGET_EXHAUSTED,
          report.outcome.value)
    check("no-action turns still cost budget", report.iterations == 2)

    report, _ = _run([_turn("HYPOTHESIS: x", "apply_patch", {"diff": REAL_FIX})],
                     cfg=Config(loop=LoopConfig(max_iterations=1)))
    check("budget limit is enforced", report.outcome is Outcome.BUDGET_EXHAUSTED,
          report.outcome.value)

    # --- task hygiene ----------------------------------------------------
    report, events = _run([], target="tests/test_cart.py::test_subtotal")
    check("already-green task refused", report.iterations == 0)
    check("refusal is not a success", report.outcome is not Outcome.FIXED)
    check("invalid task traced",
          any(e.get("detail") == "target test already passes" for e in events))


def suite_benchmark() -> None:
    """The benchmark itself, offline: no network, no API key, no live model."""
    from green_agent.types import Outcome, RunReport

    from benchmark.ablations import ABLATIONS, config_for
    from benchmark.catalog import BUGS, load_tasks
    from benchmark.prepare import checkout, verify
    from benchmark.scoring import (
        INVALID,
        SOLVED,
        UNSOLVED,
        Result,
        aggregate,
        score_report,
        touched_test_files,
    )

    tasks = load_tasks()

    # --- the catalogue ----------------------------------------------------
    check("six tasks on disk", len(tasks) == 6, str(len(tasks)))
    check("every spec was written out", {t.task_id for t in tasks} == {b.task_id for b in BUGS})
    check("two tasks per project",
          sorted(Counter(t.project for t in tasks).values()) == [2, 2, 2],
          str(Counter(t.project for t in tasks)))
    check("bug shapes are all different",
          len({t.shape for t in tasks}) == 6, str(sorted({t.shape for t in tasks})))
    check("at least two tracebacks point away from the bug",
          sum(1 for t in tasks if t.traceback == "away from the bug") >= 2)
    check("every task names its project directory",
          all(t.project_dir.is_dir() for t in tasks))
    check("every task has a patch file", all(t.patch_path.is_file() for t in tasks))
    check("every task names a test file as its target",
          all(t.target_test.startswith("tests/") and "::" in t.target_test for t in tasks))

    # A spec whose anchor text drifted still writes a patch that no longer
    # applies, so regenerating must reproduce exactly what is committed.
    for spec in BUGS:
        stored = (TASKS / spec.task_id / "bug.patch").read_text(encoding="utf-8")
        check(f"patch matches the source: {spec.task_id}", spec.diff() == stored)

    # --- every bug is observable -----------------------------------------
    # The expensive one, and the one that matters: green without the patch,
    # exactly the named test failing with it. Without this a benchmark reports
    # numbers about tasks that were never valid.
    for task in tasks:
        outcome = verify(task)
        check(f"observable: {task.task_id}", outcome.ok, "; ".join(outcome.problems))

    # --- ablations are config changes and nothing else --------------------
    check("five ablations", sorted(ABLATIONS) ==
          ["depth-1", "full", "no-repeat-detection", "no-slicing", "no-test-guard"])
    base = config_for("full")
    check("full is the shipped default", base == Config())
    check("no-slicing only moves whole_file_context",
          config_for("no-slicing").context.whole_file_context is True
          and config_for("no-slicing").loop == base.loop)
    check("depth-1 only moves callee_depth",
          config_for("depth-1").context.callee_depth == 1
          and config_for("depth-1").guards == base.guards)
    check("no-test-guard only drops the guard",
          config_for("no-test-guard").guards.forbid_test_file_edits is False
          and config_for("no-test-guard").context == base.context)
    check("no-repeat-detection only disables the detector",
          config_for("no-repeat-detection").loop.repeat_detection is False
          and config_for("no-repeat-detection").context == base.context)
    try:
        Config.from_env_and_overrides(**{"context.no_such_knob": 1})
        check("a typo'd override is refused", False)
    except KeyError:
        check("a typo'd override is refused", True)

    # --- scoring ----------------------------------------------------------
    def _report(outcome, diff="", task="t", iterations=3, tokens=100):
        return RunReport(task, outcome, iterations, tokens, 1.0, diff)

    solved = score_report(_report(Outcome.FIXED, REAL_FIX), "full")
    check("a verified fix scores solved", solved.score == SOLVED)
    check("solved run records what it changed", solved.changed_files == ["cart.py"])

    # The rule the no-test-guard ablation exists to keep honest: green tests
    # reached by editing the test are not a solve, whatever the harness said.
    cheated = score_report(_report(Outcome.FIXED, CHEAT), "no-test-guard")
    check("a test-file edit scores INVALID, not solved", cheated.score == INVALID,
          cheated.score)
    check("INVALID names the file", cheated.test_files_touched == ["tests/test_cart.py"])
    check("INVALID explains itself", "modified test file" in cheated.detail)
    check("the harness outcome is still recorded", cheated.outcome == "fixed")
    check("a fix plus a test edit is still INVALID",
          score_report(_report(Outcome.FIXED, REAL_FIX + CHEAT), "full").score == INVALID)
    check("touched_test_files ignores source files", touched_test_files(REAL_FIX) == [])
    check("conftest counts as a test file",
          touched_test_files("--- a/conftest.py\n+++ b/conftest.py\n") == ["conftest.py"])

    check("budget exhaustion is unsolved",
          score_report(_report(Outcome.BUDGET_EXHAUSTED), "full").score == UNSOLVED)
    check("stagnation is unsolved",
          score_report(_report(Outcome.STAGNATED), "full").score == UNSOLVED)
    check("a harness error is not an unsolved task",
          score_report(_report(Outcome.ERROR), "full").score == "error")

    # --- report arithmetic ------------------------------------------------
    synthetic = [
        Result("a", "full", 1, SOLVED, "fixed", iterations=4, tokens=1000, wall_time_s=1.0),
        Result("a", "full", 2, SOLVED, "fixed", iterations=6, tokens=3000, wall_time_s=1.0),
        Result("b", "full", 1, UNSOLVED, "budget", iterations=12, tokens=6000, wall_time_s=1.0),
        Result("c", "full", 1, INVALID, "fixed", iterations=2, tokens=2000, wall_time_s=1.0),
    ]
    totals = aggregate(synthetic)
    check("solve rate counts only solved", totals.solved == 2 and totals.runs == 4)
    check("solve rate is a fraction of runs", abs(totals.solve_rate - 0.5) < 1e-9)
    check("invalid is counted separately", totals.invalid == 1 and totals.unsolved == 1)
    check("mean iterations over all runs", totals.iterations == 6.0, str(totals.iterations))
    check("mean tokens over all runs", totals.tokens == 3000.0, str(totals.tokens))
    check("tokens per solve divides by solves, not runs",
          totals.tokens_per_solved == 6000.0, str(totals.tokens_per_solved))
    check("an all-invalid file solves nothing",
          aggregate([synthetic[3]]).solved == 0)
    check("nothing solved does not divide by zero",
          aggregate([synthetic[2]]).tokens_per_solved == float("inf"))
    check("an empty result set is empty, not an error", aggregate([]).runs == 0)

    # --- one offline agent run over a real task ---------------------------
    # ReplayLLM, so the whole path -- checkout, bug applied at HEAD, agent loop,
    # independent verification -- runs in the gate with no key and no network.
    task = next(t for t in tasks if t.task_id == "datalib-top-n-default")
    with tempfile.TemporaryDirectory() as trace_dir, checkout(task) as repo:
        # Diffed against the checkout, not the green project: the fix has to go
        # 1 -> 3, and generating it here asserts the bug is actually present.
        fix = _diff_in(repo, "aggregate.py",
                       lambda t: t.replace("DEFAULT_TOP_N = 1", "DEFAULT_TOP_N = 3"))
        script = [
            _turn("HYPOTHESIS: the default constant is wrong", "apply_patch", {"diff": fix}),
            _turn("HYPOTHESIS: verify", "run_tests", {}),
            _turn("HYPOTHESIS: done", "finish", {"summary": "DEFAULT_TOP_N was 1"}),
        ] + [_turn("HYPOTHESIS: spare", "run_tests", {}) for _ in range(6)]
        report = Agent(Config(trace_dir=trace_dir), ReplayLLM(script)).run(
            repo, task.target_test, task.task_id)

    check("offline benchmark run reaches FIXED", report.outcome is Outcome.FIXED,
          report.outcome.value)
    check("offline run took the scripted three turns", report.iterations == 3,
          str(report.iterations))
    check("offline run scores solved", score_report(report, "full").score == SOLVED)
    check("offline run touched no test file", touched_test_files(report.final_diff) == [])
    check("offline run recorded its trace", Path(report.trace_path).name.endswith(".jsonl"))


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
    "traceback_parse": suite_traceback_parse,
    "slicer": suite_slicer,
    "budget": suite_budget,
    "tools": suite_tools,
    "loop": suite_loop,
    "benchmark": suite_benchmark,
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
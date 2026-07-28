import pytest

from registry import Registry, ToolError


def add(a, b):
    return a + b


@pytest.fixture
def registry() -> Registry:
    reg = Registry()
    reg.register("add", add, required=("a", "b"), description="add two numbers")
    reg.register("ping", lambda: "pong")
    return reg


def test_call_dispatches_to_the_handler(registry):
    assert registry.call("add", {"a": 1, "b": 2}) == 3


def test_names_are_sorted(registry):
    assert registry.names() == ["add", "ping"]


def test_schema_exposes_the_required_arguments(registry):
    schema = next(s for s in registry.schemas() if s["name"] == "add")
    assert schema["required"] == ["a", "b"]
    assert schema["description"] == "add two numbers"


def test_unknown_tool_lists_what_exists(registry):
    with pytest.raises(ToolError) as err:
        registry.call("teleport", {})
    assert "add" in str(err.value)


def test_missing_argument_is_named(registry):
    with pytest.raises(ToolError) as err:
        registry.call("add", {"a": 1})
    assert "b" in str(err.value)


def test_registering_twice_is_refused(registry):
    with pytest.raises(ToolError):
        registry.register("add", add)


def test_a_tool_without_required_arguments_needs_none(registry):
    assert registry.call("ping", {}) == "pong"

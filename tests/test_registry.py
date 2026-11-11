from dataclasses import FrozenInstanceError

import pytest

from runtime.models import JSONValue, ToolPolicy
from runtime.registry import RegisteredTool, ToolRegistry


async def echo(value: JSONValue) -> JSONValue:
    return value


def test_registration_preserves_policy_and_function():
    registry = ToolRegistry()
    policy = ToolPolicy("echo", idempotent=True)

    registered = registry.register(policy, echo)

    assert isinstance(registered, RegisteredTool)
    assert registered.policy is policy
    assert registered.function is echo
    assert registry.get("echo") is registered


@pytest.mark.parametrize("field", ["policy", "function"])
def test_registered_tool_is_frozen(field):
    registered = ToolRegistry().register(ToolPolicy("echo", True), echo)

    with pytest.raises(FrozenInstanceError):
        setattr(registered, field, None)


def test_unknown_tool_raises_key_error():
    with pytest.raises(KeyError) as error:
        ToolRegistry().get("missing")

    assert error.value.args == ("missing",)


def test_duplicate_registration_does_not_replace_original():
    registry = ToolRegistry()
    original = registry.register(ToolPolicy("echo", True), echo)

    async def replacement(value):
        return value

    with pytest.raises(ValueError, match="echo.*already registered"):
        registry.register(ToolPolicy("echo", False), replacement)

    assert registry.get("echo") is original
    assert registry.snapshot() == {"echo": original}


def test_snapshot_has_independent_membership_in_both_directions():
    registry = ToolRegistry()
    original = registry.register(ToolPolicy("echo", True), echo)
    snapshot = registry.snapshot()
    later = registry.register(ToolPolicy("later", False), echo)

    assert snapshot == {"echo": original}
    snapshot.pop("echo")
    snapshot["local"] = later
    assert registry.snapshot() == {"echo": original, "later": later}
    assert registry.snapshot() is not registry.snapshot()


def test_empty_snapshot_is_not_changed_by_registration():
    registry = ToolRegistry()
    snapshot = registry.snapshot()

    registry.register(ToolPolicy("echo", True), echo)

    assert snapshot == {}


def test_argument_validation_binds_without_execution_or_mutation():
    calls = []

    async def function(value, default=1, *, required, optional=None):
        calls.append(value)
        return value

    registered = ToolRegistry().register(ToolPolicy("bind", True), function)
    args = {"value": {"nested": [1, None]}, "required": True}

    assert registered.validate_arguments(args) is None
    assert args == {"value": {"nested": [1, None]}, "required": True}
    with pytest.raises(TypeError):
        registered.validate_arguments({})
    assert calls == []


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"value": 1},
        {"required": True},
        {"value": 1, "required": True, "unexpected": 2},
    ],
)
def test_argument_validation_rejects_missing_and_unexpected_arguments(args):
    async def function(value, *, required):
        return value

    registered = ToolRegistry().register(ToolPolicy("bind", True), function)

    with pytest.raises(TypeError):
        registered.validate_arguments(args)


def test_argument_validation_does_not_enforce_annotations_or_json_encoding():
    async def function(value: int) -> int:
        raise AssertionError("Validation must not execute the tool")

    registered = ToolRegistry().register(ToolPolicy("bind", True), function)

    assert registered.validate_arguments({"value": "not an integer"}) is None
    assert registered.validate_arguments({"value": float("nan")}) is None


def test_argument_validation_supports_variadic_parameters():
    async def function(value, *args, **kwargs):
        return value

    registered = ToolRegistry().register(ToolPolicy("bind", True), function)

    assert registered.validate_arguments({"value": 1, "extra": [2]}) is None
    with pytest.raises(TypeError):
        registered.validate_arguments({"extra": 2})


def test_optional_positional_only_parameter_can_use_its_default():
    async def function(value=1, /, *, label):
        return value

    registered = ToolRegistry().register(ToolPolicy("bind", True), function)

    assert registered.validate_arguments({"label": "default"}) is None
    with pytest.raises(TypeError):
        registered.validate_arguments({"value": 2, "label": "override"})


def test_required_positional_only_parameter_is_rejected():
    async def function(value, /, **kwargs):
        return value

    registry = ToolRegistry()
    with pytest.raises(TypeError, match="value.*required and positional-only"):
        registry.register(ToolPolicy("bind", True), function)
    assert registry.snapshot() == {}


@pytest.mark.parametrize(
    "kind", ["sync", "awaitable_factory", "generator", "async_generator", "noncallable"]
)
def test_invalid_functions_are_rejected_without_registration_or_execution(kind):
    def sync(value):
        raise AssertionError("Registration must not execute the tool")

    def awaitable_factory(value):
        return echo(value)

    def generator(value):
        yield value

    async def async_generator(value):
        yield value

    functions = {
        "sync": sync,
        "awaitable_factory": awaitable_factory,
        "generator": generator,
        "async_generator": async_generator,
        "noncallable": None,
    }
    registry = ToolRegistry()

    with pytest.raises(TypeError, match="async function"):
        registry.register(ToolPolicy("invalid", True), functions[kind])
    assert registry.snapshot() == {}


async def test_decorator_defaults_preserve_original_callable_and_signature():
    registry = ToolRegistry()

    decorated = registry.tool(idempotent=True)(echo)

    assert decorated is echo
    assert await decorated(value="direct call") == "direct call"
    registered = registry.get("echo")
    assert registered.function is echo
    assert registered.policy == ToolPolicy("echo", True)
    assert registered.validate_arguments({"value": None}) is None
    with pytest.raises(TypeError):
        registered.validate_arguments({"unexpected": None})


async def test_decorator_accepts_explicit_policy_settings():
    registry = ToolRegistry()

    @registry.tool(
        name="custom",
        idempotent=False,
        timeout_ms=321,
        max_retries=0,
        base_backoff_ms=0,
    )
    async def function(value):
        return value

    registered = registry.get("custom")
    assert registered.function is function
    assert registered.policy == ToolPolicy("custom", False, 321, 0, 0)
    assert await function(value=42) == 42
    with pytest.raises(KeyError):
        registry.get("function")


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": ""},
        {"name": "  "},
        {"name": 123},
        {"idempotent": 1},
        {"idempotent": None},
        {"timeout_ms": 0},
        {"timeout_ms": True},
        {"max_retries": -1},
        {"max_retries": 1.5},
        {"base_backoff_ms": -1},
        {"base_backoff_ms": "50"},
    ],
)
def test_decorator_uses_tool_policy_validation(overrides):
    registry = ToolRegistry()
    options = {"idempotent": True, **overrides}

    with pytest.raises(ValueError):
        registry.tool(**options)(echo)
    assert registry.snapshot() == {}


def test_decorator_requires_explicit_idempotency():
    with pytest.raises(TypeError):
        ToolRegistry().tool()


def test_decorator_does_not_replace_duplicate_name():
    registry = ToolRegistry()
    original = registry.register(ToolPolicy("echo", True), echo)

    with pytest.raises(ValueError, match="already registered"):
        registry.tool(idempotent=False)(echo)
    assert registry.get("echo") is original


def test_decorator_rejects_synchronous_function():
    registry = ToolRegistry()

    def sync():
        return None

    with pytest.raises(TypeError, match="async function"):
        registry.tool(idempotent=True)(sync)
    assert registry.snapshot() == {}

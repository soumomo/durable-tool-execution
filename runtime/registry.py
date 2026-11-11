from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeAlias, TypeVar

from .models import JSONValue, ToolPolicy

ToolFunction: TypeAlias = Callable[..., Awaitable[JSONValue]]
_ToolFunction = TypeVar("_ToolFunction", bound=ToolFunction)


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    policy: ToolPolicy
    function: ToolFunction
    _signature: inspect.Signature = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not inspect.iscoroutinefunction(self.function):
            raise TypeError("Tool must be an async function, not an async generator")
        signature = inspect.signature(self.function)
        for parameter in signature.parameters.values():
            if (
                parameter.kind is inspect.Parameter.POSITIONAL_ONLY
                and parameter.default is inspect.Parameter.empty
            ):
                raise TypeError(
                    f"Tool parameter {parameter.name!r} is required and positional-only; "
                    "tools are called with keyword arguments"
                )
        object.__setattr__(self, "_signature", signature)

    def validate_arguments(self, args: dict[str, JSONValue]) -> None:
        """Bind arguments without executing the tool or validating value schemas."""
        self._signature.bind(**args)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, policy: ToolPolicy, function: ToolFunction) -> RegisteredTool:
        if policy.name in self._tools:
            raise ValueError(f"Tool {policy.name!r} is already registered")
        tool = RegisteredTool(policy=policy, function=function)
        self._tools[policy.name] = tool
        return tool

    def get(self, name: str) -> RegisteredTool:
        return self._tools[name]

    def snapshot(self) -> dict[str, RegisteredTool]:
        """Copy membership so later registrations cannot change a runtime's tools."""
        return self._tools.copy()

    def tool(
        self,
        *,
        name: str | None = None,
        idempotent: bool,
        timeout_ms: int = 2000,
        max_retries: int = 3,
        base_backoff_ms: int = 50,
    ) -> Callable[[_ToolFunction], _ToolFunction]:
        def decorate(function: _ToolFunction) -> _ToolFunction:
            policy = ToolPolicy(
                name=function.__name__ if name is None else name,
                idempotent=idempotent,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
                base_backoff_ms=base_backoff_ms,
            )
            self.register(policy, function)
            return function

        return decorate

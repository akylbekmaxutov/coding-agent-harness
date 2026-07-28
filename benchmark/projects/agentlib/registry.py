"""A tool registry: name -> handler plus schema, with argument checking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


class ToolError(Exception):
    """Anything the caller got wrong about a tool. Never a bug in the handler."""


@dataclass(frozen=True)
class Tool:
    name: str
    handler: Callable[..., Any]
    required: tuple[str, ...] = ()
    description: str = ""

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "required": list(self.required),
        }


@dataclass
class Registry:
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, name, handler, required=(), description="") -> Tool:
        if name in self._tools:
            raise ToolError(f"tool {name!r} is already registered")
        self._tools[name] = Tool(name, handler, tuple(required), description)
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict]:
        return [self._tools[name].schema() for name in self.names()]

    def missing_arguments(self, tool: Tool, arguments: dict) -> list[str]:
        return [name for name in tool.required if name not in arguments]

    def call(self, name: str, arguments: dict) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"no tool named {name!r}; have {', '.join(self.names()) or 'none'}")
        missing = self.missing_arguments(tool, arguments)
        if missing:
            raise ToolError(f"{name} is missing required arguments: {', '.join(missing)}")
        return tool.handler(**arguments)

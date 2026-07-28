from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from ..config import ContextConfig
from ..types import CodeSlice, Frame

_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", "node_modules"}
_DEF = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True)
class Symbol:
    path: str          # repo-relative
    qualname: str      # "shipping_fee" or "Cart.total"
    node: ast.AST
    source: str        # full file text, for line extraction


class SymbolIndex:
    """Every function, method and class in the repo, addressable by name.

    Resolution is deliberately naive: names within a module, `from x import y`
    targets, and unique method names across the project. No type inference, no
    cross-module dataflow. It covers the common single-repo bug, and where it
    cannot resolve a call it returns nothing rather than guessing -- the agent
    can always ask for more with read_file.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.by_path: dict[str, dict[str, Symbol]] = {}
        self.imports: dict[str, dict[str, str]] = {}   # path -> {alias: module}
        self.methods: dict[str, list[Symbol]] = {}     # bare method name -> symbols
        self.functions: dict[str, list[Symbol]] = {}   # bare function name -> symbols
        self._build()

    def _build(self) -> None:
        for file in sorted(self.root.rglob("*.py")):
            if any(part in _SKIP_DIRS for part in file.parts):
                continue
            try:
                source = file.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            rel = str(file.relative_to(self.root))
            symbols: dict[str, Symbol] = {}
            self.imports[rel] = {}

            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module:
                    for alias in node.names:
                        self.imports[rel][alias.asname or alias.name] = node.module
                elif isinstance(node, _DEF):
                    sym = Symbol(rel, node.name, node, source)
                    symbols[node.name] = sym
                    self.functions.setdefault(node.name, []).append(sym)
                elif isinstance(node, ast.ClassDef):
                    symbols[node.name] = Symbol(rel, node.name, node, source)
                    for item in node.body:
                        if isinstance(item, _DEF):
                            qual = f"{node.name}.{item.name}"
                            sym = Symbol(rel, qual, item, source)
                            symbols[qual] = sym
                            self.methods.setdefault(item.name, []).append(sym)

            self.by_path[rel] = symbols

    # -- lookup ------------------------------------------------------------

    def enclosing(self, rel_path: str, lineno: int) -> Symbol | None:
        """Smallest def containing lineno; falls back to the enclosing class."""
        best: Symbol | None = None
        for sym in self.by_path.get(rel_path, {}).values():
            start, end = _span(sym.node)
            if start <= lineno <= end:
                if best is None or _span(best.node)[0] < start:
                    best = sym
        return best

    def resolve_call(self, caller: Symbol, node: ast.Call) -> Symbol | None:
        func = node.func
        if isinstance(func, ast.Name):
            local = self.by_path.get(caller.path, {}).get(func.id)
            if local is not None:
                return self._narrow(local)
            module = self.imports.get(caller.path, {}).get(func.id)
            if module:
                target = self._module_path(module)
                if target and func.id in self.by_path.get(target, {}):
                    return self._narrow(self.by_path[target][func.id])
            return _unique(self.functions.get(func.id, []))
        if isinstance(func, ast.Attribute):
            candidates = self.methods.get(func.attr, [])
            same_file = [s for s in candidates if s.path == caller.path]
            return _unique(same_file) or _unique(candidates)
        return None

    def _narrow(self, sym: Symbol) -> Symbol | None:
        """A constructor call means __init__, not the whole class body.

        Emitting the class would duplicate every method it contains and blow
        the budget on code the traceback never implicated.
        """
        if not isinstance(sym.node, ast.ClassDef):
            return sym
        return self.by_path.get(sym.path, {}).get(f"{sym.qualname}.__init__")

    def _module_path(self, module: str) -> str | None:
        candidate = module.replace(".", "/") + ".py"
        if candidate in self.by_path:
            return candidate
        tail = candidate.rsplit("/", 1)[-1]
        matches = [p for p in self.by_path if p == tail or p.endswith("/" + tail)]
        return matches[0] if len(matches) == 1 else None


def slices_for_failure(
    repo_root: str | Path,
    frames: tuple[Frame, ...] | list[Frame],
    cfg: ContextConfig,
    index: SymbolIndex | None = None,
) -> list[CodeSlice]:
    """Build the code context for one failure, innermost frame first."""
    index = index or SymbolIndex(repo_root)
    slices: list[CodeSlice] = []
    seen: set[tuple[str, str]] = set()

    in_repo = [f for f in frames if f.in_repo]
    ordered = list(reversed(in_repo))                 # innermost first

    if cfg.whole_file_context:
        return whole_files(repo_root, ordered)

    roots: list[Symbol] = []
    for frame in ordered:
        sym = index.enclosing(frame.path, frame.lineno)
        if sym is None or (sym.path, sym.qualname) in seen:
            continue
        seen.add((sym.path, sym.qualname))
        slices.append(_to_slice(sym, "traceback frame"))
        roots.append(sym)

    if not cfg.include_external_frames:
        pass  # external frames are dropped entirely; one line each was noise

    frontier = roots
    for depth in range(1, cfg.callee_depth + 1):
        next_frontier: list[Symbol] = []
        for caller in frontier:
            for call in _calls_in(caller.node):
                target = index.resolve_call(caller, call)
                if target is None or (target.path, target.qualname) in seen:
                    continue
                seen.add((target.path, target.qualname))
                slices.append(_to_slice(target, f"called by {caller.qualname}"))
                next_frontier.append(target)
        frontier = next_frontier
        if not frontier:
            break

    return slices


def whole_files(repo_root: str | Path, frames) -> list[CodeSlice]:
    """Every file the traceback named, entire. The no-slicing baseline.

    Deliberately the same entry points as the slicer with the AST work removed,
    so the ablation measures symbol extraction and call-graph expansion rather
    than a different retrieval strategy. It is a weak baseline on purpose: a
    plain assertion names only the test file, and dumping that whole file still
    does not contain the bug.
    """
    root = Path(repo_root)
    slices: list[CodeSlice] = []
    seen: set[str] = set()
    for frame in frames:
        if frame.path in seen:
            continue
        seen.add(frame.path)
        try:
            source = (root / frame.path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        slices.append(
            CodeSlice(
                path=frame.path,
                symbol="<module>",
                start_line=1,
                end_line=len(source.splitlines()) or 1,
                source=source,
                reason="whole file (slicing disabled)",
            )
        )
    return slices


def _calls_in(node: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def _span(node: ast.AST) -> tuple[int, int]:
    start = getattr(node, "lineno", 1)
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        start = min(start, min(d.lineno for d in decorators))
    return start, getattr(node, "end_lineno", start)


def _to_slice(sym: Symbol, reason: str) -> CodeSlice:
    start, end = _span(sym.node)
    lines = sym.source.splitlines()
    return CodeSlice(
        path=sym.path,
        symbol=sym.qualname,
        start_line=start,
        end_line=end,
        source="\n".join(lines[start - 1 : end]),
        reason=reason,
    )


def _unique(symbols: list[Symbol]) -> Symbol | None:
    """One candidate means a confident resolution; several means guessing."""
    return symbols[0] if len(symbols) == 1 else None
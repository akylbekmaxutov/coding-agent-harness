from __future__ import annotations

import shutil
import subprocess
import tempfile
from fnmatch import fnmatch
from pathlib import Path

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".git", ".venv")
_GIT_ID = ("-c", "user.name=green-agent", "-c", "user.email=agent@localhost")


class WorkspaceError(RuntimeError):
    pass


class PathEscape(WorkspaceError):
    """A model-supplied path pointed outside the workspace."""


class Workspace:
    def __init__(
        self,
        source: str | Path,
        root: str | Path | None = None,
        test_globs: tuple[str, ...] = ("tests/**", "test_*.py", "*_test.py", "conftest.py"),
        keep: bool = False,
    ) -> None:
        self.source = Path(source).resolve()
        self._requested_root = Path(root).resolve() if root else None
        self.test_globs = test_globs
        self.keep = keep
        self.root: Path | None = None
        self.baseline: str | None = None
        self._is_worktree = False

    # -- lifecycle ---------------------------------------------------------

    def setup(self) -> "Workspace":
        if not self.source.is_dir():
            raise WorkspaceError(f"Not a directory: {self.source}")

        self.root = self._requested_root or Path(tempfile.mkdtemp(prefix="green-agent-"))

        # A directory counts as a repo only if it is a repo *root*. A plain
        # folder sitting inside someone's checkout gets copied, not worktree'd,
        # so we never accidentally pull in the parent project.
        if (self.source / ".git").exists():
            if self.root.exists():
                shutil.rmtree(self.root)
            self._git(self.source, "worktree", "add", "--detach", str(self.root), "HEAD")
            self._is_worktree = True
        else:
            if self.root.exists():
                shutil.rmtree(self.root)
            shutil.copytree(self.source, self.root, ignore=_IGNORE)
            self._git(self.root, "init", "-q")
            self._commit("baseline")

        self.baseline = self._head()
        return self

    def teardown(self) -> None:
        if self.root is None or self.keep:
            return
        if self._is_worktree:
            self._git(self.source, "worktree", "remove", "--force", str(self.root), check=False)
        shutil.rmtree(self.root, ignore_errors=True)
        self.root = None

    def __enter__(self) -> "Workspace":
        return self.setup()

    def __exit__(self, *exc) -> None:
        self.teardown()

    # -- safety ------------------------------------------------------------

    def resolve(self, rel_path: str) -> Path:
        """Resolve a path inside the workspace or refuse it."""
        if self.root is None:
            raise WorkspaceError("Workspace not set up.")
        root = self.root.resolve()
        candidate = (root / rel_path).resolve()
        if candidate != root and not candidate.is_relative_to(root):
            raise PathEscape(f"Path {rel_path!r} resolves outside the workspace.")
        return candidate

    def is_test_file(self, rel_path: str) -> bool:
        norm = str(rel_path).replace("\\", "/").lstrip("./")
        name = norm.rsplit("/", 1)[-1]
        return any(fnmatch(norm, g) or fnmatch(name, g) for g in self.test_globs)

    # -- state -------------------------------------------------------------

    def checkpoint(self, label: str = "checkpoint") -> str:
        """Commit the current state and return a handle to restore it."""
        self._commit(label)
        return self._head()

    def restore(self, handle: str) -> None:
        self._git(self.root, "reset", "--hard", "-q", handle)
        self._git(self.root, "clean", "-fdq")

    def diff(self, since: str | None = None) -> str:
        """Unified diff against the baseline, including untracked files."""
        self._git(self.root, "add", "-A", "-N")
        return self._git(self.root, "diff", since or self.baseline or "HEAD")

    def changed_files(self, since: str | None = None) -> list[str]:
        self._git(self.root, "add", "-A", "-N")
        out = self._git(self.root, "diff", "--name-only", since or self.baseline or "HEAD")
        return [line for line in out.splitlines() if line]

    # -- git ---------------------------------------------------------------

    def _commit(self, message: str) -> None:
        self._git(self.root, "add", "-A")
        self._git(self.root, *_GIT_ID, "commit", "-q", "--allow-empty", "-m", message)

    def _head(self) -> str:
        return self._git(self.root, "rev-parse", "HEAD").strip()

    @staticmethod
    def _git(cwd: Path | None, *args: str, check: bool = True) -> str:
        proc = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
        if check and proc.returncode != 0:
            raise WorkspaceError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc.stdout
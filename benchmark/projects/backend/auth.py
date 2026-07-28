"""Role to rights. Every permission decision in the service goes through here."""

from __future__ import annotations

from errors import Forbidden

ROLE_RIGHTS = {
    "admin": frozenset({"read", "write", "delete"}),
    "editor": frozenset({"read", "write"}),
    "viewer": frozenset({"read"}),
}


def rights_for(role: str) -> frozenset[str]:
    # An unknown role gets nothing rather than a default, so a typo in a role
    # name can never widen access.
    return ROLE_RIGHTS.get(role, frozenset())


def is_allowed(role: str, right: str) -> bool:
    return right in rights_for(role)


def require(role: str, right: str) -> None:
    if not is_allowed(role, right):
        raise Forbidden(f"role {role!r} may not {right}")

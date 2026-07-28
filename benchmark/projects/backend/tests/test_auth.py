import pytest

from auth import is_allowed, require, rights_for
from errors import Forbidden


def test_admin_may_delete():
    assert is_allowed("admin", "delete") is True


def test_viewer_may_not_write():
    assert is_allowed("viewer", "write") is False


def test_unknown_role_has_no_rights():
    assert rights_for("intern") == frozenset()


def test_require_raises_forbidden_and_names_the_role():
    with pytest.raises(Forbidden) as err:
        require("viewer", "delete")
    assert "viewer" in str(err.value)


def test_require_is_silent_when_allowed():
    assert require("editor", "write") is None

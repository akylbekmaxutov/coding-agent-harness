import pytest

from service import handle
from store import Store


@pytest.fixture
def store() -> Store:
    return Store({"a": {"title": "alpha"}, "b": {"title": "beta"}, "c": {"title": "gamma"}})


def test_list_returns_every_record(store):
    response = handle({"action": "list", "role": "viewer"}, store)
    assert response["status"] == 200
    assert len(response["body"]["items"]) == 3


def test_missing_record_is_not_found(store):
    response = handle({"action": "get", "role": "viewer", "id": "zz"}, store)
    assert response["status"] == 404


def test_viewer_cannot_delete(store):
    response = handle({"action": "delete", "role": "viewer", "id": "a"}, store)
    assert response["status"] == 403


def test_admin_delete_removes_the_record(store):
    assert handle({"action": "delete", "role": "admin", "id": "a"}, store)["status"] == 204
    assert handle({"action": "get", "role": "admin", "id": "a"}, store)["status"] == 404


def test_unknown_action_is_a_bad_request(store):
    response = handle({"action": "teleport", "role": "admin"}, store)
    assert response["status"] == 400
    assert "teleport" in response["body"]["detail"]

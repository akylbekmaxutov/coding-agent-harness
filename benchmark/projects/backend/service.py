"""Request routing. A request is a dict; a response is a status plus a body."""

from __future__ import annotations

from auth import require
from errors import AppError, BadRequest, error_body, status_for
from pagination import DEFAULT_PAGE_SIZE, paginate


def list_records(request: dict, store) -> dict:
    require(request.get("role", "viewer"), "read")
    page = paginate(
        store.list(),
        request.get("page", 1),
        request.get("size", DEFAULT_PAGE_SIZE),
    )
    return {"status": 200, "body": {"items": page.items, "pages": page.pages}}


def get_record(request: dict, store) -> dict:
    require(request.get("role", "viewer"), "read")
    return {"status": 200, "body": store.get(request["id"])}


def put_record(request: dict, store) -> dict:
    require(request.get("role", "viewer"), "write")
    return {"status": 200, "body": store.put(request["id"], request.get("body", {}))}


def delete_record(request: dict, store) -> dict:
    require(request.get("role", "viewer"), "delete")
    store.delete(request["id"])
    return {"status": 204, "body": None}


ROUTES = {
    "list": list_records,
    "get": get_record,
    "put": put_record,
    "delete": delete_record,
}


def handle(request: dict, store) -> dict:
    handler = ROUTES.get(request.get("action"))
    # An unroutable request is the caller's mistake, and it has to be answered
    # here: past this point there is no handler to raise anything.
    if handler is None:
        unknown = BadRequest(f"unknown action {request.get('action')!r}")
        return {"status": status_for(unknown), "body": error_body(unknown)}
    try:
        return handler(request, store)
    except AppError as exc:
        return {"status": status_for(exc), "body": error_body(exc)}

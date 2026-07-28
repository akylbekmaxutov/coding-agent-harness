"""Errors, and the single place they become HTTP status codes."""

from __future__ import annotations


class AppError(Exception):
    """Base for every error the service is willing to answer with."""

    status = 500


class BadRequest(AppError):
    status = 400


class Forbidden(AppError):
    status = 403


class NotFound(AppError):
    status = 404


STATUS_TEXT = {
    400: "Bad Request",
    403: "Forbidden",
    404: "Not Found",
    500: "Internal Server Error",
}


def status_for(exc: Exception) -> int:
    if isinstance(exc, AppError):
        return exc.status
    return 500


def error_body(exc: Exception) -> dict:
    status = status_for(exc)
    return {"error": STATUS_TEXT.get(status, "Error"), "detail": str(exc)}

"""Offset pagination over an already-materialised list of records."""

from __future__ import annotations

from dataclasses import dataclass

from errors import BadRequest, NotFound

DEFAULT_PAGE_SIZE = 10


@dataclass(frozen=True)
class Page:
    items: list
    page: int
    size: int
    total: int
    pages: int


def page_count(total: int, size: int) -> int:
    if size < 1:
        raise BadRequest("page size must be at least 1")
    if total <= 0:
        return 0
    # Ceiling division: a partial final page still counts, and a total that
    # divides exactly must not produce an empty extra page on the end.
    return (total + size - 1) // size


def window(page: int, size: int) -> tuple[int, int]:
    start = (page - 1) * size
    return start, start + size


def paginate(records, page: int = 1, size: int = DEFAULT_PAGE_SIZE) -> Page:
    total = len(records)
    pages = page_count(total, size)
    if page < 1:
        raise BadRequest("page numbers start at 1")
    if pages and page > pages:
        raise NotFound(f"page {page} does not exist; there are {pages}")
    start, end = window(page, size)
    return Page(
        items=list(records[start:end]),
        page=page,
        size=size,
        total=total,
        pages=pages,
    )

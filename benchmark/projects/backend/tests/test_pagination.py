import pytest

from errors import BadRequest, NotFound
from pagination import page_count, paginate


def records(n: int) -> list[dict]:
    return [{"id": f"r{i:02d}"} for i in range(n)]


def test_first_page_holds_one_page_size():
    page = paginate(records(25), page=1, size=10)
    assert len(page.items) == 10
    assert page.items[0]["id"] == "r00"


def test_partial_last_page_is_shorter():
    page = paginate(records(25), page=3, size=10)
    assert len(page.items) == 5


def test_last_page_is_not_empty():
    # 20 records at 10 per page is exactly two pages. The last page number the
    # caller is offered must contain records, or the client paginates into a
    # void it was told exists.
    page = paginate(records(20), page=1, size=10)
    assert page.pages == 2
    last = paginate(records(20), page=page.pages, size=10)
    assert last.items != []


def test_page_beyond_the_end_is_not_found():
    with pytest.raises(NotFound):
        paginate(records(25), page=4, size=10)


def test_page_size_below_one_is_rejected():
    with pytest.raises(BadRequest):
        page_count(10, 0)


def test_empty_input_has_no_pages():
    page = paginate([], page=1, size=10)
    assert (page.pages, page.items) == (0, [])

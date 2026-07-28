from errors import BadRequest, Forbidden, NotFound, error_body, status_for


def test_status_comes_from_the_error_class():
    assert (status_for(NotFound("x")), status_for(Forbidden("x"))) == (404, 403)


def test_unknown_exception_is_a_server_error():
    assert status_for(ValueError("boom")) == 500


def test_error_body_names_the_status_and_keeps_the_detail():
    body = error_body(BadRequest("page size must be at least 1"))
    assert body["error"] == "Bad Request"
    assert "page size" in body["detail"]

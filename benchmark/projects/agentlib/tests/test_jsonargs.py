import pytest

from jsonargs import ArgumentError, extract_object, strip_fences


def test_plain_object_is_parsed():
    assert extract_object('{"path": "cart.py"}') == {"path": "cart.py"}


def test_fenced_object_is_unwrapped():
    assert extract_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_fences_without_a_language_tag_work_too():
    assert strip_fences("```\n{}\n```") == "{}"


def test_prose_around_a_fence_is_ignored():
    assert extract_object('Sure, here you go:\n```json\n{"n": 2}\n```\nHope that helps!') == {"n": 2}


def test_prose_that_opens_a_brace_is_not_arguments():
    # "there is no object here" and "the object here is broken" send the caller
    # in different directions, so a sentence must not be reported as malformed
    # JSON.
    with pytest.raises(ArgumentError) as err:
        extract_object("{the harness} should read cart.py next")
    assert "no JSON object found" in str(err.value)


def test_malformed_object_is_reported_as_invalid_json():
    with pytest.raises(ArgumentError) as err:
        extract_object('{"a": }')
    assert "invalid JSON" in str(err.value)


def test_an_empty_reply_is_refused():
    with pytest.raises(ArgumentError):
        extract_object("")

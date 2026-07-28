import pytest

from coerce import CoerceError, coerce_record, to_bool, to_float, to_int


def test_int_from_padded_text():
    assert to_int("  42 ") == 42


def test_float_accepts_integer_text():
    assert to_float("7") == 7.0


def test_bool_reads_both_vocabularies():
    assert (to_bool("YES"), to_bool("off")) == (True, False)


def test_bool_passes_through_real_booleans():
    assert to_bool(False) is False


def test_unrecognised_bool_word_is_an_error():
    with pytest.raises(CoerceError):
        to_bool("maybe")


def test_coerce_record_applies_the_spec_and_leaves_the_rest():
    out = coerce_record({"n": "3", "flag": "y", "note": "hi"}, {"n": "int", "flag": "bool"})
    assert out == {"n": 3, "flag": True, "note": "hi"}


def test_coerce_record_names_the_field_that_failed():
    with pytest.raises(CoerceError) as err:
        coerce_record({"n": "twelve"}, {"n": "int"})
    assert str(err.value).startswith("n:")


def test_absent_field_is_not_invented():
    assert coerce_record({"a": "1"}, {"b": "int"}) == {"a": "1"}

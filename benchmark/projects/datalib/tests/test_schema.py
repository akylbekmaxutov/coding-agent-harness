from schema import Field, is_blank, validate

FIELDS = [
    Field("name", "str"),
    Field("amount", "float"),
    Field("note", "str", required=False),
]


def test_valid_record_has_no_errors():
    assert validate({"name": "ada", "amount": "10.5", "note": "x"}, FIELDS) == []


def test_missing_required_field_is_reported():
    errors = validate({"amount": "10.5"}, FIELDS)
    assert errors == ["name: required field is missing"]


def test_optional_field_may_be_absent():
    assert validate({"name": "ada", "amount": "1"}, FIELDS) == []


def test_wrong_type_is_reported_with_the_field_name():
    errors = validate({"name": "ada", "amount": "lots"}, FIELDS)
    assert len(errors) == 1 and errors[0].startswith("amount:")


def test_whitespace_counts_as_blank():
    assert is_blank("   ") is True
    assert is_blank("0") is False

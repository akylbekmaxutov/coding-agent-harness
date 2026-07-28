import pytest

from loader import LoadError, parse_table, split_rows

CSV = """\
name,region,amount
ada,north,10.50
grace,south,3.25
"""


def test_header_becomes_the_record_keys():
    records = parse_table(CSV)
    assert len(records) == 2
    assert records[0] == {"name": "ada", "region": "north", "amount": "10.50"}


def test_blank_lines_are_skipped():
    assert split_rows("a\n\n   \nb\n") == ["a", "b"]


def test_cells_are_stripped():
    assert parse_table("a, b\n1 ,  2 \n")[0] == {"a": "1", "b": "2"}


def test_row_with_the_wrong_field_count_is_rejected():
    with pytest.raises(LoadError) as err:
        parse_table("a,b\n1,2,3\n")
    assert "row 2" in str(err.value)


def test_empty_input_yields_no_records():
    assert parse_table("   \n") == []


def test_header_only_input_yields_no_records():
    assert parse_table("a,b\n") == []


def test_custom_delimiter():
    assert parse_table("a\tb\n1\t2\n", delimiter="\t") == [{"a": "1", "b": "2"}]

from aggregate import Summary, group_by, mean, summarise, top_groups

ROWS = [
    {"region": "north", "amount": "10.00"},
    {"region": "north", "amount": "20.00"},
    {"region": "south", "amount": "4.00"},
    {"region": "south", "amount": "6.00"},
]


def test_group_by_collects_rows_under_their_key():
    groups = group_by(ROWS, "region")
    assert sorted(groups) == ["north", "south"]
    assert len(groups["north"]) == 2


def test_summarise_counts_and_totals_each_group():
    out = summarise(ROWS, "region", "amount")
    assert out["north"].count == 2
    assert out["north"].total == 30.00
    assert out["south"].total == 10.00


def test_mean_of_an_empty_group_is_zero():
    assert mean([]) == 0.0


def test_group_of_one_averages_to_its_own_value():
    # A region with a single sale still has an average, and it is that sale.
    out = summarise([{"region": "east", "amount": "12.50"}], "region", "amount")
    assert out["east"].mean == 12.50


def test_top_groups_defaults_to_three():
    summaries = {
        "a": Summary(1, 40.0, 40.0),
        "b": Summary(1, 30.0, 30.0),
        "c": Summary(1, 20.0, 20.0),
        "d": Summary(1, 10.0, 10.0),
    }
    assert top_groups(summaries) == ["a", "b", "c"]


def test_top_groups_honours_an_explicit_limit():
    summaries = {"a": Summary(1, 40.0, 40.0), "b": Summary(1, 30.0, 30.0)}
    assert top_groups(summaries, limit=1) == ["a"]

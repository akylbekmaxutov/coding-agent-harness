from budget import TurnBudget


def test_a_fresh_budget_is_not_exhausted():
    assert TurnBudget().exhausted is False


def test_turn_limit_ends_the_run():
    budget = TurnBudget(max_turns=2)
    budget.charge(10)
    assert budget.exhausted is False
    budget.charge(10)
    assert budget.exhausted is True
    assert "turn limit" in budget.reason


def test_token_limit_ends_the_run_before_the_turns_run_out():
    budget = TurnBudget(max_turns=12, max_tokens=100)
    budget.charge(150)
    assert budget.exhausted is True
    assert "token limit" in budget.reason


def test_remaining_turns_never_goes_negative():
    budget = TurnBudget(max_turns=1)
    budget.charge()
    budget.charge()
    assert budget.remaining_turns() == 0

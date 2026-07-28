import pytest

from retry import RetryPolicy, next_delay, run_with_retries, should_retry


def test_delay_grows_geometrically():
    # 0.5, then doubled each attempt. The first retry waits base_delay, not
    # nothing: a zero first wait is a thundering herd against a struggling
    # endpoint.
    policy = RetryPolicy(base_delay=0.5, factor=2.0, max_delay=8.0)
    assert [next_delay(policy, attempt) for attempt in range(3)] == [0.5, 1.0, 2.0]


def test_delay_is_capped():
    policy = RetryPolicy(base_delay=1.0, factor=10.0, max_delay=8.0)
    assert next_delay(policy, 5) == 8.0


def test_listed_error_is_retried():
    assert should_retry(RetryPolicy(), 0, TimeoutError("slow")) is True


def test_unlisted_error_is_not_retried():
    assert should_retry(RetryPolicy(), 0, ValueError("nope")) is False


def test_last_attempt_is_not_retried():
    assert should_retry(RetryPolicy(max_attempts=3), 2, TimeoutError()) is False


def test_run_with_retries_succeeds_on_a_later_attempt():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("slow")
        return "ok"

    assert run_with_retries(flaky, RetryPolicy(max_attempts=3)) == "ok"
    assert calls["n"] == 3


def test_run_with_retries_gives_up_and_reraises():
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise TimeoutError("slow")

    with pytest.raises(TimeoutError):
        run_with_retries(always_fails, RetryPolicy(max_attempts=2))
    assert calls["n"] == 2


def test_run_with_retries_does_not_retry_an_unlisted_error():
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise ValueError("nope")

    with pytest.raises(ValueError):
        run_with_retries(bad)
    assert calls["n"] == 1

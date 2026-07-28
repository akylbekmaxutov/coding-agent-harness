"""Retry policy: which errors are worth another attempt, and how long to wait."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.5
    factor: float = 2.0
    max_delay: float = 8.0
    retry_on: tuple[str, ...] = ("TimeoutError", "ConnectionError")


def backoff_delay(base: float, attempt: int, factor: float, cap: float) -> float:
    """Geometric backoff, capped. `attempt` is 0 for the first retry."""
    return round(min(cap, base * factor**attempt), 3)


def next_delay(policy: RetryPolicy, attempt: int) -> float:
    return backoff_delay(policy.base_delay, attempt, policy.factor, policy.max_delay)


def should_retry(policy: RetryPolicy, attempt: int, error: BaseException) -> bool:
    # `attempt` counts retries already made, so the first call is attempt 0 and
    # max_attempts counts calls, not retries.
    if attempt + 1 >= policy.max_attempts:
        return False
    return type(error).__name__ in policy.retry_on


def run_with_retries(call, policy: RetryPolicy | None = None, sleep=lambda _seconds: None):
    policy = policy or RetryPolicy()
    attempt = 0
    while True:
        try:
            return call()
        except Exception as exc:
            if not should_retry(policy, attempt, exc):
                raise
            sleep(next_delay(policy, attempt))
            attempt += 1

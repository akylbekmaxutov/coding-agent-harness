"""Turn and token budget for an agent loop. Whichever limit trips first ends it."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnBudget:
    max_turns: int = 12
    max_tokens: int = 150_000
    turns: int = 0
    tokens: int = 0

    def charge(self, tokens: int = 0) -> None:
        self.turns += 1
        self.tokens += max(0, tokens)

    @property
    def exhausted(self) -> bool:
        return self.turns >= self.max_turns or self.tokens >= self.max_tokens

    @property
    def reason(self) -> str:
        if self.turns >= self.max_turns:
            return f"turn limit ({self.max_turns})"
        if self.tokens >= self.max_tokens:
            return f"token limit ({self.max_tokens})"
        return "not exhausted"

    def remaining_turns(self) -> int:
        return max(0, self.max_turns - self.turns)

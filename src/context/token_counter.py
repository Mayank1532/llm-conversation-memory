from typing import Protocol


class TokenCounter(Protocol):
    """Protocol for counting tokens in conversation messages."""

    def count(self, messages: list[dict[str, str]]) -> int:
        """Return the number of tokens represented by messages."""
        ...

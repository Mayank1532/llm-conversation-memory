from src.context.token_counter import TokenCounter


class ContextManager:
    """Select conversation messages within a token budget."""

    def __init__(
        self,
        token_counter: TokenCounter,
        max_tokens: int = 512,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1.")

        self.token_counter = token_counter
        self.max_tokens = max_tokens

    def select_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        if not messages:
            return []

        selected: list[dict[str, str]] = []

        for message in reversed(messages):
            candidate = [message, *reversed(selected)]

            if self.token_counter.count(candidate) > self.max_tokens:
                if not selected:
                    return [message]

                break

            selected.insert(0, message)

        return selected

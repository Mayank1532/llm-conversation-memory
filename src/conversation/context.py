class ContextManager:
    """Select the conversation messages sent to the LLM."""

    def __init__(self, max_messages: int = 10) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1.")

        self.max_messages = max_messages

    def select_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        if not messages:
            return []

        return list(messages[-self.max_messages:])

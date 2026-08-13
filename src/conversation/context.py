from src.context.token_counter import TokenCounter


class ContextManager:
    """Build token-aware LLM context from history and persistent memory."""

    def __init__(
        self,
        token_counter: TokenCounter,
        max_tokens: int = 512,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1.")

        self.token_counter = token_counter
        self.max_tokens = max_tokens

    def _build_memory_message(
        self,
        memories: dict[str, str],
    ) -> dict[str, str] | None:
        if not memories:
            return None

        lines = [
            "Persistent memory about the user:",
        ]

        for key, value in memories.items():
            lines.append(f"- {key}: {value}")

        return {
            "role": "system",
            "content": "\n".join(lines),
        }

    def select_messages(
        self,
        messages: list[dict[str, str]],
        memories: dict[str, str] | None = None,
    ) -> list[dict[str, str]]:
        """Return persistent memory plus the newest history within the budget."""

        memory_message = self._build_memory_message(memories or {})

        if not messages and memory_message is None:
            return []

        selected_history: list[dict[str, str]] = []

        for message in reversed(messages):
            candidate_history = [message, *selected_history]

            if memory_message is not None:
                candidate = [memory_message, *candidate_history]
            else:
                candidate = candidate_history

            if (
                self.token_counter.count(candidate) > self.max_tokens
                and selected_history
            ):
                break

            selected_history.insert(0, message)

        if memory_message is not None:
            return [memory_message, *selected_history]

        return selected_history

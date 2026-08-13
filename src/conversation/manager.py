from src.conversation.context import ContextManager
from src.llm.local_llm import LocalLLM


class ConversationManager:
    """Manage conversation history and model context."""

    def __init__(
        self,
        llm: LocalLLM,
        context_manager: ContextManager | None = None,
    ) -> None:
        self.llm = llm
        self.context_manager = context_manager or ContextManager()
        self.messages: list[dict[str, str]] = []

    def add_user_message(self, content: str) -> None:
        if not content.strip():
            raise ValueError("User message cannot be empty.")

        self.messages.append(
            {
                "role": "user",
                "content": content.strip(),
            }
        )

    def add_assistant_message(self, content: str) -> None:
        if not content.strip():
            raise ValueError("Assistant message cannot be empty.")

        self.messages.append(
            {
                "role": "assistant",
                "content": content.strip(),
            }
        )

    def get_context(self) -> list[dict[str, str]]:
        """Return the messages that will be sent to the LLM."""
        return self.context_manager.select_messages(self.messages)

    def generate_response(self) -> str:
        if not self.messages:
            raise ValueError(
                "Cannot generate a response without a user message."
            )

        context = self.get_context()

        response = self.llm.generate(context)

        self.add_assistant_message(response)

        return response

    def get_history(self) -> list[dict[str, str]]:
        """Return the complete conversation history."""
        return list(self.messages)

    def reset(self) -> None:
        self.messages.clear()

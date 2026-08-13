from src.conversation.context import ContextManager
from src.llm.local_llm import LocalLLM
from src.memory.store import MemoryStore


class ConversationManager:
    """Manage conversation history, memory, and model context."""

    def __init__(
        self,
        llm: LocalLLM,
        context_manager: ContextManager,
        memory_store: MemoryStore,
    ) -> None:
        self.llm = llm
        self.context_manager = context_manager
        self.memory_store = memory_store
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
        """Return the messages selected for the LLM."""
        return self.context_manager.select_messages(self.messages)

    def generate_response(self) -> str:
        if not self.messages:
            raise ValueError(
                "Cannot generate a response without a user message."
            )

        response = self.llm.generate(self.get_context())

        self.add_assistant_message(response)

        return response

    def remember(self, key: str, value: str) -> None:
        """Store an explicit long-term memory."""
        self.memory_store.remember(key, value)

    def recall(self, key: str) -> str | None:
        """Retrieve an explicit long-term memory."""
        return self.memory_store.recall(key)

    def get_memories(self) -> dict[str, str]:
        """Return all stored memories."""
        return self.memory_store.get_all()

    def get_history(self) -> list[dict[str, str]]:
        """Return the complete conversation history."""
        return list(self.messages)

    def reset(self) -> None:
        """Clear conversation history but preserve memory."""
        self.messages.clear()

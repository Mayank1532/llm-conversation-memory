from src.llm.local_llm import LocalLLM


class ConversationManager:
    """Manage short-term conversation history for one conversation."""

    def __init__(self, llm: LocalLLM) -> None:
        self.llm = llm
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
        self.messages.append(
            {
                "role": "assistant",
                "content": content.strip(),
            }
        )

    def generate_response(self) -> str:
        if not self.messages:
            raise ValueError(
                "Cannot generate a response without a user message."
            )

        response = self.llm.generate(self.messages)

        self.add_assistant_message(response)

        return response

    def get_history(self) -> list[dict[str, str]]:
        return list(self.messages)

    def reset(self) -> None:
        self.messages.clear()

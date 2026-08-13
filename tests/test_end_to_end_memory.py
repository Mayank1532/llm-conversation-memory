from src.conversation.context import ContextManager
from src.conversation.manager import ConversationManager


class FakeTokenCounter:
    def count(self, messages: list[dict[str, str]]) -> int:
        return sum(len(message["content"]) for message in messages)


class FakeLLM:
    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 64,
    ) -> str:
        return "Test response"


def test_end_to_end_conversation_memory_flow() -> None:
    conversation = ConversationManager(
        llm=FakeLLM(),
        context_manager=ContextManager(
            token_counter=FakeTokenCounter(),
            max_tokens=100,
        ),
    )

    conversation.add_user_message("My name is Mayank.")
    conversation.generate_response()

    conversation.add_user_message("I am learning generative AI.")
    conversation.generate_response()

    conversation.add_user_message("What is my name?")

    assert conversation.recall("name") == "Mayank"
    assert conversation.recall("learning") == "generative AI"

    context = conversation.get_context()

    memory_messages = [
        message
        for message in context
        if message["role"] == "system"
    ]

    assert len(memory_messages) == 1
    assert "name: Mayank" in memory_messages[0]["content"]
    assert "learning: generative AI" in memory_messages[0]["content"]

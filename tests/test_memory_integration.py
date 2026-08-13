from src.conversation.context import ContextManager
from src.conversation.manager import ConversationManager


class FakeTokenCounter:
    def count(self, messages: list[dict[str, str]]) -> int:
        return sum(len(message["content"]) for message in messages)


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 64,
    ) -> str:
        self.calls.append(list(messages))
        return "Fake response"


def create_conversation() -> tuple[
    ConversationManager,
    FakeLLM,
]:
    llm = FakeLLM()

    context_manager = ContextManager(
        token_counter=FakeTokenCounter(),
        max_tokens=100,
    )

    conversation = ConversationManager(
        llm=llm,
        context_manager=context_manager,
    )

    return conversation, llm


def test_user_message_automatically_creates_memory() -> None:
    conversation, _ = create_conversation()

    conversation.add_user_message(
        "My name is Mayank."
    )

    assert conversation.get_memories() == {
        "name": "Mayank",
    }


def test_multiple_memories_are_extracted_automatically() -> None:
    conversation, _ = create_conversation()

    conversation.add_user_message(
        "I am learning generative AI."
    )

    conversation.add_user_message(
        "My goal is to become a data scientist."
    )

    memories = conversation.get_memories()

    assert memories["learning"] == "generative AI"
    assert memories["goal"] == "to become a data scientist"


def test_memory_is_sent_to_llm_automatically() -> None:
    conversation, llm = create_conversation()

    conversation.add_user_message(
        "My name is Mayank."
    )

    conversation.add_user_message(
        "What is my name?"
    )

    conversation.generate_response()

    assert len(llm.calls) == 1

    sent_context = llm.calls[0]

    memory_messages = [
        message
        for message in sent_context
        if message["role"] == "system"
    ]

    assert len(memory_messages) == 1
    assert "name: Mayank" in memory_messages[0]["content"]


def test_reset_clears_history_but_preserves_extracted_memory() -> None:
    conversation, _ = create_conversation()

    conversation.add_user_message(
        "My name is Mayank."
    )

    conversation.add_assistant_message(
        "Nice to meet you."
    )

    conversation.reset()

    assert conversation.get_history() == []

    assert conversation.get_memories() == {
        "name": "Mayank",
    }

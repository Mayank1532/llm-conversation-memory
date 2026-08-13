from src.conversation.context import ContextManager
from src.conversation.manager import ConversationManager
from src.memory.store import MemoryStore


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
        return "Fake assistant response"


def create_conversation() -> tuple[
    ConversationManager,
    FakeLLM,
    MemoryStore,
]:
    llm = FakeLLM()

    context_manager = ContextManager(
        token_counter=FakeTokenCounter(),
        max_tokens=100,
    )

    memory = MemoryStore()

    conversation = ConversationManager(
        llm=llm,
        context_manager=context_manager,
        memory_store=memory,
    )

    return conversation, llm, memory


def test_conversation_stores_full_history() -> None:
    conversation, _, _ = create_conversation()

    conversation.add_user_message("Hello")
    response = conversation.generate_response()

    assert response == "Fake assistant response"
    assert len(conversation.get_history()) == 2
    assert conversation.get_history()[0]["role"] == "user"
    assert conversation.get_history()[1]["role"] == "assistant"


def test_conversation_sends_selected_context_to_llm() -> None:
    conversation, llm, _ = create_conversation()

    conversation.add_user_message("11111")
    conversation.add_assistant_message("22222")
    conversation.add_user_message("33333")

    conversation.generate_response()

    assert len(llm.calls) == 1

    sent_context = llm.calls[0]

    assert sent_context[-1]["content"] == "33333"
    assert any(
    message["content"] == "11111"
    for message in sent_context
)


def test_empty_user_message_is_rejected() -> None:
    conversation, _, _ = create_conversation()

    try:
        conversation.add_user_message("   ")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_generate_without_history_is_rejected() -> None:
    conversation, _, _ = create_conversation()

    try:
        conversation.generate_response()
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_reset_clears_history_but_preserves_memory() -> None:
    conversation, _, _ = create_conversation()

    conversation.add_user_message("Hello")
    conversation.add_assistant_message("Hi")

    conversation.remember("name", "Mayank")

    conversation.reset()

    assert conversation.get_history() == []
    assert conversation.recall("name") == "Mayank"


def test_conversation_can_store_and_recall_memory() -> None:
    conversation, _, _ = create_conversation()

    conversation.remember("name", "Mayank")
    conversation.remember("goal", "Learn generative AI")

    assert conversation.recall("name") == "Mayank"
    assert conversation.recall("goal") == "Learn generative AI"


def test_get_memories_returns_all_stored_memory() -> None:
    conversation, _, _ = create_conversation()

    conversation.remember("name", "Mayank")
    conversation.remember("goal", "Learn generative AI")

    assert conversation.get_memories() == {
        "name": "Mayank",
        "goal": "Learn generative AI",
    }

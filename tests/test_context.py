from src.conversation.context import ContextManager


class FakeTokenCounter:
    """Deterministic token counter for context tests."""

    def count(self, messages: list[dict[str, str]]) -> int:
        return sum(len(message["content"]) for message in messages)


def test_select_messages_within_token_budget() -> None:
    counter = FakeTokenCounter()

    messages = [
        {"role": "user", "content": "11111"},
        {"role": "assistant", "content": "22222"},
        {"role": "user", "content": "33333"},
        {"role": "assistant", "content": "44444"},
        {"role": "user", "content": "55555"},
        {"role": "assistant", "content": "66666"},
    ]

    context_manager = ContextManager(
        token_counter=counter,
        max_tokens=15,
    )

    selected = context_manager.select_messages(messages)

    assert len(selected) == 3
    assert counter.count(selected) == 15
    assert selected[0]["content"] == "44444"
    assert selected[-1]["content"] == "66666"


def test_newest_message_is_preserved_when_oversized() -> None:
    counter = FakeTokenCounter()

    messages = [
        {"role": "user", "content": "old message"},
        {"role": "assistant", "content": "old response"},
        {
            "role": "user",
            "content": "this is a very long newest message",
        },
    ]

    context_manager = ContextManager(
        token_counter=counter,
        max_tokens=10,
    )

    selected = context_manager.select_messages(messages)

    assert len(selected) == 1
    assert selected[0]["content"] == "this is a very long newest message"
    assert counter.count(selected) > 10


def test_empty_history_returns_empty_context() -> None:
    counter = FakeTokenCounter()

    context_manager = ContextManager(
        token_counter=counter,
        max_tokens=100,
    )

    assert context_manager.select_messages([]) == []


def test_invalid_token_budget_is_rejected() -> None:
    counter = FakeTokenCounter()

    try:
        ContextManager(
            token_counter=counter,
            max_tokens=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
from src.conversation.context import ContextManager


class FakeTokenCounter:
    """Deterministic token counter for context tests."""

    def count(self, messages: list[dict[str, str]]) -> int:
        return sum(len(message["content"]) for message in messages)


def test_memory_is_included_in_context() -> None:
    counter = FakeTokenCounter()

    context_manager = ContextManager(
        token_counter=counter,
        max_tokens=100,
    )

    messages = [
        {
            "role": "user",
            "content": "Hello",
        }
    ]

    memories = {
        "name": "Mayank",
        "goal": "Learn generative AI",
    }

    selected = context_manager.select_messages(
        messages,
        memories=memories,
    )

    assert selected[0]["role"] == "system"
    assert "Persistent memory about the user:" in selected[0]["content"]
    assert "name: Mayank" in selected[0]["content"]
    assert "goal: Learn generative AI" in selected[0]["content"]


def test_memory_is_preserved_when_history_is_trimmed() -> None:
    counter = FakeTokenCounter()

    context_manager = ContextManager(
        token_counter=counter,
        max_tokens=80,
    )

    messages = [
        {
            "role": "user",
            "content": "This is an old conversation message.",
        },
        {
            "role": "assistant",
            "content": "This is an old assistant response.",
        },
        {
            "role": "user",
            "content": "This is a newer conversation message.",
        },
        {
            "role": "assistant",
            "content": "This is a newer assistant response.",
        },
    ]

    memories = {
        "name": "Mayank",
    }

    selected = context_manager.select_messages(
        messages,
        memories=memories,
    )

    memory_messages = [
        message
        for message in selected
        if message["role"] == "system"
    ]

    assert len(memory_messages) == 1
    assert "name: Mayank" in memory_messages[0]["content"]


def test_empty_memory_does_not_add_system_message() -> None:
    counter = FakeTokenCounter()

    context_manager = ContextManager(
        token_counter=counter,
        max_tokens=100,
    )

    selected = context_manager.select_messages(
        [
            {
                "role": "user",
                "content": "Hello",
            }
        ],
        memories={},
    )

    assert len(selected) == 1
    assert selected[0]["role"] == "user"


def test_memory_only_context_is_supported() -> None:
    counter = FakeTokenCounter()

    context_manager = ContextManager(
        token_counter=counter,
        max_tokens=100,
    )

    selected = context_manager.select_messages(
        [],
        memories={"name": "Mayank"},
    )

    assert len(selected) == 1
    assert selected[0]["role"] == "system"
    assert "name: Mayank" in selected[0]["content"]

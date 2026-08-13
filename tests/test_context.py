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

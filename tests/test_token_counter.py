from pathlib import Path

from src.context.huggingface_token_counter import HuggingFaceTokenCounter


MODEL_PATH = (
    Path(r"D:\HuggingFaceCache\hub")
    / "models--Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots"
    / "7ae557604adf67be50417f59c2c2f167def9a775"
)


def test_empty_messages_have_zero_tokens() -> None:
    counter = HuggingFaceTokenCounter(MODEL_PATH)

    assert counter.count([]) == 0


def test_token_count_increases_with_additional_message() -> None:
    counter = HuggingFaceTokenCounter(MODEL_PATH)

    messages = [
        {
            "role": "user",
            "content": "My name is Mayank.",
        }
    ]

    first_count = counter.count(messages)

    messages.append(
        {
            "role": "assistant",
            "content": "Nice to meet you.",
        }
    )

    second_count = counter.count(messages)

    assert first_count > 0
    assert second_count > first_count

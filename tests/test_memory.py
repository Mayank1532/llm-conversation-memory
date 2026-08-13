import pytest

from src.memory.store import MemoryStore


def test_remember_and_recall() -> None:
    memory = MemoryStore()

    memory.remember("name", "Mayank")

    assert memory.recall("name") == "Mayank"


def test_unknown_memory_returns_none() -> None:
    memory = MemoryStore()

    assert memory.recall("unknown") is None


def test_memory_can_be_updated() -> None:
    memory = MemoryStore()

    memory.remember("name", "Mayank")
    memory.remember("name", "Mayank Kumar")

    assert memory.recall("name") == "Mayank Kumar"


def test_get_all_returns_copy() -> None:
    memory = MemoryStore()

    memory.remember("name", "Mayank")

    facts = memory.get_all()
    facts["name"] = "Changed externally"

    assert memory.recall("name") == "Mayank"


def test_empty_key_is_rejected() -> None:
    memory = MemoryStore()

    with pytest.raises(ValueError):
        memory.remember("", "value")


def test_empty_value_is_rejected() -> None:
    memory = MemoryStore()

    with pytest.raises(ValueError):
        memory.remember("key", "")


def test_clear_removes_all_memory() -> None:
    memory = MemoryStore()

    memory.remember("name", "Mayank")
    memory.remember("goal", "Learn generative AI")

    memory.clear()

    assert memory.get_all() == {}

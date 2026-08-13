from src.memory.extractor import MemoryExtractor


def test_extracts_name() -> None:
    extractor = MemoryExtractor()

    assert extractor.extract("My name is Mayank.") == {
        "name": "Mayank"
    }


def test_extracts_learning_topic() -> None:
    extractor = MemoryExtractor()

    assert extractor.extract("I am learning generative AI.") == {
        "learning": "generative AI"
    }


def test_extracts_goal() -> None:
    extractor = MemoryExtractor()

    assert extractor.extract(
        "My goal is to become a data scientist."
    ) == {
        "goal": "to become a data scientist"
    }


def test_ignores_unrelated_message() -> None:
    extractor = MemoryExtractor()

    assert extractor.extract(
        "What is the weather today?"
    ) == {}


def test_empty_message_returns_empty_memory() -> None:
    extractor = MemoryExtractor()

    assert extractor.extract("   ") == {}


def test_extraction_is_case_insensitive() -> None:
    extractor = MemoryExtractor()

    assert extractor.extract("MY NAME IS MAYANK.") == {
        "name": "MAYANK"
    }

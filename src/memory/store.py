class MemoryStore:
    """Store simple persistent facts for one conversation."""

    def __init__(self) -> None:
        self._facts: dict[str, str] = {}

    def remember(self, key: str, value: str) -> None:
        key = key.strip()
        value = value.strip()

        if not key:
            raise ValueError("Memory key cannot be empty.")

        if not value:
            raise ValueError("Memory value cannot be empty.")

        self._facts[key] = value

    def recall(self, key: str) -> str | None:
        return self._facts.get(key.strip())

    def get_all(self) -> dict[str, str]:
        return dict(self._facts)

    def clear(self) -> None:
        self._facts.clear()

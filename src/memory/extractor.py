import re


class MemoryExtractor:
    """Extract explicitly stated facts from user messages."""

    _PATTERNS = [
        (
            "name",
            re.compile(
                r"^\s*my name is\s+(.+?)\s*[.!?]?\s*$",
                re.IGNORECASE,
            ),
        ),
        (
            "learning",
            re.compile(
                r"^\s*i am learning\s+(.+?)\s*[.!?]?\s*$",
                re.IGNORECASE,
            ),
        ),
        (
            "goal",
            re.compile(
                r"^\s*my goal is\s+(.+?)\s*[.!?]?\s*$",
                re.IGNORECASE,
            ),
        ),
    ]

    def extract(self, message: str) -> dict[str, str]:
        """Return explicitly stated facts found in a message."""

        if not message.strip():
            return {}

        for key, pattern in self._PATTERNS:
            match = pattern.match(message)

            if match:
                value = match.group(1).strip()

                if value:
                    return {key: value}

        return {}

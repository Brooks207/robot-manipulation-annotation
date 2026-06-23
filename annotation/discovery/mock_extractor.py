"""Explicit mock discovery extractor for pipeline wiring tests."""

from __future__ import annotations

import logging

from .base import ObjectDiscoverer

logger = logging.getLogger(__name__)


class MockExtractor(ObjectDiscoverer):
    """Deterministic local mock extraction without HTTP requests."""

    def __init__(self) -> None:
        pass

    def discover_objects(self, instruction: str, config: dict) -> list[str]:
        """Return deterministic mock queries without making HTTP requests."""
        logger.debug("Using mock extraction for: %s", instruction)

        instruction_lower = instruction.lower()
        mock_objects = set()

        object_keywords = [
            "cup", "block", "drawer", "spoon", "fork", "knife",
            "plate", "bowl", "bottle", "can", "box", "toy",
            "red", "blue", "green", "yellow", "black", "white",
        ]

        words = instruction_lower.split()
        for i, word in enumerate(words):
            word_clean = word.strip(".,!?")
            if word_clean in object_keywords:
                if i > 0 and words[i - 1].strip(".,!?") in [
                    "red", "blue", "green", "yellow", "black", "white"
                ]:
                    mock_objects.add(f"{words[i - 1].strip('.,!?')} {word_clean}")
                else:
                    mock_objects.add(word_clean)

        always_include = config.get("always_include", [])
        mock_objects.update(item.lower() for item in always_include)

        result = sorted(mock_objects)
        logger.info("MockExtractor: extracted %d objects", len(result))
        return result

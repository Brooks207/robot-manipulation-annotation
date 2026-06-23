"""Rule-based object extractor for discovery layer."""

import logging
import re

from .base import ObjectDiscoverer

logger = logging.getLogger(__name__)


class RuleBasedExtractor(ObjectDiscoverer):
    """
    Extract objects from task instructions using regex and POS tagging rules.

    Fallback extractor when Qwen is not available.
    """

    # Common action verbs in manipulation tasks
    ACTION_VERBS = {
        "pick", "place", "grasp", "put", "move", "push", "pull",
        "open", "close", "stack", "retrieve", "insert", "remove",
        "lift", "drop", "slide", "rotate", "turn", "flip",
    }

    # Common object descriptors
    COLOR_WORDS = {
        "red", "blue", "green", "yellow", "black", "white",
        "orange", "purple", "pink", "brown", "gray", "grey",
    }

    def discover_objects(self, instruction: str, config: dict) -> list[str]:
        """
        Extract objects from instruction using rules.

        Args:
            instruction: Task instruction string
            config: Discovery config dict (uses always_include)

        Returns:
            Deduplicated list of object query strings
        """
        instruction_lower = instruction.lower()
        objects = set()

        # Pattern 1: "the <adj>* <noun>"
        pattern_the = r"\bthe\s+((?:\w+\s+)*\w+)\b"
        for match in re.finditer(pattern_the, instruction_lower):
            phrase = match.group(1).strip()
            # Filter out pure action verbs
            words = phrase.split()
            if words[-1] not in self.ACTION_VERBS:
                objects.add(phrase)

        # Pattern 2: "<color> <noun>"
        for color in self.COLOR_WORDS:
            pattern_color = rf"\b{color}\s+(\w+)\b"
            for match in re.finditer(pattern_color, instruction_lower):
                noun = match.group(1)
                if noun not in self.ACTION_VERBS:
                    objects.add(f"{color} {noun}")

        # Pattern 3: Common manipulation objects (heuristic list)
        common_objects = {
            "cup", "block", "drawer", "spoon", "fork", "knife",
            "plate", "bowl", "bottle", "can", "box", "toy",
            "ball", "cube", "cylinder", "handle", "button",
        }
        for obj in common_objects:
            if re.search(rf"\b{obj}s?\b", instruction_lower):
                objects.add(obj)

        # Add always_include items
        always_include = config.get("always_include", [])
        objects.update(item.lower() for item in always_include)

        # Normalize and deduplicate
        normalized = self._normalize_queries(list(objects))

        logger.info(
            f"RuleBasedExtractor: extracted {len(normalized)} objects from instruction"
        )
        logger.debug(f"  Instruction: {instruction}")
        logger.debug(f"  Objects: {normalized}")

        return normalized

    def _normalize_queries(self, queries: list[str]) -> list[str]:
        """
        Normalize and deduplicate queries.

        Args:
            queries: List of raw object queries

        Returns:
            Normalized, deduplicated list
        """
        normalized = set()
        for q in queries:
            # Strip whitespace, remove articles
            q = q.strip()
            q = re.sub(r"\b(a|an|the)\b\s*", "", q)
            q = " ".join(q.split())  # Collapse multiple spaces
            if q:
                normalized.add(q)

        return sorted(normalized)  # Sort for determinism

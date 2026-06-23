"""Base interface for object discovery layer."""

from abc import ABC, abstractmethod


class ObjectDiscoverer(ABC):
    """Abstract base for discovering which objects to segment in an episode."""

    @abstractmethod
    def discover_objects(self, instruction: str, config: dict) -> list[str]:
        """
        Discover which objects should be segmented for this episode.

        Args:
            instruction: Task instruction string (e.g., "pick up the red cup")
            config: Discovery configuration dict

        Returns:
            List of object query strings (e.g., ["red cup", "robot hand"])
        """
        pass

"""Manual/oracle object extractor for targeted quality checks."""

from .base import ObjectDiscoverer


class ManualExtractor(ObjectDiscoverer):
    """Return curated query lists for selected episodes."""

    def discover_objects(self, instruction: str, config: dict) -> list[str]:
        episode_idx = config.get("episode_idx")
        manual_queries = config.get("manual_queries", {})
        if episode_idx is not None:
            if episode_idx in manual_queries:
                return list(manual_queries[episode_idx])
            if str(episode_idx) in manual_queries:
                return list(manual_queries[str(episode_idx)])
        if "default" in manual_queries:
            return list(manual_queries["default"])
        raise ValueError(
            "manual discovery requires manual_queries keyed by episode_idx or 'default'"
        )

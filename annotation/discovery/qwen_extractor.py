"""Qwen HTTP client object extractor for discovery layer."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import requests

from .base import ObjectDiscoverer

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an object extractor for robot operation datasets.
Given one task instruction, output the concrete objects that should be visually segmented.
Rules:
- Output only short noun phrases for specific visible objects, such as "green bottle" or "blue basket".
- Do not output full sentences, verbs, actions, or robot commands.
- Do not output generic labels such as "object", "item", "thing", or "toy".
- Include color or attributes only when they distinguish the object.
- Output exactly one JSON string array and no extra text."""

FEW_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": "Pick the green bottle from the table and place it inside the blue basket.",
    },
    {"role": "assistant", "content": '["green bottle", "blue basket"]'},
    {
        "role": "user",
        "content": "Open the drawer, retrieve the spoon, then put the spoon into the white bowl.",
    },
    {"role": "assistant", "content": '["drawer", "spoon", "white bowl"]'},
    {
        "role": "user",
        "content": (
            "Move the red toy drill away from the yellow box, pick up the small screw, "
            "and drop it into the metal tray."
        ),
    },
    {
        "role": "assistant",
        "content": '["red toy drill", "yellow box", "small screw", "metal tray"]',
    },
]

GENERIC_OBJECT_NAMES = {
    "object", "objects", "item", "items", "thing", "things", "toy", "toys"
}


class QwenExtractor(ObjectDiscoverer):
    """
    Extract objects from task instructions using an existing OpenAI-compatible
    Qwen chat/completions service.
    """

    def __init__(self):
        """Initialize Qwen extractor."""
        self._cache: dict[str, list[str]] = {}

    def discover_objects(self, instruction: str, config: dict) -> list[str]:
        """
        Extract objects from instruction using a Qwen HTTP service.

        Args:
            instruction: Task instruction string
            config: Discovery config dict

        Returns:
            Deduplicated list of object query strings. Shared always_include
            handling is applied by NormalizingDiscoverer in factory.py.
        """
        cache_key = " ".join(str(instruction or "").split())
        if cache_key in self._cache:
            logger.debug("QwenExtractor: cache hit for instruction")
            return list(self._cache[cache_key])

        if not cache_key:
            logger.warning("QwenExtractor: empty instruction; using fallback queries")
            self._cache[cache_key] = []
            return []

        endpoint = str(config.get("qwen_endpoint") or "").strip()
        model = str(config.get("qwen_model") or "").strip()
        if not endpoint or not model:
            logger.warning(
                "QwenExtractor: qwen_endpoint or qwen_model missing; using fallback queries"
            )
            self._cache[cache_key] = []
            return []

        try:
            response_text = self._request_completion(cache_key, config)
            objects = self._parse_response(response_text)
        except Exception as exc:
            logger.warning("QwenExtractor: request/parse failed: %s", exc)
            objects = []

        self._cache[cache_key] = objects
        logger.info("QwenExtractor: extracted %d objects", len(objects))
        logger.debug("  Instruction: %s", instruction)
        logger.debug("  Objects: %s", objects)
        return list(objects)

    def _request_completion(self, instruction: str, config: dict) -> str:
        """Call the configured OpenAI-compatible chat/completions endpoint."""
        headers = {"Content-Type": "application/json"}
        api_key = config.get("qwen_api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": config["qwen_model"],
            "messages": self._build_messages(instruction),
            "temperature": float(config.get("qwen_temperature", 0.0)),
            "max_tokens": int(config.get("qwen_max_tokens", 128)),
        }

        response = self._post_with_retry(
            config["qwen_endpoint"],
            headers=headers,
            payload=payload,
            timeout=float(config.get("qwen_timeout", 30.0)),
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices") or []
        if not choices:
            raise ValueError("missing choices in Qwen response")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            content = choices[0].get("text")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("missing assistant content in Qwen response")
        return content

    def _post_with_retry(
        self,
        endpoint: str,
        headers: dict[str, str],
        payload: dict,
        timeout: float,
    ) -> requests.Response:
        """Post with one retry for transient network/timeout failures."""
        retry_errors = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
            except retry_errors as exc:
                last_error = exc
                if attempt == 0:
                    logger.warning("QwenExtractor: HTTP request failed, retrying: %s", exc)
                    time.sleep(1.0)
        raise last_error if last_error is not None else RuntimeError("Qwen request failed")

    def _build_messages(self, instruction: str) -> list[dict[str, str]]:
        """Build OpenAI chat messages for object extraction."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *FEW_SHOT_MESSAGES,
            {"role": "user", "content": instruction},
        ]

    def _parse_response(self, response: str) -> list[str]:
        """
        Parse a Qwen JSON-array response with tolerance for markdown fences and
        surrounding text.
        """
        text = self._strip_markdown_fence(response).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[[\s\S]*\]", text)
            if not match:
                raise ValueError(f"Qwen response is not a JSON array: {response!r}")
            parsed = json.loads(match.group(0))

        if not isinstance(parsed, list):
            raise ValueError("Qwen response JSON must be an array")

        return self._normalize_queries(parsed)

    def _strip_markdown_fence(self, text: str) -> str:
        """Remove a single surrounding markdown code fence if present."""
        stripped = text.strip()
        fence_match = re.fullmatch(
            r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE
        )
        if fence_match:
            return fence_match.group(1)
        return stripped

    def _normalize_queries(self, queries: list[Any]) -> list[str]:
        """Normalize, filter generic labels, and deduplicate query strings."""
        normalized = set()
        for query in queries:
            if not isinstance(query, str):
                continue
            value = " ".join(query.strip().lower().split())
            value = re.sub(r"^(?:a|an|the)\s+", "", value)
            if value and value not in GENERIC_OBJECT_NAMES:
                normalized.add(value)
        return sorted(normalized)

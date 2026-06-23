import logging

import pytest
import requests

from annotation.discovery.factory import create_discoverer
from annotation.discovery.qwen_extractor import QwenExtractor


class FakeResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


def qwen_config(**overrides):
    config = {
        "extractor": "qwen",
        "always_include": ["robot hand", "gripper"],
        "qwen_endpoint": "http://qwen.test/v1/chat/completions",
        "qwen_model": "qwen3-test",
        "qwen_temperature": 0.1,
        "qwen_max_tokens": 64,
        "qwen_timeout": 3.0,
        "qwen_api_key": "test-key",
    }
    config.update(overrides)
    return config


def test_parse_response_strips_fences_dedupes_and_filters_generics():
    extractor = QwenExtractor()

    objects = extractor._parse_response(
        """```json
        ["Green Bottle", " green bottle ", "object", "THE blue basket", 123]
        ```"""
    )

    assert objects == ["blue basket", "green bottle"]


def test_qwen_discover_posts_config_and_uses_instruction_cache(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse('["green bottle", "blue basket", "green bottle"]')

    monkeypatch.setattr(requests, "post", fake_post)
    discoverer = create_discoverer(qwen_config())

    first = discoverer.discover_objects(
        "Pick the green bottle and place it in the blue basket.", qwen_config()
    )
    second = discoverer.discover_objects(
        "Pick the green bottle and place it in the blue basket.", qwen_config()
    )

    assert first == ["blue basket", "green bottle", "gripper", "robot hand"]
    assert second == first
    assert len(calls) == 1
    assert calls[0]["url"] == "http://qwen.test/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert calls[0]["json"]["model"] == "qwen3-test"
    assert calls[0]["json"]["temperature"] == 0.1
    assert calls[0]["json"]["max_tokens"] == 64
    assert calls[0]["timeout"] == 3.0
    assert calls[0]["json"]["messages"][-1]["content"].startswith("Pick the green bottle")


def test_qwen_failure_falls_back_to_always_include(monkeypatch, caplog):
    def fake_post(url, headers, json, timeout):
        raise requests.Timeout("service unavailable")

    monkeypatch.setattr(requests, "post", fake_post)
    discoverer = create_discoverer(qwen_config())

    with caplog.at_level(logging.WARNING):
        objects = discoverer.discover_objects("Pick the green bottle.", qwen_config())

    assert objects == ["gripper", "robot hand"]
    assert "request/parse failed" in caplog.text


def test_qwen_bad_json_falls_back_to_always_include(monkeypatch, caplog):
    def fake_post(url, headers, json, timeout):
        return FakeResponse("not a json array")

    monkeypatch.setattr(requests, "post", fake_post)
    discoverer = create_discoverer(qwen_config())

    with caplog.at_level(logging.WARNING):
        objects = discoverer.discover_objects("Pick the green bottle.", qwen_config())

    assert objects == ["gripper", "robot hand"]
    assert "request/parse failed" in caplog.text


def test_mock_extractor_still_uses_local_mock_path(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("mock extractor must not make HTTP requests")

    monkeypatch.setattr(requests, "post", fail_if_called)
    discoverer = create_discoverer(
        {"extractor": "mock", "always_include": ["robot hand", "gripper"]}
    )

    objects = discoverer.discover_objects(
        "Pick up the red cup.", {"always_include": ["robot hand", "gripper"]}
    )

    assert "red cup" in objects
    assert "robot hand" in objects
    assert "gripper" in objects

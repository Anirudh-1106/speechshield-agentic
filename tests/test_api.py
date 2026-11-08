"""Tests for the moderation HTTP API (auth, limits, validation)."""

import pytest
from fastapi.testclient import TestClient

from config.settings import ServerSettings
from core.metrics_server import create_metrics_app


class FakeGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def moderate(self, text: str, speaker: str = "api") -> dict:
        self.calls.append((text, speaker))
        return {"id": "api_1", "text": text, "toxic": "idiot" in text, "rewritten": text.replace("idiot", "person")}

    def snapshot(self) -> dict:
        return {"pass_counter": 3, "log": []}


def client(**cfg) -> tuple[TestClient, FakeGateway]:
    gw = FakeGateway()
    return TestClient(create_metrics_app(gw, ServerSettings(**cfg))), gw


def test_moderate_happy_path() -> None:
    c, gw = client()
    r = c.post("/v1/moderate", json={"text": "you idiot", "speaker": "user42"})
    assert r.status_code == 200
    assert r.json()["toxic"] is True
    assert gw.calls == [("you idiot", "user42")]


def test_moderate_requires_text() -> None:
    c, _ = client()
    assert c.post("/v1/moderate", json={"text": ""}).status_code == 422


def test_moderate_rejects_long_text() -> None:
    c, _ = client(max_text_length=10)
    assert c.post("/v1/moderate", json={"text": "x" * 11}).status_code == 413


@pytest.mark.parametrize("header,status", [(None, 401), ("Bearer wrong", 401), ("Bearer s3cret", 200)])
def test_api_key(header, status) -> None:
    c, _ = client(api_key="s3cret")
    headers = {"Authorization": header} if header else {}
    assert c.post("/v1/moderate", json={"text": "hi"}, headers=headers).status_code == status


def test_health_endpoints_do_not_need_key() -> None:
    c, _ = client(api_key="s3cret")
    assert c.get("/health").status_code == 200
    assert c.get("/metrics").status_code == 200


def test_rate_limit() -> None:
    c, _ = client(rate_limit_per_minute=3)
    codes = [c.post("/v1/moderate", json={"text": "hi"}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]


def test_state_endpoint() -> None:
    c, _ = client()
    body = c.get("/v1/state").json()
    assert body["pass_counter"] == 3
    assert "health" in body


def test_without_pipeline_returns_503() -> None:
    c = TestClient(create_metrics_app())
    assert c.post("/v1/moderate", json={"text": "hi"}).status_code == 503

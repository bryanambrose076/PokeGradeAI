import requests
from fastapi.testclient import TestClient

import main
import pipeline


def test_scan_returns_match_when_pricing_fails(monkeypatch):
    def pricing_down(*args, **kwargs):
        raise requests.ConnectionError("pricing API down")

    monkeypatch.setattr(pipeline, "get_card_prices", pricing_down)
    client = TestClient(main.app)

    with open("tests/fixtures/mew_ex_gold.jpg", "rb") as f:
        response = client.post("/scan", files={"file": ("photo.jpg", f, "image/jpeg")})

    assert response.status_code == 200
    data = response.json()
    assert data["card"]["name"] == "Mew ex"
    assert data["matches"][0]["key"] == "sv03.5/205.png"
    assert data["prices"] is None
    assert data["profit"] is None

import os

import pytest
from fastapi.testclient import TestClient

from pokegrade import db, pipeline
from pokegrade.price_cache import get_cached, save

pytestmark = pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")


def age_row(card_id, hours):
    db.run(
        "UPDATE price_cache SET fetched_at = now() - make_interval(hours => %s) WHERE card_id = %s",
        (hours, card_id),
    )


def test_miss_then_hit():
    assert get_cached("sv06.5/082") == (False, None)
    save("sv06.5/082", (2.28, 15.38, 82))
    assert get_cached("sv06.5/082") == (True, (2.28, 15.38, 82))


def test_priced_row_expires_after_24h():
    save("sv06.5/082", (2.28, 15.38, 82))
    age_row("sv06.5/082", 25)
    assert get_cached("sv06.5/082") == (False, None)


def test_no_price_found_expires_after_6h():
    save("svp/176", None)
    age_row("svp/176", 5)
    assert get_cached("svp/176") == (True, None)
    age_row("svp/176", 7)
    assert get_cached("svp/176") == (False, None)


# Nothing listening on port 1: should miss quietly, not crash
def test_database_down(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/x")
    monkeypatch.setattr(db, "_conn", None)
    assert get_cached("sv06.5/082") == (False, None)
    save("sv06.5/082", (2.28, 15.38, 82))


# Real model + fake pricing API that counts calls
def test_second_scan_uses_cache(monkeypatch):
    from pokegrade import main

    calls = []

    def fake_prices(name, set_name, card_number, api_key):
        calls.append((name, set_name, card_number))
        return (500.0, 600.0, 900.0)

    monkeypatch.setattr(pipeline, "get_card_prices", fake_prices)
    client = TestClient(main.app)

    with open("tests/fixtures/mew_ex_gold.jpg", "rb") as f:
        photo = f.read()
    first = client.post("/scan", files={"file": ("photo.jpg", photo, "image/jpeg")}).json()
    second = client.post("/scan", files={"file": ("photo.jpg", photo, "image/jpeg")}).json()

    assert len(calls) == 1
    assert first["prices"] == second["prices"] == {"raw": 500.0, "psa9": 600.0, "psa10": 900.0}

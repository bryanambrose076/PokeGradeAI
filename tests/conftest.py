import os

import pytest

import price_cache


# Point the cache at an empty test database (or turn it off) so tests never touch Neon
@pytest.fixture(autouse=True)
def test_database(monkeypatch):
    monkeypatch.setattr(price_cache, "_conn", None)
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        monkeypatch.delenv("DATABASE_URL", raising=False)
        return
    monkeypatch.setenv("DATABASE_URL", url)
    with open("schema.sql") as f:
        price_cache._run(f.read(), ())
    price_cache._run("TRUNCATE price_cache", ())

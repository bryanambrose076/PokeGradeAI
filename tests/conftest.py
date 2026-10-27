import os

import pytest

from pokegrade import db


# Point the app at an empty test database (or turn it off) so tests never touch Neon
@pytest.fixture(autouse=True)
def test_database(monkeypatch):
    monkeypatch.setattr(db, "_conn", None)
    monkeypatch.delenv("SCANS_BUCKET", raising=False)
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        monkeypatch.delenv("DATABASE_URL", raising=False)
        return
    monkeypatch.setenv("DATABASE_URL", url)
    with open("schema.sql") as f:
        db.run(f.read())
    db.run("TRUNCATE price_cache, scans, corrections, labels")

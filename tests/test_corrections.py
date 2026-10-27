import os
import uuid

import pytest
from fastapi.testclient import TestClient

from pokegrade import db, main
from pokegrade.search import search_cards

needs_db = pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
client = TestClient(main.app)


# A bare scans row to report on
def make_scan():
    scan_id = str(uuid.uuid4())
    db.run("INSERT INTO scans (id, top_matches, low_confidence) VALUES (%s, '[]', false)", (scan_id,))
    return scan_id


def report(scan_id, query, card_id=None):
    return client.post("/corrections", json={"scan_id": scan_id, "query": query, "card_id": card_id})


def saved(scan_id):
    return db.run("SELECT query, card_id, status FROM corrections WHERE scan_id = %s", (scan_id,))


def test_search_finds_card():
    assert search_cards("mew 205")[0]["card_id"] == "sv03.5/205"


def test_search_ignores_leading_zeros():
    assert search_cards("umbreon 59")[0]["card_id"] == "sv08.5/059"


def test_search_whole_words_only():
    assert all(card["name"] != "Mewtwo" for card in search_cards("mew"))


def test_search_empty():
    assert client.get("/search").json() == []


@needs_db
def test_report_saved():
    scan_id = make_scan()
    assert report(scan_id, "gold mew", "sv03.5/205").status_code == 200
    assert saved(scan_id) == ("gold mew", "sv03.5/205", "unverified")


@needs_db
def test_none_of_these():
    scan_id = make_scan()
    assert report(scan_id, "umbreon promo").status_code == 200
    assert saved(scan_id) == ("umbreon promo", None, "unverified")


@needs_db
def test_resubmit_replaces_unless_reviewed():
    scan_id = make_scan()
    report(scan_id, "mew", "sv03.5/151")
    report(scan_id, "mew 205", "sv03.5/205")
    assert saved(scan_id) == ("mew 205", "sv03.5/205", "unverified")

    db.run("UPDATE corrections SET status = 'approved' WHERE scan_id = %s", (scan_id,))
    report(scan_id, "umbreon", "sv03/130")
    assert saved(scan_id) == ("mew 205", "sv03.5/205", "approved")


@needs_db
def test_old_scan_rejected():
    scan_id = make_scan()
    db.run("UPDATE scans SET created_at = now() - make_interval(hours => 25) WHERE id = %s", (scan_id,))
    response = report(scan_id, "mew 205", "sv03.5/205")
    assert response.status_code == 400
    assert response.json()["detail"] == "This scan is too old to report"
    assert saved(scan_id) is None


@needs_db
@pytest.mark.parametrize("scan_id, query, card_id, error", [
    (None, "mew", "fake/1", "Unknown card"),
    ("new", "mew", None, "Unknown scan"),
    ("abc", "mew", None, "Unknown scan"),
    (None, "x" * 101, None, "Search text must be 1-100 characters"),
    (None, "   ", None, "Search text must be 1-100 characters"),
])
def test_bad_reports_rejected(scan_id, query, card_id, error):
    if scan_id is None:
        scan_id = make_scan()
    elif scan_id == "new":
        scan_id = str(uuid.uuid4())
    response = report(scan_id, query, card_id)
    assert response.status_code == 400
    assert response.json()["detail"] == error

import os

import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

from pokegrade import db, main, pipeline, scan_history

pytestmark = pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")


# Records uploads instead of sending them to S3
class FakeS3:
    def __init__(self, fail=False):
        self.fail = fail
        self.keys = []

    def put_object(self, Bucket, Key, Body, ContentType):
        if self.fail:
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")
        self.keys.append(Key)


@pytest.fixture
def s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setenv("SCANS_BUCKET", "test-bucket")
    monkeypatch.setattr(scan_history, "_s3", fake)
    return fake


@pytest.fixture(autouse=True)
def fake_pricing(monkeypatch):
    monkeypatch.setattr(pipeline, "get_card_prices", lambda *args: (500.0, 600.0, 900.0))


def scan_mew():
    with open("tests/fixtures/mew_ex_gold.jpg", "rb") as f:
        response = TestClient(main.app).post("/scan", files={"file": ("photo.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    return response.json()


def test_scan_is_saved(s3):
    data = scan_mew()
    scan_id = data["scan_id"]
    assert scan_id is not None

    row = db.run(
        "SELECT card_id, jsonb_array_length(top_matches), gap, low_confidence, duration_ms, photo_key, crop_key FROM scans WHERE id = %s",
        (scan_id,),
    )
    card_id, n_matches, gap, low_confidence, duration_ms, photo_key, crop_key = row
    assert card_id == "sv03.5/205"
    assert n_matches == 5
    assert gap > 0
    assert low_confidence == data["low_confidence"]
    assert duration_ms > 0
    assert photo_key.startswith("scans/") and photo_key.endswith(f"{scan_id}.jpg")
    assert crop_key.endswith(f"{scan_id}_crop.jpg")
    assert s3.keys == [photo_key, crop_key]


def test_s3_down_still_saves_row(s3):
    s3.fail = True
    data = scan_mew()
    assert data["prices"] is not None
    assert db.run("SELECT photo_key, crop_key FROM scans WHERE id = %s", (data["scan_id"],)) == (None, None)


# Nothing listening on port 1
def test_database_down(s3, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/x")
    monkeypatch.setattr(db, "_conn", None)
    data = scan_mew()
    assert data["card"]["name"] == "Mew ex"
    assert data["scan_id"] is None


def test_saving_off_without_bucket():
    data = scan_mew()
    assert data["scan_id"] is None
    assert db.run("SELECT count(*) FROM scans") == (0,)

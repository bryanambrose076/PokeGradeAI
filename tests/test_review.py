import os
import uuid

import pytest
from botocore.exceptions import NoCredentialsError
from fastapi.testclient import TestClient

from pokegrade import db, review
from pokegrade.corrections import save_correction
from scripts.review import app

pytestmark = pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")


# Records copies instead of sending them to S3
class FakeS3:
    def __init__(self):
        self.copies = []

    def copy_object(self, Bucket, Key, CopySource):
        self.copies.append((CopySource["Key"], Key))

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        return f"https://example.com/{Params['Key']}"


@pytest.fixture(autouse=True)
def s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setenv("SCANS_BUCKET", "test-bucket")
    monkeypatch.setattr(review, "_s3", fake)
    return fake


# A saved scan with a report on it; returns the report's ID
def make_report(query, user_card, app_card="sv03.5/205", crop=True):
    scan_id = str(uuid.uuid4())
    db.run(
        "INSERT INTO scans (id, photo_key, crop_key, card_id, top_matches, gap, low_confidence) "
        "VALUES (%s, %s, %s, %s, '[]', 0.01, true)",
        (scan_id, f"scans/d/{scan_id}.jpg", f"scans/d/{scan_id}_crop.jpg" if crop else None, app_card),
    )
    assert save_correction(scan_id, query, user_card) is None
    return db.run("SELECT id FROM corrections WHERE scan_id = %s", (scan_id,))[0]


def label(report_id):
    return db.run(
        "SELECT l.card_id, l.split, l.photo_key, l.crop_key FROM labels l "
        "JOIN corrections c ON c.scan_id = l.scan_id WHERE c.id = %s",
        (report_id,),
    )


def status(report_id):
    return db.run("SELECT status FROM corrections WHERE id = %s", (report_id,))[0]


def test_list_pending_oldest_first():
    first = make_report("mew 151", "sv03.5/151")
    second = make_report("kakuna", "sv03.5/014", app_card="sv03.5/014")
    rows = review.list_pending()
    assert [r[0] for r in rows] == [first, second]
    assert rows[0][2:5] == ("mew 151", "sv03.5/151", "sv03.5/205")


def test_approve_saves_label_and_copies_photos(s3):
    report_id = make_report("mew 151", "sv03.5/151")
    assert review.approve(report_id) is None
    card_id, split, photo_key, crop_key = label(report_id)
    assert card_id == "sv03.5/151"
    assert split in ("tune", "holdout")
    assert photo_key.startswith("labels/") and crop_key.endswith("_crop.jpg")
    assert [new for _, new in s3.copies] == [photo_key, crop_key]
    assert status(report_id) == "approved"
    assert review.list_pending() == []


def test_approve_without_crop(s3):
    report_id = make_report("mew 151", "sv03.5/151", crop=False)
    assert review.approve(report_id) is None
    assert label(report_id)[3] is None
    assert len(s3.copies) == 1


def test_split_holdout(monkeypatch):
    monkeypatch.setattr(review.random, "random", lambda: 0.1)
    report_id = make_report("mew 151", "sv03.5/151")
    review.approve(report_id)
    assert label(report_id)[1] == "holdout"


def test_approve_as_other_card():
    report_id = make_report("umbreon promo", None)
    assert review.approve(report_id) == "Pick a real card to approve as"
    assert review.approve(report_id, "fake/1") == "Pick a real card to approve as"
    assert status(report_id) == "unverified"
    assert review.approve(report_id, "svp/176") is None
    assert label(report_id)[0] == "svp/176"


def test_approve_twice_refused():
    report_id = make_report("mew 151", "sv03.5/151")
    review.approve(report_id)
    assert review.approve(report_id) == "No unverified report with that ID"


def test_reject():
    report_id = make_report("kakuna", "sv03.5/014")
    review.reject(report_id)
    assert status(report_id) == "rejected"
    assert label(report_id) is None
    assert review.approve(report_id) == "No unverified report with that ID"


def test_page_escapes_user_text():
    make_report("<script>alert(1)</script>", "sv03.5/151")
    page = TestClient(app).get("/").text
    assert "Reports waiting for review (1)" in page
    assert "<script>" not in page and "&lt;script&gt;" in page


def test_page_aws_login_expired(monkeypatch):
    def expired(*args, **kwargs):
        raise NoCredentialsError()

    monkeypatch.setattr(review, "photo_url", expired)
    make_report("mew 151", "sv03.5/151")
    response = TestClient(app).get("/")
    assert response.status_code == 503
    assert "aws login" in response.text


def test_page_approve_and_errors():
    client = TestClient(app)
    report_id = make_report("umbreon promo", None)
    response = client.post(f"/approve/{report_id}")
    assert "Pick a real card to approve as" in response.text
    client.post(f"/approve/{report_id}", data={"card_id": " svp/176 "})
    assert label(report_id)[0] == "svp/176"
    assert "Nothing to review." in client.get("/").text

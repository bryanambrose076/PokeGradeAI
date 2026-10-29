import csv
import io
import os
import uuid

import pytest
from psycopg.types.json import Jsonb

from pokegrade import db, review
from scripts.export_labels import export

pytestmark = pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")


# Returns each key's name as the photo bytes
class FakeS3:
    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(Key.encode())}


def label(card_id, split):
    scan_id = str(uuid.uuid4())
    db.run("INSERT INTO scans (id, card_id, top_matches, gap, low_confidence) VALUES (%s, 'x/1', %s, 0.1, false)",
           (scan_id, Jsonb([])))
    db.run("INSERT INTO labels (scan_id, card_id, split, photo_key) VALUES (%s, %s, %s, %s)",
           (scan_id, card_id, split, f"labels/{scan_id}.jpg"))
    return scan_id


def test_exports_tune_only(tmp_path, monkeypatch):
    monkeypatch.setenv("SCANS_BUCKET", "test-bucket")
    monkeypatch.setattr(review, "_s3", FakeS3())
    tune = label("sv03.5/205", "tune")
    label("sv09/173", "holdout")
    (tmp_path / "stale.jpg").write_text("old")

    assert export(tmp_path) == 1
    with open(tmp_path / "labels.csv") as f:
        assert list(csv.DictReader(f)) == [{"filename": f"{tune}.jpg", "card_id": "sv03.5/205"}]
    assert (tmp_path / f"{tune}.jpg").read_text() == f"labels/{tune}.jpg"
    assert not (tmp_path / "stale.jpg").exists()

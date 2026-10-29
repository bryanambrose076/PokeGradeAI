import asyncio
import io
import os
import uuid

import cv2
import numpy as np
import pytest
from botocore.exceptions import LoginRefreshRequired
from mcp.server.mcpserver.exceptions import ToolError
from psycopg.types.json import Jsonb

from pokegrade import db, mcp_server, review

needs_db = pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")

MEW = "sv03.5/205"
UMBREON = "sv08.5/059"
NO_IMAGE = "svp/176"


# Serves a blank 3000x2000 JPEG for any key and records which keys were asked for
class FakeS3:
    def __init__(self):
        self.keys = []

    def get_object(self, Bucket, Key):
        self.keys.append(Key)
        jpeg = cv2.imencode(".jpg", np.zeros((2000, 3000, 3), np.uint8))[1].tobytes()
        return {"Body": io.BytesIO(jpeg)}


@pytest.fixture
def s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setenv("SCANS_BUCKET", "test-bucket")
    monkeypatch.setattr(review, "_s3", fake)
    return fake


# A scans row where the app guessed `guess`, with `others` filling the rest of the top 5
def make_scan(guess, others=(), low_confidence=False):
    scan_id = str(uuid.uuid4())
    matches = [{"card_id": c, "score": 0.9 - i * 0.1} for i, c in enumerate([guess, *others])]
    db.run(
        "INSERT INTO scans (id, card_id, top_matches, gap, low_confidence) VALUES (%s, %s, %s, %s, %s)",
        (scan_id, guess, Jsonb(matches), 0.1, low_confidence),
    )
    return scan_id


def report(scan_id, card_id, status="unverified"):
    db.run("INSERT INTO corrections (scan_id, query, card_id, status) VALUES (%s, 'test', %s, %s)",
           (scan_id, card_id, status))


def label(scan_id, card_id, split="tune"):
    db.run("INSERT INTO labels (scan_id, card_id, split) VALUES (%s, %s, %s)", (scan_id, card_id, split))


def test_every_tool_has_a_description():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    assert {t.name for t in tools} == {
        "list_unsure_scans", "get_scan", "view_crop", "view_reference", "search_cards",
        "list_pending_reports", "list_label_misses", "log_suggestion",
    }
    assert all(len(t.description) > 100 for t in tools)


def test_search_cards():
    assert mcp_server.search_cards("mew 205")[0]["card_id"] == MEW


@needs_db
def test_list_unsure_scans():
    make_scan(MEW)
    unsure = make_scan(MEW, low_confidence=True)
    report(make_scan(MEW, low_confidence=True), UMBREON)

    scans = mcp_server.list_unsure_scans()
    assert [s["scan_id"] for s in scans] == [unsure]
    assert scans[0]["guess"]["name"].startswith("Mew")


@needs_db
def test_get_scan():
    scan_id = make_scan(MEW, [UMBREON])
    report(scan_id, UMBREON, "approved")
    label(scan_id, UMBREON)

    scan = mcp_server.get_scan(scan_id)
    assert [m["card_id"] for m in scan["top_matches"]] == [MEW, UMBREON]
    assert scan["report"]["card"]["card_id"] == UMBREON
    assert scan["label"] == {"card": mcp_server._card(UMBREON), "split": "tune"}


@needs_db
@pytest.mark.parametrize("scan_id", ["abc", str(uuid.uuid4())])
def test_get_scan_unknown(scan_id):
    with pytest.raises(ToolError):
        mcp_server.get_scan(scan_id)


@needs_db
def test_list_pending_reports():
    pending = make_scan(MEW)
    report(pending, None)
    report(make_scan(MEW), UMBREON, "rejected")

    reports = mcp_server.list_pending_reports()
    assert [r["scan_id"] for r in reports] == [pending]
    assert reports[0]["user_pick"] is None


@needs_db
def test_list_label_misses_tune_only():
    right = make_scan(MEW)
    label(right, MEW)
    near_miss = make_scan(MEW, [UMBREON])
    label(near_miss, UMBREON)
    total_miss = make_scan(MEW)
    label(total_miss, UMBREON)
    label(make_scan(MEW), UMBREON, "holdout")

    misses = {m["scan_id"]: m["correct_rank"] for m in mcp_server.list_label_misses()}
    assert misses == {near_miss: 2, total_miss: None}


# Reference images are gitignored and not in the Docker image, so this only runs where they've been downloaded
@pytest.mark.skipif(not (mcp_server.IMAGES_DIR / f"{MEW}.png").exists(), reason="reference images not downloaded")
def test_view_reference():
    image = mcp_server.view_reference(MEW)
    assert image.data.startswith(b"\x89PNG")


@pytest.mark.parametrize("card_id", ["fake/1", "../../.env", NO_IMAGE])
def test_view_reference_refused(card_id):
    with pytest.raises(ToolError):
        mcp_server.view_reference(card_id)


# A scan with a photo and crop in scans/
def make_photo_scan():
    scan_id = make_scan(MEW)
    db.run("UPDATE scans SET photo_key = %s, crop_key = %s WHERE id = %s",
           (f"scans/d/{scan_id}.jpg", f"scans/d/{scan_id}_crop.jpg", scan_id))
    return scan_id


@needs_db
def test_view_crop_shrinks_photo(s3):
    scan_id = make_photo_scan()
    image = mcp_server.view_crop(scan_id, full_photo=True)
    assert s3.keys == [f"scans/d/{scan_id}.jpg"]
    height, width = cv2.imdecode(np.frombuffer(image.data, np.uint8), cv2.IMREAD_COLOR).shape[:2]
    assert (width, height) == (1568, 1045)


@needs_db
def test_view_crop_prefers_label_copy(s3):
    scan_id = make_photo_scan()
    db.run("INSERT INTO labels (scan_id, card_id, split, crop_key) VALUES (%s, %s, 'tune', %s)",
           (scan_id, MEW, f"labels/{scan_id}_crop.jpg"))
    mcp_server.view_crop(scan_id)
    assert s3.keys == [f"labels/{scan_id}_crop.jpg"]


@needs_db
def test_view_crop_no_photo(s3):
    with pytest.raises(ToolError, match="no saved crop"):
        mcp_server.view_crop(make_scan(MEW))


@needs_db
def test_image_tools_return_pictures(s3):
    scan_id = make_photo_scan()
    for name, args in [("view_crop", {"scan_id": scan_id}), ("view_reference", {"card_id": MEW})]:
        result = asyncio.run(mcp_server.mcp.call_tool(name, args))
        assert not result.is_error
        assert result.content[0].type == "image"


# The agent should see why S3 failed, not just "Error executing tool view_crop"
@needs_db
def test_view_crop_aws_login_expired(s3, monkeypatch):
    def expired(**kwargs):
        raise LoginRefreshRequired()
    monkeypatch.setattr(s3, "get_object", expired)
    with pytest.raises(ToolError, match="aws login"):
        asyncio.run(mcp_server.mcp.call_tool("view_crop", {"scan_id": make_photo_scan()}))


def suggestions():
    return db.run_all("SELECT kind, card_id, reason FROM suggestions ORDER BY id")


@needs_db
def test_log_suggestion_replaces_within_run():
    scan_id = make_scan(MEW)
    first = mcp_server.log_suggestion(scan_id, "code_fix", " crop includes the toploader ")
    second = mcp_server.log_suggestion(scan_id, "approve", "report is right", UMBREON)
    assert first == second
    assert suggestions() == [("approve", UMBREON, "report is right")]


@needs_db
@pytest.mark.parametrize("kind, reason, card_id, error", [
    ("fix", "x", None, "kind must be"),
    ("reject", "   ", None, "reason must be"),
    ("reject", "x" * 501, None, "reason must be"),
    ("approve", "x", None, "needs a card_id"),
    ("eval_photo", "x", None, "needs a card_id"),
    ("reject", "x", "fake/1", "Not a real card"),
])
def test_log_suggestion_refused(kind, reason, card_id, error):
    with pytest.raises(ToolError, match=error):
        mcp_server.log_suggestion(make_scan(MEW), kind, reason, card_id)
    assert suggestions() == []


@needs_db
def test_log_suggestion_unknown_scan():
    with pytest.raises(ToolError, match="No scan"):
        mcp_server.log_suggestion(str(uuid.uuid4()), "reject", "spam")

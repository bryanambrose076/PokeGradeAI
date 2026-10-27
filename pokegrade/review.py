"""Review actions for "Wrong card?" reports: list pending ones, approve into labels, or reject."""
import os
import random

import boto3

from pokegrade import db
from pokegrade.search import is_real_card

HOLDOUT_SHARE = 0.2
_s3 = None


# Connect to S3 the first time it's needed, then reuse the connection
def _client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


# Get every report that still needs a decision, oldest first, with the app's guess and photo locations
def list_pending():
    return db.run_all(
        """
        SELECT c.id, c.created_at, c.query, c.card_id, s.card_id, s.gap, s.photo_key, s.crop_key
        FROM corrections c JOIN scans s ON s.id = c.scan_id
        WHERE c.status = 'unverified'
        ORDER BY c.created_at
        """
    )


# Create a link that works for an hour, so the review page can show a private photo
def photo_url(key):
    return _client().generate_presigned_url(
        "get_object", Params={"Bucket": os.environ["SCANS_BUCKET"], "Key": key}, ExpiresIn=3600
    )


# Copy a photo into labels/ so it's kept after the 90-day cleanup
def _keep(key, scan_id, suffix):
    if key is None:
        return None
    new_key = f"labels/{scan_id}{suffix}"
    bucket = os.environ["SCANS_BUCKET"]
    _client().copy_object(Bucket=bucket, Key=new_key, CopySource={"Bucket": bucket, "Key": key})
    return new_key


# Approve a report and save it as a label, using the user's card unless a different one is given
def approve(correction_id, card_id=None):
    row = db.run(
        "SELECT c.scan_id, c.card_id, s.photo_key, s.crop_key FROM corrections c "
        "JOIN scans s ON s.id = c.scan_id WHERE c.id = %s AND c.status = 'unverified'",
        (correction_id,),
    )
    if row is None:
        return "No unverified report with that ID"
    scan_id, user_card, photo_key, crop_key = row
    card_id = card_id or user_card
    if card_id is None or not is_real_card(card_id):
        return "Pick a real card to approve as"

    split = "holdout" if random.random() < HOLDOUT_SHARE else "tune"
    db.run(
        "INSERT INTO labels (scan_id, card_id, split, photo_key, crop_key) VALUES (%s, %s, %s, %s, %s)",
        (scan_id, card_id, split, _keep(photo_key, scan_id, ".jpg"), _keep(crop_key, scan_id, "_crop.jpg")),
    )
    db.run("UPDATE corrections SET status = 'approved' WHERE id = %s", (correction_id,))
    return None


# Reject a report; its photo is deleted with the rest after 90 days
def reject(correction_id):
    db.run("UPDATE corrections SET status = 'rejected' WHERE id = %s AND status = 'unverified'", (correction_id,))

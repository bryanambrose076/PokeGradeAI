import os
import uuid
from datetime import datetime, timezone

import boto3
import cv2
import psycopg
from botocore.exceptions import BotoCoreError, ClientError
from psycopg.types.json import Jsonb

import db

_s3 = None


def _upload(key, data):
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    _s3.put_object(Bucket=os.environ["SCANS_BUCKET"], Key=key, Body=data, ContentType="image/jpeg")


def save_scan(photo_bytes, crop, result, duration_ms):
    if not os.getenv("SCANS_BUCKET") or not os.getenv("DATABASE_URL"):
        return None
    scan_id = str(uuid.uuid4())
    prefix = f"scans/{datetime.now(timezone.utc):%Y-%m-%d}/{scan_id}"

    # Photos first; if an upload fails, the row is still saved without it
    photo_key, crop_key = f"{prefix}.jpg", f"{prefix}_crop.jpg"
    try:
        _upload(photo_key, photo_bytes)
    except (BotoCoreError, ClientError):
        photo_key = None
    try:
        _upload(crop_key, cv2.imencode(".jpg", crop)[1].tobytes())
    except (BotoCoreError, ClientError):
        crop_key = None

    matches = [{"card_id": m["key"].removesuffix(".png"), "score": float(m["score"])} for m in result["matches"]]
    card_id = matches[0]["card_id"] if result["card"] is not None else None
    gap = matches[0]["score"] - matches[1]["score"] if len(matches) > 1 else None

    try:
        db.run(
            """
            INSERT INTO scans (id, photo_key, crop_key, card_id, top_matches, gap, low_confidence, duration_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (scan_id, photo_key, crop_key, card_id, Jsonb(matches), gap, result["low_confidence"], duration_ms),
        )
    except psycopg.Error:
        return None
    return scan_id

"""MCP server for the review agent: read tools over scans, reports, labels, and cards, plus log_suggestion.

Run with: python -m pokegrade.mcp_server
"""
import os
import uuid

import cv2
import numpy as np
from botocore.exceptions import BotoCoreError
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from pokegrade import db, review, search
from pokegrade.pricing import lookup_local_card

IMAGES_DIR = search.SETS_DIR.parent / "images"
MAX_IMAGE_SIDE = 1568
MAX_REASON_LENGTH = 500
KINDS = ("approve", "reject", "eval_photo", "code_fix")
# Each time the server starts counts as one review run
RUN_ID = str(uuid.uuid4())

# Tools raise ToolError, not ValueError: only ToolError messages reach the agent,
# anything else shows up as a bare "Error executing tool X"
mcp = MCPServer("pokegrade")

# Turn a card ID into {card_id, name}, or None if there's no card
def _card(card_id):
    if card_id is None:
        return None
    info = lookup_local_card(card_id)
    return {"card_id": card_id, "name": f"{info[0]} ({info[1]} {info[2]})" if info else "unknown"}


# Find where a card landed in a scan's top 5 (1 = top guess), or None if it wasn't there
def _rank(top_matches, card_id):
    for i, match in enumerate(top_matches):
        if match["card_id"] == card_id:
            return i + 1
    return None


# Refuse anything that isn't shaped like a scan ID before it reaches the database
def _check_scan_id(scan_id):
    try:
        uuid.UUID(scan_id)
    except ValueError:
        raise ToolError(f"Not a scan ID: {scan_id}")


# Download a photo from S3 and shrink it so its longest side is at most MAX_IMAGE_SIDE
def _download_jpeg(key):
    if not os.getenv("SCANS_BUCKET"):
        raise ToolError("SCANS_BUCKET isn't set; add it to the server's env in .mcp.json")
    try:
        data = review._client().get_object(Bucket=os.environ["SCANS_BUCKET"], Key=key)["Body"].read()
    except BotoCoreError as e:
        raise ToolError(f"AWS error: {e} The owner needs to run `aws login`; this isn't a problem with the scan.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    scale = MAX_IMAGE_SIDE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img)[1].tobytes()


@mcp.tool()
def list_unsure_scans(days: int = 7, limit: int = 50) -> list[dict]:
    """List low-confidence scans nobody reported, newest first. The app wasn't sure
    of its guess but the user didn't flag it, so the guess may be wrong without
    anyone noticing. Call get_scan for the full details of one. Reported scans are
    in list_pending_reports instead.

    days: how far back to look (1-90). limit: max results (1-50).

    Each result has scan_id, created_at, guess (the app's top match as card_id and
    name, or null if no card was found), and gap (top match score minus second match
    score; under 0.03 the app marks the scan low confidence). Returns [] if there
    are none.
    """
    days = min(max(days, 1), 90)
    limit = min(max(limit, 1), 50)
    rows = db.run_all(
        """
        SELECT s.id, s.created_at, s.card_id, s.gap
        FROM scans s LEFT JOIN corrections c ON c.scan_id = s.id
        WHERE s.created_at > now() - make_interval(days => %s)
          AND s.low_confidence AND c.id IS NULL
        ORDER BY s.created_at DESC
        LIMIT %s
        """,
        (days, limit),
    )
    return [
        {"scan_id": str(r[0]), "created_at": r[1].isoformat(), "guess": _card(r[2]), "gap": r[3]}
        for r in rows
    ]


@mcp.tool()
def get_scan(scan_id: str) -> dict:
    """Get everything about one scan by its scan_id.

    Returns scan_id, created_at, guess (the app's top match, or null), gap,
    low_confidence, top_matches (up to 5 cards, best first, each with card_id, name,
    and score; higher score means more similar), report (the user's "Wrong card?"
    report with query, their picked card or null for "none of these", and status
    unverified/approved/rejected; null if not reported), and label (the confirmed
    card and its split tune/holdout; null if not approved). Fails if the scan_id
    doesn't exist.
    """
    _check_scan_id(scan_id)
    scan = db.run(
        "SELECT created_at, card_id, gap, low_confidence, top_matches FROM scans WHERE id = %s", (scan_id,)
    )
    if scan is None:
        raise ToolError(f"No scan with ID {scan_id}")
    report = db.run("SELECT query, card_id, status FROM corrections WHERE scan_id = %s", (scan_id,))
    label = db.run("SELECT card_id, split FROM labels WHERE scan_id = %s", (scan_id,))
    return {
        "scan_id": scan_id,
        "created_at": scan[0].isoformat(),
        "guess": _card(scan[1]),
        "gap": scan[2],
        "low_confidence": scan[3],
        "top_matches": [{**_card(m["card_id"]), "score": m["score"]} for m in scan[4]],
        "report": {"query": report[0], "card": _card(report[1]), "status": report[2]} if report else None,
        "label": {"card": _card(label[0]), "split": label[1]} if label else None,
    }


@mcp.tool()
def view_crop(scan_id: str, full_photo: bool = False) -> Image:
    """Show the card crop from a scan: the part of the user's photo the app cut out
    and compared against reference images. Look at it next to view_reference of the
    app's guess and of the right card to see why matching went wrong (glare, blur,
    a cut-off corner, the wrong rectangle picked, or a same-art variant).

    full_photo: show the whole uploaded photo instead of the crop, e.g. to check
    whether the crop missed part of the card or grabbed a sleeve or toploader.

    Returns a JPEG, at most 1568 pixels on its longest side. Fails if the scan_id
    doesn't exist or the scan has no saved photo (the upload failed or it was
    deleted after 90 days).
    """
    _check_scan_id(scan_id)
    row = db.run(
        "SELECT s.photo_key, s.crop_key, l.photo_key, l.crop_key "
        "FROM scans s LEFT JOIN labels l ON l.scan_id = s.id WHERE s.id = %s",
        (scan_id,),
    )
    if row is None:
        raise ToolError(f"No scan with ID {scan_id}")
    # Approved scans have a kept copy in labels/, which outlives the 90-day cleanup
    key = (row[2] or row[0]) if full_photo else (row[3] or row[1])
    if key is None:
        raise ToolError(f"Scan {scan_id} has no saved {'photo' if full_photo else 'crop'}")
    return Image(data=_download_jpeg(key), format="jpeg")


@mcp.tool()
def view_reference(card_id: str) -> Image:
    """Show a card's official reference image, the one the matcher compares crops
    against. card_id looks like "sv03.5/205" (use search_cards to find one).

    Returns a PNG. Fails if the card_id isn't a supported card.
    """
    path = IMAGES_DIR / f"{card_id}.png"
    if not search.is_real_card(card_id) or not path.exists():
        raise ToolError(f"Not a supported card: {card_id}")
    return Image(data=path.read_bytes(), format="png")


@mcp.tool()
def search_cards(query: str) -> list[dict]:
    """Search all 4,619 cards by name, set name, and number, e.g. "mew 151 205".

    Matches whole words only ("mew" doesn't match "Mewtwo"); numbers ignore leading
    zeros ("52" matches "052"). Use it to turn a user's description into a card_id.
    Returns up to 10 cards, most matching words first, each with card_id, name,
    set_name, and number. Returns [] if nothing matches.
    """
    return search.search_cards(query[:100])


@mcp.tool()
def list_pending_reports() -> list[dict]:
    """List "Wrong card?" reports that haven't been approved or rejected yet, oldest first.

    Each result has report_id, scan_id (pass to get_scan for the scores),
    created_at, query (what the user typed), user_pick (the card they chose, or null
    if they picked "none of these"), guess (the app's top match, or null), and gap.
    Returns [] if there are none.
    """
    rows = db.run_all(
        """
        SELECT c.id, c.scan_id, c.created_at, c.query, c.card_id, s.card_id, s.gap
        FROM corrections c JOIN scans s ON s.id = c.scan_id
        WHERE c.status = 'unverified'
        ORDER BY c.created_at
        """
    )
    return [
        {"report_id": r[0], "scan_id": str(r[1]), "created_at": r[2].isoformat(), "query": r[3],
         "user_pick": _card(r[4]), "guess": _card(r[5]), "gap": r[6]}
        for r in rows
    ]


@mcp.tool()
def list_label_misses() -> list[dict]:
    """List confirmed scans where the app's top guess was wrong: the failures worth studying.

    Only covers the tune split; holdout labels are kept hidden so they stay a fair
    test of any fix. Each result has scan_id, label (the correct card), guess (what
    the app picked, or null), gap, and correct_rank (where the correct card landed in
    the app's top 5, or null if it wasn't there at all). Returns [] if there are none.
    """
    rows = db.run_all(
        """
        SELECT l.scan_id, l.card_id, s.card_id, s.gap, s.top_matches
        FROM labels l JOIN scans s ON s.id = l.scan_id
        WHERE l.split = 'tune' AND s.card_id IS DISTINCT FROM l.card_id
        ORDER BY l.created_at
        """
    )
    return [
        {"scan_id": str(r[0]), "label": _card(r[1]), "guess": _card(r[2]), "gap": r[3],
         "correct_rank": _rank(r[4], r[1])}
        for r in rows
    ]


@mcp.tool()
def log_suggestion(scan_id: str, kind: str, reason: str, card_id: str | None = None) -> dict:
    """Save your conclusion about one scan for the owner to act on. This is the only
    tool that writes anything; it never changes reports, labels, or code itself.

    kind is one of:
    - "approve": the user's "Wrong card?" report is right. card_id = the right card.
    - "reject": the report is wrong or spam. card_id is optional.
    - "eval_photo": a clear, useful photo to add to the eval set. card_id = the card it shows.
    - "code_fix": matching failed in a way the code should handle (bad crop, glare,
      same-art variant). card_id is optional.
    reason: what you saw and why you chose this kind, in 1-500 characters, e.g.
    "crop includes the toploader edge; right card is #2 in the top 5".
    card_id looks like "sv03.5/205" (use search_cards to find one).

    Calling it again for the same scan in this run replaces your earlier suggestion.
    Returns {"suggestion_id": ...}. Fails if the scan doesn't exist, the kind is
    unknown, the reason is empty or too long, or card_id is missing or not a real card.
    """
    _check_scan_id(scan_id)
    if kind not in KINDS:
        raise ToolError(f"kind must be one of {', '.join(KINDS)}")
    reason = reason.strip()
    if not 1 <= len(reason) <= MAX_REASON_LENGTH:
        raise ToolError(f"reason must be 1-{MAX_REASON_LENGTH} characters")
    if card_id is None and kind in ("approve", "eval_photo"):
        raise ToolError(f"{kind} needs a card_id")
    if card_id is not None and not search.is_real_card(card_id):
        raise ToolError(f"Not a real card: {card_id}")
    if db.run("SELECT 1 FROM scans WHERE id = %s", (scan_id,)) is None:
        raise ToolError(f"No scan with ID {scan_id}")
    row = db.run(
        """
        INSERT INTO suggestions (run_id, scan_id, kind, card_id, reason) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (run_id, scan_id) DO UPDATE
          SET kind = EXCLUDED.kind, card_id = EXCLUDED.card_id, reason = EXCLUDED.reason, created_at = now()
        RETURNING id
        """,
        (RUN_ID, scan_id, kind, card_id, reason),
    )
    return {"suggestion_id": row[0]}


if __name__ == "__main__":
    # Use the read-only agent role, never the owner login from .env
    if not os.getenv("AGENT_DATABASE_URL"):
        raise SystemExit("AGENT_DATABASE_URL isn't set; add it to .env")
    os.environ["DATABASE_URL"] = os.environ["AGENT_DATABASE_URL"]
    mcp.run()

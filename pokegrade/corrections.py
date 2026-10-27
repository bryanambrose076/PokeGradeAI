"""Saves "Wrong card?" reports from the scanner page into the corrections table."""
from datetime import timedelta

import psycopg

from pokegrade import db
from pokegrade.search import is_real_card

MAX_QUERY_LENGTH = 100
REPORT_WINDOW = timedelta(hours=24)


# Check the report is valid, then save it as unverified; returns an error message, or None if saved
def save_correction(scan_id, query, card_id):
    query = query.strip()
    if not query or len(query) > MAX_QUERY_LENGTH:
        return "Search text must be 1-100 characters"
    if card_id is not None and not is_real_card(card_id):
        return "Unknown card"
    try:
        # Only accept reports on recent scans, so old scan IDs can't be reported later
        row = db.run("SELECT now() - created_at < %s FROM scans WHERE id = %s", (REPORT_WINDOW, scan_id))
        if row is None:
            return "Unknown scan"
        if not row[0]:
            return "This scan is too old to report"
        db.run(
            """
            INSERT INTO corrections (scan_id, query, card_id)
            VALUES (%s, %s, %s)
            ON CONFLICT (scan_id) DO UPDATE SET
              query = EXCLUDED.query,
              card_id = EXCLUDED.card_id,
              created_at = now()
            WHERE corrections.status = 'unverified'
            """,
            (scan_id, query, card_id),
        )
    except (psycopg.errors.ForeignKeyViolation, psycopg.errors.InvalidTextRepresentation):
        return "Unknown scan"
    return None

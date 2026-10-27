"""Remember each card's prices in Postgres for 24h."""
import os
from datetime import timedelta

import psycopg

from pokegrade import db

# How long a saved price counts as fresh; "no price found" expires sooner in case the card gets listed
FRESH_FOR = timedelta(hours=24)
NOT_FOUND_FRESH_FOR = timedelta(hours=6)


# Return (hit, prices); a miss if there's no row, it's too old, or the database is down
def get_cached(card_id):
    if not os.getenv("DATABASE_URL"):
        return False, None
    try:
        row = db.run(
            "SELECT found, raw, psa9, psa10, now() - fetched_at FROM price_cache WHERE card_id = %s",
            (card_id,),
        )
    except psycopg.Error:
        return False, None
    if row is None:
        return False, None

    found, raw, psa9, psa10, age = row
    if age >= (FRESH_FOR if found else NOT_FOUND_FRESH_FOR):
        return False, None
    return True, ((raw, psa9, psa10) if found else None)


# Save or overwrite this card's prices; database errors are ignored
def save(card_id, prices):
    if not os.getenv("DATABASE_URL"):
        return
    raw, psa9, psa10 = prices if prices is not None else (None, None, None)
    try:
        db.run(
            """
            INSERT INTO price_cache (card_id, found, raw, psa9, psa10)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (card_id) DO UPDATE SET
              found = EXCLUDED.found,
              raw = EXCLUDED.raw,
              psa9 = EXCLUDED.psa9,
              psa10 = EXCLUDED.psa10,
              fetched_at = now()
            """,
            (card_id, prices is not None, raw, psa9, psa10),
        )
    except psycopg.Error:
        pass

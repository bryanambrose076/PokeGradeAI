import os
from datetime import timedelta

import psycopg
from dotenv import load_dotenv

load_dotenv()

FRESH_FOR = timedelta(hours=24)
NOT_FOUND_FRESH_FOR = timedelta(hours=6)

_conn = None


def _run(query, params):
    global _conn
    for attempt in range(2):
        try:
            if _conn is None or _conn.closed:
                _conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True, connect_timeout=5)
            cur = _conn.execute(query, params)
            return cur.fetchone() if cur.description else None
        except psycopg.OperationalError:
            _conn = None
            if attempt == 1:
                raise


def get_cached(card_id):
    if not os.getenv("DATABASE_URL"):
        return False, None
    try:
        row = _run(
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


def save(card_id, prices):
    if not os.getenv("DATABASE_URL"):
        return
    raw, psa9, psa10 = prices if prices is not None else (None, None, None)
    try:
        _run(
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

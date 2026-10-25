import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

_conn = None


# One connection per Lambda instance; reconnect once if Neon dropped it
def run(query, params=()):
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

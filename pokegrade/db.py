"""Shared Neon (Postgres) connection."""
import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

_conn = None


# Run one SQL query; keeps one connection open, reconnects once if Neon dropped it
def _execute(query, params):
    global _conn
    for attempt in range(2):
        try:
            if _conn is None or _conn.closed:
                _conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True, connect_timeout=5)
            return _conn.execute(query, params)
        except psycopg.OperationalError:
            _conn = None
            if attempt == 1:
                raise


# Run a query and return its first row (or None)
def run(query, params=()):
    cur = _execute(query, params)
    return cur.fetchone() if cur.description else None


# Run a query and return all its rows
def run_all(query, params=()):
    return _execute(query, params).fetchall()

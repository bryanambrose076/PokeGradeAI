CREATE TABLE IF NOT EXISTS price_cache (
  card_id    text PRIMARY KEY,
  found      boolean NOT NULL,
  raw        double precision,
  psa9       double precision,
  psa10      double precision,
  fetched_at timestamptz NOT NULL DEFAULT now()
);

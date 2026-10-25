-- Records prices per card, so repeat scans skip the pricing API
CREATE TABLE IF NOT EXISTS price_cache (
  card_id    text PRIMARY KEY,
  found      boolean NOT NULL,
  raw        double precision,
  psa9       double precision,
  psa10      double precision,
  fetched_at timestamptz NOT NULL DEFAULT now()
);

-- Logs scan data: what the app guessed, how sure it was, and where the photos are in S3
CREATE TABLE IF NOT EXISTS scans (
  id             uuid PRIMARY KEY,
  created_at     timestamptz NOT NULL DEFAULT now(),
  photo_key      text,
  crop_key       text,
  card_id        text,
  top_matches    jsonb NOT NULL,
  gap            double precision,
  low_confidence boolean NOT NULL,
  duration_ms    integer
);

CREATE INDEX IF NOT EXISTS scans_created_at ON scans (created_at);

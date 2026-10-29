-- Records prices per card, so repeat scans skip the pricing API
CREATE TABLE IF NOT EXISTS price_cache (
  card_id    text PRIMARY KEY,
  found      boolean NOT NULL,
  raw        double precision,
  psa9       double precision,
  psa10      double precision,
  fetched_at timestamptz NOT NULL DEFAULT now()
);

-- Logs scan data: what the app guessed, how sure it was, and stores the photos in S3
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

-- User reports that a scan matched the wrong card; unverified until approved on the review page
CREATE TABLE IF NOT EXISTS corrections (
  id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  scan_id    uuid NOT NULL UNIQUE REFERENCES scans (id),
  query      text NOT NULL,
  card_id    text,
  status     text NOT NULL DEFAULT 'unverified' CHECK (status IN ('unverified', 'approved', 'rejected')),
  created_at timestamptz NOT NULL DEFAULT now()
);

-- Approved reports: real photos with a confirmed card, for measuring and improving matching
CREATE TABLE IF NOT EXISTS labels (
  scan_id    uuid PRIMARY KEY REFERENCES scans (id),
  card_id    text NOT NULL,
  split      text NOT NULL CHECK (split IN ('tune', 'holdout')),
  photo_key  text,
  crop_key   text,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- Review agent output: one suggestion per scan per run, for me to act on
CREATE TABLE IF NOT EXISTS suggestions (
  id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  run_id     uuid NOT NULL,
  scan_id    uuid NOT NULL REFERENCES scans (id),
  kind       text NOT NULL CHECK (kind IN ('approve', 'reject', 'eval_photo', 'code_fix')),
  card_id    text,
  reason     text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (run_id, scan_id)
);

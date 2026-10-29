"""Download the approved tune labels from S3 so the eval can run on real user scans.

Run: python -m scripts.export_labels, then python -m scripts.eval eval/tune/labels.csv --photos eval/tune
Holdout labels are never exported, so they can't end up where tuning happens.
"""
import csv
import os
import shutil

from pokegrade import db, review

OUT_DIR = "eval/tune"


# Download every tune label's photo into OUT_DIR and write labels.csv in the eval's format
def export(out_dir=OUT_DIR):
    rows = db.run_all("SELECT scan_id, card_id, photo_key FROM labels "
                      "WHERE split = 'tune' AND photo_key IS NOT NULL ORDER BY created_at")
    # Start clean so a label that was removed or moved doesn't linger
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, "labels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "card_id"])
        for scan_id, card_id, photo_key in rows:
            filename = f"{scan_id}.jpg"
            body = review._client().get_object(Bucket=os.environ["SCANS_BUCKET"], Key=photo_key)["Body"]
            with open(os.path.join(out_dir, filename), "wb") as photo:
                photo.write(body.read())
            writer.writerow([filename, card_id])
    return len(rows)


if __name__ == "__main__":
    # Read-only login (db.py has already loaded .env): this script only needs to look things up
    os.environ["DATABASE_URL"] = os.environ["AGENT_DATABASE_URL"]
    os.environ.setdefault("SCANS_BUCKET", "pokegrade-scans-017421876144-us-east-2-an")
    print(f"Exported {export()} tune labels to {OUT_DIR}/")

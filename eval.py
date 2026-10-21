import csv
import os
import sys

import cv2

from cosineSimilarity import load_matcher
from pipeline import match_photo

# Runs every labeled photo through the same crop + match steps as /scan (no pricing calls)
# and reports top-1 / top-5 accuracy and the #1-#2 score gap for each photo.

LABELS_PATH = "eval/labels.csv"
PHOTOS_DIR = "eval/photos"
CROPS_DIR = "eval/crops"  # cropped cards are saved here so misses can be checked by eye


def load_labels(path=LABELS_PATH):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_eval(labels, matcher, top_k=5):
    os.makedirs(CROPS_DIR, exist_ok=True)
    results = []
    for row in labels:
        img = cv2.imread(os.path.join(PHOTOS_DIR, row["filename"]))
        if img is None:
            print(f"Could not read {row['filename']}, skipping")
            continue

        matches, cropped = match_photo(img, matcher, top_k=top_k)
        cv2.imwrite(os.path.join(CROPS_DIR, row["filename"]), cropped)

        # embedding keys look like "sv03.5/205.png"; labels look like "sv03.5/205"
        predicted = [key.removesuffix(".png") for key, _ in matches]

        results.append({
            "filename": row["filename"],
            "expected": row["card_id"],
            "predicted": predicted,
            "scores": [score for _, score in matches],
            "gap": matches[0][1] - matches[1][1],
            "rank": predicted.index(row["card_id"]) + 1 if row["card_id"] in predicted else None,
        })
    return results


def print_report(results):
    print(f"{'file':<14} {'expected':<13} {'got':<13} {'score':>6} {'gap':>6}  result")
    for r in results:
        if r["rank"] == 1:
            outcome = "ok"
        elif r["rank"] is not None:
            outcome = f"MISS (correct at #{r['rank']})"
        else:
            outcome = "MISS (not in top 5)"
        print(f"{r['filename']:<14} {r['expected']:<13} {r['predicted'][0]:<13} "
              f"{r['scores'][0]:>6.3f} {r['gap']:>6.3f}  {outcome}")

    total = len(results)
    top1 = sum(1 for r in results if r["rank"] == 1)
    top5 = sum(1 for r in results if r["rank"] is not None)
    print()
    print(f"top-1 accuracy: {top1}/{total} ({top1 / total:.1%})")
    print(f"top-5 accuracy: {top5}/{total} ({top5 / total:.1%})")

    correct_gaps = sorted(r["gap"] for r in results if r["rank"] == 1)
    wrong_gaps = sorted(r["gap"] for r in results if r["rank"] != 1)
    if correct_gaps:
        print(f"gap when correct: min {correct_gaps[0]:.3f}, median {correct_gaps[len(correct_gaps) // 2]:.3f}")
    if wrong_gaps:
        print(f"gap when wrong:   max {wrong_gaps[-1]:.3f}")


if __name__ == "__main__":
    labels_path = sys.argv[1] if len(sys.argv) > 1 else LABELS_PATH
    results = run_eval(load_labels(labels_path), load_matcher())
    print_report(results)

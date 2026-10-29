"""For every miss in one or more eval.py --json files, save one image: crop | right card | app's guess.

Run from the repo root: python .claude/skills/fix-matcher/compare.py /tmp/eval.json [/tmp/tune.json ...]
Images go to eval/crops/compare/. The crops come from eval/crops/, which the eval writes on every run.
"""
import json
import os
import sys

import cv2
import numpy as np

CROPS_DIR = "eval/crops"
IMAGES_DIR = "data/images"
OUT_DIR = "eval/crops/compare"
W, H, BAND = 500, 700, 60


# Load an image at the panel size, or a grey panel saying why it's missing
def panel(path):
    img = cv2.imread(path) if os.path.exists(path) else None
    if img is None:
        img = np.full((H, W, 3), 128, np.uint8)
        cv2.putText(img, "no image", (150, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return cv2.resize(img, (W, H))


# Put a white band with a caption above a panel
def caption(img, text):
    band = np.full((BAND, W, 3), 255, np.uint8)
    cv2.putText(band, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return np.vstack([band, img])


def compare(photo):
    rank = f"#{photo['rank']}" if photo["rank"] else "not in top 5"
    guess = photo["predicted"][0]
    panels = [
        caption(panel(os.path.join(CROPS_DIR, photo["filename"])), f"crop (gap {photo['gap']:.3f})"),
        caption(panel(os.path.join(IMAGES_DIR, f"{photo['expected']}.png")), f"right {photo['expected']} ({rank})"),
        caption(panel(os.path.join(IMAGES_DIR, f"{guess}.png")), f"guess {guess}"),
    ]
    path = os.path.join(OUT_DIR, photo["filename"])
    cv2.imwrite(path, np.hstack(panels))
    return path


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for results_path in sys.argv[1:]:
        with open(results_path) as f:
            for photo in json.load(f)["photos"]:
                if photo["rank"] != 1:
                    print(compare(photo))

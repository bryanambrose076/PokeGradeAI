import os
import sys

import cv2
import numpy as np
import requests

from imageCrop import crop_candidates
from cosineSimilarity import load_matcher, match_candidates
from pricing import lookup_local_card, get_card_prices, grading_roi,  API_KEY
from price_cache import get_cached, save

# If the top match beats the runner-up by less than this, the result is flagged as uncertain.
# Tuned for DINOv2 scores: correct matches on real test photos had gaps of 0.09+ (one at 0.014).
LOW_CONFIDENCE_GAP = 0.03


def match_photo(img, matcher, top_k=5):
    """img: BGR array. Returns (matches, chosen crop). Shared by /scan and eval.py."""
    candidates = crop_candidates(img)
    matches, chosen = match_candidates(candidates, matcher, top_k=top_k)
    return matches, candidates[chosen]


def scan_image(image_bytes, matcher, top_k=5):
    # 1. Decode the raw bytes into a BGR image array (same as cv2.imread gives)
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")

    # 2-3. Crop the card several ways and keep whichever crop matches a card best
    matches, _ = match_photo(img, matcher, top_k=top_k)

    result = {
        "matches": [{"key": key, "score": score} for key, score in matches],
        "card": None,
        "prices": None,
        "profit": None,
        "low_confidence": False,
        "runner_up": None,
    }

    # 4. Look up the top match's name, set, and number
    local_card_info = lookup_local_card(matches[0][0])
    if local_card_info is None:
        return result
    name, set_name, card_number = local_card_info
    result["card"] = {"name": name, "set_name": set_name, "number": card_number}

    # If the top two matches are too close to call, include the runner-up so the user can check
    if matches[0][1] - matches[1][1] < LOW_CONFIDENCE_GAP:
        result["low_confidence"] = True
        runner_up_info = lookup_local_card(matches[1][0])
        if runner_up_info is not None:
            ru_name, ru_set_name, ru_number = runner_up_info
            result["runner_up"] = {"name": ru_name, "set_name": ru_set_name, "number": ru_number}

    # 5. Fetch raw, PSA 9, and PSA 10 prices for that card (cached for a day)
    card_id = matches[0][0].removesuffix(".png")
    hit, prices = get_cached(card_id)
    if not hit:
        try:
            prices = get_card_prices(name, set_name, card_number, API_KEY)
            save(card_id, prices)
        except requests.RequestException:
            prices = None
    if prices is not None:
        raw_price, psa9_price, psa10_price = prices
        result["prices"] = {"raw": raw_price, "psa9": psa9_price, "psa10": psa10_price}
        result["profit"] = grading_roi(raw_price, psa10_price)

    # 6. Return everything as one dictionary
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = os.path.join(os.path.expanduser("~"), "Desktop", "testimage2.jpg")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    result = scan_image(image_bytes, load_matcher())

    print(f"Top {len(result['matches'])} matches for {image_path}:")
    for match in result["matches"]:
        print(f"  {match['key']:<20} {match['score']:.4f}")

    if result["card"] is not None:
        card = result["card"]
        print(f"Price data for {card['name']} ({card['set_name']} {card['number']}):")
        if result["prices"] is not None:
            print(f"  Raw price: {result['prices']['raw']}$")
            print(f"  PSA 9 price: {result['prices']['psa9']}$")
            print(f"  PSA 10 price: {result['prices']['psa10']}$")
            print(f"  Potential Profit: {result['profit']}$")

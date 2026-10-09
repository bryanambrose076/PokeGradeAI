import os
import sys

import cv2
import numpy as np

from imageCrop import crop_card
from cosineSimilarity import load_matcher, match_card
from pricing import lookup_local_card, get_card_prices, API_KEY


def scan_image(image_bytes, matcher, top_k=5):
    # 1. Decode the raw bytes into a BGR image array (same as cv2.imread gives)
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")

    # 2. Crop the card out of the photo
    cropped = crop_card(img)

    # 3. Match the cropped card against the embeddings
    matches = match_card(cropped, matcher, top_k=top_k)

    result = {
        "matches": [{"key": key, "score": score} for key, score in matches],
        "card": None,
        "prices": None,
    }

    # 4. Look up the top match's name, set, and number
    local_card_info = lookup_local_card(matches[0][0])
    if local_card_info is None:
        return result
    name, set_name, card_number = local_card_info
    result["card"] = {"name": name, "set_name": set_name, "number": card_number}

    # 5. Fetch raw and PSA 10 prices for that card
    prices = get_card_prices(name, set_name, card_number, API_KEY)
    if prices is not None:
        raw_price, psa10_price = prices
        result["prices"] = {"raw": raw_price, "psa10": psa10_price}

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
            print(f"  Raw price: {result['prices']['raw']}")
            print(f"  PSA 10 price: {result['prices']['psa10']}")

"""PokemonPriceTracker API lookups and grading profit."""
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("POKEMONPRICETRACKER_API_KEY")

# Parse JSON files to identify the card's name, set, and number for price lookup
def lookup_local_card(tcgdex_key):
    set_id, local_id = tcgdex_key.split("/")
    local_id = local_id.removesuffix(".png")
    set_path = f"data/sets/{set_id}.json"
    with open(set_path) as f:
        cards = json.load(f)

    match = None
    for card in cards:
        if card["localId"] == local_id:
            match = card
            break

    if match is None:
        print(f"no local card found for {tcgdex_key}")
        return None

    return (match["name"], match["set_name"], match["localId"])


def get_card_prices(name, set_name, card_number, api_key):
    # the requests.get() call to PokemonPriceTracker
    # returns raw + graded prices
    response = requests.get(
        "https://www.pokemonpricetracker.com/api/v2/cards",
        headers={
          "Authorization": f"Bearer {api_key}"
        },
        params={
          "language": "english",
          "set": set_name,
          "search": f"{name} {card_number}",
          "includeEbay": "true",
          "limit": "1"
        },
        timeout=10
    )
    response.raise_for_status()
    data = response.json()

    if not data["data"]:
        print(f"no price data found for {name} {set_name} {card_number}")
        return None

    card = data["data"][0]
    raw_price = card.get("prices", {}).get("market")
    psa9_price = card.get("ebay", {}).get("salesByGrade", {}).get("psa9", {}).get("smartMarketPrice", {}).get("price")
    psa10_price = card.get("ebay", {}).get("salesByGrade", {}).get("psa10", {}).get("smartMarketPrice", {}).get("price")

    return (raw_price, psa9_price, psa10_price)

def grading_roi(raw_price, psa10_price):
    if raw_price is None or psa10_price is None:
        return None
    profit = None
    if psa10_price <= 500:
        profit = psa10_price - raw_price - 27
    elif psa10_price <= 1000:
        profit = psa10_price - raw_price - 66
    elif psa10_price <= 1500:
        profit = psa10_price - raw_price - 88
    elif psa10_price <= 2500:
        profit = psa10_price - raw_price - 220
    else:
        profit = psa10_price - raw_price - 380

    return profit

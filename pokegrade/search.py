"""Card search for the "Wrong card?" box: finds cards by what the user types, for the "Did you mean…?" list."""
import json
import re
from pathlib import Path

SETS_DIR = Path(__file__).resolve().parent.parent / "data" / "sets"
_cards = None


# Split text into lowercase words; numbers lose leading zeros so "52" matches "052"
def _words(text):
    words = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        words.add((w.lstrip("0") or "0") if w.isdigit() else w)
    return words


# Load every card once: id, name, set name, number, plus its words for searching
def _load():
    global _cards
    if _cards is None:
        _cards = []
        for path in sorted(SETS_DIR.glob("*.json")):
            for c in json.loads(path.read_text()):
                card = {
                    "card_id": f"{c['set_id']}/{c['localId']}",
                    "name": c["name"],
                    "set_name": c["set_name"],
                    "number": c["localId"],
                }
                _cards.append((card, _words(f"{c['name']} {c['set_name']} {c['localId']}")))
    return _cards


# Score each card by how many typed words match a whole word in its name, set, or number
def search_cards(query, limit=10):
    words = _words(query)
    if not words:
        return []
    results = []
    for card, card_words in _load():
        score = len(words & card_words)
        if score:
            results.append((score, card))
    results.sort(key=lambda r: -r[0])
    return [card for _, card in results[:limit]]


# Check that a card ID exists, so corrections can't use made-up IDs
def is_real_card(card_id):
    return any(card["card_id"] == card_id for card, _ in _load())

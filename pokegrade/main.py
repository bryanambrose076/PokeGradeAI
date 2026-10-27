"""Web server: the scanner page, /scan, /search, /corrections, and /health."""
import time

import numpy as np
from fastapi import FastAPI, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pokegrade.corrections import MAX_QUERY_LENGTH, save_correction
from pokegrade.matcher import load_matcher, match_candidates
from pokegrade.pipeline import scan_image
from pokegrade.scan_history import save_scan
from pokegrade.search import search_cards

app = FastAPI()


matcher = load_matcher()

# Run one match at startup so the slow first pass over the model weights
# happens here (on Lambda: during startup or a keep-warm ping), not in someone's scan
match_candidates([np.zeros((700, 500, 3), np.uint8)], matcher)


# Quick "is the app up?" check, used by the deploy step
@app.get("/health")
def health():
    return {"status": "ok"}


# Scan an uploaded photo, save it to history, and return the match and prices
@app.post("/scan")
def scan(file: bytes = File()):
    start = time.perf_counter()
    try:
        result, crop = scan_image(file, matcher)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    duration_ms = round((time.perf_counter() - start) * 1000)

    # Save the scan before responding; Lambda freezes once the response is sent
    result["scan_id"] = save_scan(file, crop, result, duration_ms)
    return result


# Search cards for the "Did you mean…?" list
@app.get("/search")
def search(q: str = ""):
    return search_cards(q[:MAX_QUERY_LENGTH])


class Correction(BaseModel):
    scan_id: str
    query: str
    card_id: str | None = None


# Save a "Wrong card?" report
@app.post("/corrections")
def correct(body: Correction):
    error = save_correction(body.scan_id, body.query, body.card_id)
    if error:
        raise HTTPException(status_code=400, detail=error)
    return {"ok": True}


# Serve the scanner web page
@app.get("/")
def scanner_page():
    return FileResponse("frontend/scanner.html")
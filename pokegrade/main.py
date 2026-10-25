"""Web server: the scanner page, /scan, and /health."""
import time

import numpy as np
from fastapi import FastAPI, File, HTTPException
from fastapi.responses import FileResponse

from pokegrade.matcher import load_matcher, match_candidates
from pokegrade.pipeline import scan_image
from pokegrade.scan_history import save_scan

app = FastAPI()


matcher = load_matcher()

# Run one match at startup so the slow first pass over the model weights
# happens here (on Lambda: during startup or a keep-warm ping), not in someone's scan
match_candidates([np.zeros((700, 500, 3), np.uint8)], matcher)


@app.get("/health")
def health():
    return {"status": "ok"}


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


@app.get("/")
def scanner_page():
    return FileResponse("frontend/scanner.html")
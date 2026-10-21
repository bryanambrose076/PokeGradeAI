import numpy as np
from fastapi import FastAPI, File, HTTPException
from fastapi.responses import FileResponse

from cosineSimilarity import load_matcher, match_candidates
from pipeline import scan_image

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
    try:
        return scan_image(file, matcher)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def scanner_page():
    return FileResponse("frontend/scanner.html")
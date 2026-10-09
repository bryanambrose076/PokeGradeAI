from fastapi import FastAPI, File, HTTPException

from cosineSimilarity import load_matcher
from pipeline import scan_image

app = FastAPI()


matcher = load_matcher()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/scan")
def scan(file: bytes = File()):
    try:
        return scan_image(file, matcher)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

import json

from scripts.eval import save_json, summarize


# One photo right at #1, one with the right card at #2, one with it missing from the top 5
def results():
    return [
        {"filename": "a.jpg", "expected": "x/1", "predicted": ["x/1", "x/2"], "scores": [0.9, 0.8],
         "gap": 0.1, "rank": 1, "seconds": 0.4},
        {"filename": "b.jpg", "expected": "x/2", "predicted": ["x/1", "x/2"], "scores": [0.7, 0.69],
         "gap": 0.01, "rank": 2, "seconds": 0.5},
        {"filename": "c.jpg", "expected": "x/3", "predicted": ["x/1", "x/2"], "scores": [0.6, 0.5],
         "gap": 0.1, "rank": None, "seconds": 0.6},
    ]


def test_summarize():
    assert summarize(results()) == {"total": 3, "top1": 1, "top5": 2}


def test_save_json(tmp_path):
    path = tmp_path / "eval.json"
    save_json(results(), path)
    saved = json.loads(path.read_text())
    assert saved["top1"] == 1
    assert [p["rank"] for p in saved["photos"]] == [1, 2, None]

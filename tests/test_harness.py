from scripts.harness import PROTECTED, card_flags, changed_files, gate


def diff(path, added):
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1,2 @@\n x = 1\n+{added}\n"


def test_changed_files_and_protected():
    text = diff("pokegrade/crop.py", "y = 2") + diff("scripts/eval.py", "z = 3")
    assert changed_files(text) == ["pokegrade/crop.py", "scripts/eval.py"]
    assert [f for f in changed_files(text) if f.startswith(PROTECTED)] == ["scripts/eval.py"]


def test_card_flags():
    assert card_flags(diff("pokegrade/matcher.py", 'if top == "sv03.5/205": pass')) == [
        "`sv03.5/205` in pokegrade/matcher.py"]
    # Tests may name real cards, and ordinary code shouldn't be flagged
    assert card_flags(diff("tests/test_crop.py", 'MEW = "sv03.5/205"')) == []
    assert card_flags(diff("pokegrade/crop.py", "scale = width / height")) == []


def evals(top1, ranks):
    photos = [{"filename": f"p{i}", "expected": "x/1", "rank": r, "seconds": 0.4} for i, r in enumerate(ranks)]
    return {"total": len(ranks), "top1": top1, "top5": len(ranks), "photos": photos}


BASELINE = {"eval": evals(1, [1, 2]), "tune": evals(0, [2])}


def test_gate():
    better = {"tests_passed": True, "eval": evals(2, [1, 1]), "tune": evals(1, [1])}
    assert gate(better, BASELINE) == (True, "passes")
    assert gate({**better, "tests_passed": False}, BASELINE) == (False, "tests failed")
    # Same top-1 total, but a photo that was right is now wrong
    swapped = {**better, "eval": evals(1, [2, 1])}
    assert gate(swapped, BASELINE) == (False, "breaks eval photos")
    assert gate({"error": "docker build failed"}, BASELINE) == (False, "docker build failed")

from scripts.compare_evals import compare, print_comparison


def photo(name, rank, seconds=0.4):
    return {"filename": name, "expected": "x/1", "rank": rank, "seconds": seconds}


def run(photos):
    return {"top1": sum(p["rank"] == 1 for p in photos), "top5": sum(p["rank"] is not None for p in photos),
            "photos": photos}


# a stays right, b gets fixed, c breaks, d stays wrong but moves, e and f are only in one run
BEFORE = run([photo("a", 1), photo("b", 2), photo("c", 1), photo("d", None), photo("e", 1)])
AFTER = run([photo("a", 1, 0.6), photo("b", 1, 0.6), photo("c", 3, 0.6), photo("d", 4, 0.6), photo("f", 1, 0.6)])


def test_compare():
    diff = compare(BEFORE, AFTER)
    assert [(p["filename"], old) for p, old in diff["fixed"]] == [("b", 2)]
    assert [(p["filename"], old) for p, old in diff["broken"]] == [("c", 1)]
    assert diff["only_in_one"] == ["e", "f"]
    assert diff["seconds"] == (0.4, 0.6)


def test_print_comparison(capsys):
    print_comparison(compare(BEFORE, AFTER))
    out = capsys.readouterr().out
    assert "b              x/1           #2 -> #1" in out
    assert "c              x/1           #1 -> #3" in out
    assert "+50%" in out

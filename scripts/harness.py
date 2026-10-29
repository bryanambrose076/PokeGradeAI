"""Score candidate fixes against the committed code, each in a fresh Docker container.

Run: python -m scripts.harness RUN_DIR [--holdout DIR]
RUN_DIR/candidates/*.diff are the fixes. For the committed code (the baseline) and each candidate, this builds
the image, runs the tests and the eval on the eval photos and on eval/tune, then writes RUN_DIR/report.md.
"""
import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

from pokegrade.search import _load as load_cards
from scripts.compare_evals import compare, rank_text

# A fix may not touch how it's measured or what gets installed; those candidates are rejected outright
PROTECTED = ("eval/", "scripts/", ".claude/", ".github/", "Dockerfile", ".dockerignore", "requirements", "pytest.ini")
EVAL_SETS = {"eval": ("eval/labels.csv", "eval/photos_2000"), "tune": ("eval/tune/labels.csv", "eval/tune")}
REPO = Path(__file__).resolve().parent.parent


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


# Files a diff changes, from its "+++ b/path" lines
def changed_files(diff_text):
    return [line[6:] for line in diff_text.splitlines() if line.startswith("+++ b/")]


# Added lines in app code that name a real card or a labeled photo: a sign of a card-specific hack
def card_flags(diff_text):
    names = {card["card_id"] for card, _ in load_cards()}
    for labels, _ in EVAL_SETS.values():
        if (REPO / labels).exists():
            names |= {line.split(",")[0] for line in (REPO / labels).read_text().splitlines()[1:]}
    flags, path = [], None
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
        elif line.startswith("+") and path and path.startswith("pokegrade/"):
            hits = [word for word in re.findall(r"[\w.\-]+(?:/\w+)?", line) if word in names]
            flags += [f"`{hit}` in {path}" for hit in hits]
    return flags


# Build one version of the code and run its tests and evals; results land in out_dir
def score(worktree, name, out_dir, holdout=None):
    out_dir.mkdir(parents=True)
    image = f"pokegrade-harness:{name}"
    build = run(["docker", "build", "-q", "-t", image, str(worktree)])
    if build.returncode:
        return {"error": "docker build failed", "log": build.stderr[-2000:]}
    tests = run(["docker", "run", "--rm", "-v", f"{worktree}/tests:/app/tests", "-v", f"{worktree}/scripts:/app/scripts",
                 "-v", f"{worktree}/pytest.ini:/app/pytest.ini", image,
                 "sh", "-c", "pip install -q -r requirements-dev.txt && pytest -q"])
    result = {"tests_passed": tests.returncode == 0, "tests_log": tests.stdout[-1500:]}

    # The eval photos come from this repo, read-only; crops go to a throwaway folder
    sets = dict(EVAL_SETS, **({"holdout": ("/holdout/labels.csv", "/holdout")} if holdout else {}))
    with tempfile.TemporaryDirectory() as crops:
        for set_name, (labels, photos) in sets.items():
            if set_name != "holdout" and not (REPO / labels).exists():
                continue
            mounts = ["-v", f"{REPO / 'eval'}:/app/eval:ro", "-v", f"{crops}:/app/eval/crops",
                      "-v", f"{worktree}/scripts:/app/scripts", "-v", f"{out_dir}:/out"]
            if set_name == "holdout":
                mounts += ["-v", f"{holdout}:/holdout:ro"]
            ev = run(["docker", "run", "--rm", *mounts, image, "python", "-m", "scripts.eval", labels,
                      "--photos", photos, "--json", f"/out/{set_name}.json"])
            if ev.returncode:
                return {**result, "error": f"eval on {set_name} failed", "log": ev.stderr[-2000:]}
            result[set_name] = json.loads((out_dir / f"{set_name}.json").read_text())
    run(["docker", "rmi", image])
    return result


# Pass only if tests pass, no eval set loses top-1, and no photo that was right is now wrong
def gate(result, baseline):
    if "error" in result:
        return False, result["error"]
    if not result["tests_passed"]:
        return False, "tests failed"
    for set_name in ("eval", "tune", "holdout"):
        if set_name in baseline and result[set_name]["top1"] < baseline[set_name]["top1"]:
            return False, f"{set_name} top-1 dropped"
        if set_name != "holdout" and set_name in baseline and compare(baseline[set_name], result[set_name])["broken"]:
            return False, f"breaks {set_name} photos"
    return True, "passes"


def write_report(run_dir, baseline, candidates):
    sets = [s for s in ("eval", "tune", "holdout") if s in baseline]
    lines = ["# Fix experiment report", "",
             "| Candidate | Gate | " + " | ".join(f"{s} top-1" for s in sets) + " | Fixed | Broken | Speed | Flags |",
             "|" + " --- |" * (len(sets) + 6),
             "| baseline | n/a | " + " | ".join(f"{baseline[s]['top1']}/{baseline[s]['total']}" for s in sets)
             + " | | | | |"]
    details = []
    for name, c in candidates:
        if "error" in c:
            lines.append(f"| {name} | ❌ {c['error']} |" + " |" * (len(sets) + 4))
            continue
        # Per-photo changes only for eval and tune; holdout stays as a total so its photos stay blind
        diffs = {s: compare(baseline[s], c[s]) for s in sets if s != "holdout" and s in c}
        fixed = [(s, p, old) for s, d in diffs.items() for p, old in d["fixed"]]
        broken = [(s, p, old) for s, d in diffs.items() for p, old in d["broken"]]
        old_s, new_s = diffs["eval"]["seconds"]
        lines.append(f"| {name} | {'✅' if c['passed'] else '❌'} {c['reason']} | "
                     + " | ".join(f"{c[s]['top1']}/{c[s]['total']}" for s in sets)
                     + f" | {len(fixed)} | {len(broken)} | {(new_s - old_s) / old_s:+.0%} | {len(c['flags'])} |")
        details += ["", f"## {name}", ""]
        notes = run_dir / "candidates" / f"{name}.txt"
        if notes.exists():
            details += [f"> {line}" for line in notes.read_text().splitlines()] + [""]
        details += [f"- fixed ({s}): {p['filename']} {p['expected']} {rank_text(old)} -> #1" for s, p, old in fixed]
        details += [f"- broken ({s}): {p['filename']} {p['expected']} #1 -> {rank_text(p['rank'])}"
                    for s, p, _ in broken]
        details += [f"- ⚠ card-specific? {flag}" for flag in c["flags"]]
        if not c["tests_passed"]:
            details += ["", "```", c["tests_log"], "```"]

    passing = [(name, c) for name, c in candidates if c.get("passed")]
    best = max(passing, key=lambda nc: sum(nc[1][s]["top1"] for s in sets), default=None)
    lines += ["", f"**Recommendation:** {best[0] if best else 'none; no candidate passed the gate'}"]
    (run_dir / "report.md").write_text("\n".join(lines + details) + "\n")


def main(run_dir, holdout=None):
    run_dir = Path(run_dir).resolve()
    # Docker needs this folder to exist to mount the throwaway crops folder over it
    (REPO / "eval" / "crops").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        base_tree = Path(tmp) / "baseline"
        run(["git", "-C", str(REPO), "worktree", "add", "--detach", str(base_tree), "HEAD"], check=True)
        try:
            print("Scoring baseline...")
            baseline = score(base_tree, "baseline", run_dir / "results" / "baseline", holdout)
            if "error" in baseline:
                raise SystemExit(f"Baseline failed: {baseline['error']}\n{baseline.get('log', '')}")
            # If the committed code's own tests fail, every candidate would fail too and the report would mislead
            if not baseline["tests_passed"]:
                raise SystemExit(f"Baseline tests fail; fix them first:\n{baseline['tests_log']}")
            candidates = []
            for diff in sorted((run_dir / "candidates").glob("*.diff")):
                name, text = diff.stem, diff.read_text()
                print(f"Scoring {name}...")
                touched = [f for f in changed_files(text) if f.startswith(PROTECTED)]
                if touched:
                    candidates.append((name, {"error": f"touches protected files: {', '.join(touched)}"}))
                    continue
                tree = Path(tmp) / name
                run(["git", "-C", str(REPO), "worktree", "add", "--detach", str(tree), "HEAD"], check=True)
                try:
                    applied = run(["git", "-C", str(tree), "apply", str(diff)])
                    result = ({"error": "diff doesn't apply"} if applied.returncode
                              else score(tree, name, run_dir / "results" / name, holdout))
                finally:
                    run(["git", "-C", str(REPO), "worktree", "remove", "--force", str(tree)])
                result["flags"] = card_flags(text)
                result["passed"], result["reason"] = gate(result, baseline)
                candidates.append((name, result))
        finally:
            run(["git", "-C", str(REPO), "worktree", "remove", "--force", str(base_tree)])
    write_report(run_dir, baseline, candidates)
    print(f"Report: {run_dir / 'report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="folder with candidates/*.diff; results and report.md are written here")
    parser.add_argument("--holdout", help="folder with labels.csv and photos; only totals are reported")
    args = parser.parse_args()
    main(args.run_dir, args.holdout)

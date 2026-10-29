"""Compare two eval runs saved with eval.py --json. Run: python -m scripts.compare_evals before.json after.json

Lists photos a change fixed and photos it broke, plus the change in accuracy and speed.
"""
import argparse
import json
from statistics import median


# Show a rank as "#2", or "not in top 5"
def rank_text(rank):
    return f"#{rank}" if rank is not None else "not in top 5"


# Match photos by filename and sort them into fixed (wrong -> right at #1) and broken (right -> wrong)
def compare(before, after):
    old = {p["filename"]: p for p in before["photos"]}
    new = {p["filename"]: p for p in after["photos"]}
    shared = [name for name in old if name in new]
    return {
        "top1": (before["top1"], after["top1"]),
        "top5": (before["top5"], after["top5"]),
        "fixed": [(new[n], old[n]["rank"]) for n in shared if old[n]["rank"] != 1 and new[n]["rank"] == 1],
        "broken": [(new[n], old[n]["rank"]) for n in shared if old[n]["rank"] == 1 and new[n]["rank"] != 1],
        "only_in_one": sorted(set(old) ^ set(new)),
        "seconds": (median(p["seconds"] for p in before["photos"]), median(p["seconds"] for p in after["photos"])),
    }


# Print accuracy changes, each fixed and broken photo, and the speed change
def print_comparison(diff):
    (top1_before, top1_after), (top5_before, top5_after) = diff["top1"], diff["top5"]
    print(f"top-1: {top1_before} -> {top1_after}  ({top1_after - top1_before:+d})")
    print(f"top-5: {top5_before} -> {top5_after}  ({top5_after - top5_before:+d})")

    for title, rows in [("Fixed (wrong before, right now)", diff["fixed"]),
                        ("Broken (right before, wrong now)", diff["broken"])]:
        print(f"\n{title}: {len(rows)}")
        for photo, old_rank in rows:
            print(f"  {photo['filename']:<14} {photo['expected']:<13} {rank_text(old_rank)} -> {rank_text(photo['rank'])}")

    if diff["only_in_one"]:
        print(f"\nIn only one run (not compared): {', '.join(diff['only_in_one'])}")

    old_s, new_s = diff["seconds"]
    print(f"\nSpeed: median {old_s:.2f} s -> {new_s:.2f} s per photo ({(new_s - old_s) / old_s:+.0%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("before", help="eval.py --json output for the old code")
    parser.add_argument("after", help="eval.py --json output for the new code")
    args = parser.parse_args()

    with open(args.before) as f, open(args.after) as g:
        print_comparison(compare(json.load(f), json.load(g)))

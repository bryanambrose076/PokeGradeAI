---
name: fix-matcher
description: Diagnose why PokeGrade's card matcher picks the wrong card on labeled misses, then write and test 2-3 candidate fixes on git branches. Use when asked to improve matching accuracy or fix scans the app got wrong.
---

# Fix the matcher

PokeGrade identifies a Pokemon card from a phone photo: it crops the card out of the photo, embeds the crop with DINOv2,
and picks the closest of 4,585 reference images. Your job is to find out why it gets some photos wrong and try fixes.
`pipeline.md` (next to this file) explains each step and where it lives in the code; read it before changing anything.

## 1. Diagnose

1. Run both evals and save their results:
   - `python -m scripts.eval --json /tmp/eval.json` (the 25 eval photos)
   - `python -m scripts.eval eval/tune/labels.csv --photos eval/tune --json /tmp/tune.json` (real user scans the app got wrong)
2. Make side-by-side images of every miss: `python .claude/skills/fix-matcher/compare.py /tmp/eval.json /tmp/tune.json`.
   Each image shows the crop the matcher used, the right card, and the app's guess. Look at every one.
3. Name the cause for each miss: a bad crop (wrong outline, sleeve or toploader edge, sideways), glare, a color cast,
   blur, a same-art variant, or something else. Look for what several misses have in common.
4. Read `past_attempts.md` before deciding what to try, so you don't repeat something that already failed.

## 2. Fix

Write 2 or 3 *different* candidate fixes for the cause you found. Each goes on its own branch made from `base`:
`git checkout -b fix-1 base`, change the code, check it (below), then `git commit -am "<cause>: <what the fix does>"`.
Name the branches fix-1 to fix-3.

Check each candidate before committing: `pytest -q`, then both evals again, and compare with your first run.
A candidate that makes a photo that was right go wrong is not a good candidate; try something else.

## 3. Finish

End with a short summary: the cause you found, the evidence (which photos, what the side-by-side images showed),
and one line per candidate with its eval and tune top-1.

## Rules

- **Fix causes, never cards.** No card IDs, set codes, photo names or anything that only helps the photos you can see.
  A fix must make sense for a photo of a card you have never seen. Your candidates are scored afterwards on photos
  you can't see, so a fix that only works here will show.
- **Don't edit** anything in `eval/`, `scripts/`, `.claude/`, `.github/`, `Dockerfile*`, `requirements*` or
  `pytest.ini`. Candidates that do are thrown out without being scored.
- **Prefer fixes on the photo side** (cropping, cleaning up the crop before it's embedded). The reference embeddings in
  `data/embeddings.pt` were built once; a change to how *every* image is embedded would need them all rebuilt.
- **Keep scans fast.** The eval prints time per photo; a fix that makes every scan much slower needs to be worth it.
- Keep each candidate small and focused. Don't install packages. You have no database, S3 or GitHub access and don't need them.

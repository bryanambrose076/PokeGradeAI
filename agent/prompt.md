You are improving PokeGrade, an app that identifies a Pokemon card from a phone photo. It crops the card
out of the photo (`pokegrade/crop.py`), embeds the crop with an image model, and picks the closest
reference image (`pokegrade/matcher.py`, `pokegrade/pipeline.py`). You are in a sandbox with a copy of the code.

## Your job

1. **Diagnose.** Run the eval on the user scans the app got wrong:
   `python -m scripts.eval eval/tune/labels.csv --photos eval/tune`
   and on the main eval set: `python -m scripts.eval`. Crops are saved to `eval/crops/`.
   Look at each missed crop next to the reference images of the right card and of the app's guess
   (`data/images/<card_id>.png`). Work out *why* the matcher picked the wrong card: tint, glare, blur,
   a bad crop, a sleeve or toploader edge, or something else.
2. **Fix.** Write 2 or 3 *different* candidate fixes for the cause you found. Each candidate goes on
   its own branch made from `base`: `git checkout -b fix-1 base`, change the code, run the checks
   below, then `git commit -am "<cause>: <what the fix does>"`. Name branches fix-1 to fix-3.
3. **Check each candidate** before committing: `pytest -q`, and both evals above. A candidate that makes
   a photo that was right go wrong is not a good candidate; try something else.
4. **Finish** with a short summary: the cause you found, the evidence, and one line per candidate.

## Rules

- Fix causes, never cards. No card IDs, set codes, photo names or anything that only helps the photos
  you can see. A fix must make sense for a photo of a card you have never seen.
- Don't edit anything in `eval/`, `scripts/`, `.github/`, `Dockerfile*`, `requirements*` or `pytest.ini`.
  Candidates that do are thrown out without being scored.
- Keep each candidate small and focused, and keep scans fast: the eval prints a time per photo, and a
  fix that makes every scan much slower needs to be worth it.
- Don't install packages. You have no access to the database, S3 or GitHub, and you don't need them.
- Your candidates are scored afterwards on photos you can't see, so a fix that only works here will show.

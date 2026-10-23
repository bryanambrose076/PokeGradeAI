# Decisions Log — Pokémon Card Scanner

One entry per technical fork. Written as decisions were actually made; superseded decisions are kept and marked, not deleted.

---

## 1. Card ID reference source
**Decision:** TCGdex, via the `tcgdexsdk` Python package.
**Options considered:** TCGdex, pokemontcg.io.
**Tradeoffs:** TCGdex has an official SDK, making it straightforward to pull card name/set/number/image URLs into structured JSON per set.
**Picked:** TCGdex — SDK made the reference-data pull simple, and image URLs cover the full set list needed (Scarlet & Violet era sets + related).
**Would change my mind if:** image quality/resolution turned out too low for embedding-based matching, or the SDK stopped being maintained.

**Follow-up — local image storage:** images saved as `data/images/{set_id}/{localId}.png`, nested by set. This means card numbers (which repeat across sets) stay unique without needing to edit the JSON to point at local files — the path is computed at runtime from `set_id` + `localId`, which are already in each JSON entry. No separate image-location field needed in the JSON.

**Known data gap (found 2026-10-15):** 34 cards in `svp` (Black Star Promos) have `image_url: None` in TCGdex, so they have no image and no embedding and can never be matched (e.g. Umbreon ex `svp/176`). All other sets are complete (4585 of 4619 cards have images).

---

## 2.1 Embedding model + vector index
**Status:** Superseded by 2.3 (switched to DINOv2 on 2026-10-15).
**Decision:** CLIP `openai/clip-vit-base-patch32`, loaded via HuggingFace `transformers` (`CLIPModel` / `CLIPProcessor`).
**Options considered:** OpenAI's original `clip` package, `open_clip`, larger CLIP variants (ViT-B/16, ViT-L/14, ViT-L/14@336px).
**Tradeoffs:** ViT-B/32 and ViT-B/16 both produce 512-dim embeddings; ViT-L/14 variants produce 768-dim and score higher on general benchmarks but are slower and produce larger reference files. Coarser patch size (32px) may lose fine card detail (small text, set symbols) compared to smaller-patch or higher-resolution variants.
**Picked:** ViT-B/32 via `transformers` — fast, well-documented, easy to defend as a baseline; matches the "get something working end-to-end first" approach before optimizing.
**Would change my mind if:** real-world testing shows the model can't reliably distinguish visually similar cards (see investigation note below). *→ This happened; see 2.3.*

**Bug encountered:** `CLIPModel.get_image_features()` in the installed `transformers` version returns a `ModelOutput`-style object, not a plain tensor. Indexing it with `[0]` grabs `last_hidden_state` (shape `[1, 50, 768]` — full per-patch sequence), not the pooled embedding. The correct pooled 512-dim embedding is at `.pooler_output`. This mistake caused a ~300x-oversized `.pt` file (1.2GB+ instead of ~10MB) before being caught.

**Investigation — score separation / model resolution:** Ran a control test comparing a real query photo against (a) its correct top-5 CLIP matches and (b) one deliberately unrelated reference card. Correct top-5 scores clustered ~0.84–0.89; deliberately unrelated card scored ~0.76. Real but modest separation — top candidates are a legitimate tier above "wrong," but the gap *within* the top-5 is small (~0.03 between #1 and #2), leaving some risk of a wrong answer outranking the right one on a harder photo. Not yet enough evidence to justify switching models. Revisit after step 3 (localization) — if confidence issues persist on clean, localized photos (not just on messy background-heavy ones), that's stronger evidence for a resolution-limited model and a case to test a larger CLIP variant or a different embedding model. *→ Resolved in 2.3: confidence issues did persist on clean crops, and the fix turned out to be a different kind of model, not a larger CLIP (ViT-B/16 was tested and was worse).*

---

## 2.2 Brute-force vs. FAISS benchmark
**Decision:** Brute-force cosine similarity (single matrix-vector multiply via PyTorch), no FAISS.
**Benchmark:** Full search over 4585 reference embeddings took ~0.27ms on real hardware (measured directly, with 512-dim CLIP vectors). Embeddings are now 768-dim DINOv2 (2.3), 1.5× the work, so still well under a millisecond.
**Picked:** Brute-force — FAISS is built for the millions-of-vectors, high-throughput case; at 4585 vectors the search is already sub-millisecond, so FAISS would add complexity without solving a real bottleneck.
**Would change my mind if:** reference set grows by orders of magnitude (roughly 100k+ vectors) or query volume/latency requirements change significantly.

---

## 2.3 Embedding model switch: CLIP → DINOv2
**Date:** 2026-10-15
**Decision:** Replaced CLIP ViT-B/32 with DINOv2 (`facebook/dinov2-base`, 768-dim `pooler_output`), loaded via `transformers` `AutoModel` / `AutoImageProcessor`. Preprocessing squashes the whole card to 224×224 instead of the processor's default center crop (which cuts off the card's name and set number). Card images and scan queries share one `embed_images()` function in `cosineSimilarity.py`, so both are always preprocessed identically. `embeddings.pt` rebuilt with `embedTest.py` (4585 cards, 15MB). Dockerfile pre-downloads DINOv2 instead of CLIP.
**Trigger:** Entry 2.1's "would change my mind" condition was met. Checked the real test photos by eye instead of trusting the model's output, and CLIP had been wrong on 2 of 12: `test9` (a Japanese print of Litwick, `sv10.5w/101`) came back as Tranquill, and `test10` (Alakazam ex 201/165, `sv03.5/201`) came back as Frogadier. Several correct matches were near-ties, with gaps of 0.002–0.007 to #2.
**Lesson:** my earlier "correct" labels for those photos had come from the model's own output. Ground truth has to come from looking at the physical card.
**Options considered** (12 real photos end-to-end, plus 300 synthetic photos made from reference images with tilt, glare, blur, color shifts and JPEG compression):

| Option | Real (12) | Synthetic (300) | Speed/scan |
|---|---|---|---|
| CLIP ViT-B/32 (baseline) | 10/12 | 94.0% | 33 ms |
| CLIP ViT-B/16 | 3 new misses on the original set | n/a | similar |
| CLIP, inset crop 4% / 8% | 9/12 / n/a | 93.3% / 88.7% | 33 ms |
| CLIP, whole card (no center crop) | 10/12 | 93.0% | 54 ms |
| **DINOv2-base, whole card** | **12/12** | 90.3% | 107 ms |
| CLIP + SIFT re-rank of top 10 | 11/12 | 98.0% | 578 ms |
| DINOv2 + SIFT re-rank of top 10 | 12/12 | 97.0% | 546 ms |

**Tradeoffs:**
- CLIP is trained on image–caption pairs, so it groups cards by what they're *about*; visually different cards with similar subjects score close together. DINOv2 is trained to recognize the same image under augmentations, which is closer to "which exact card is this."
- DINOv2's gap to the #2 match was 3–6× larger on most real photos (e.g. Psyduck 0.007 → 0.111, Fidough 0.044 → 0.288).
- Against it: a lower synthetic score (90.3% vs 94.0%) and ~3× slower (still small next to cropping and the pricing call). The real-photo sample is small.
- DINOv2 is trained to ignore color shifts, which helps with glare and lighting but makes same-art color variants nearly tie (gold Mew ex `sv03.5/205` vs pink `sv03.5/193`: gap 0.015).
- Raw DINOv2 scores are on a different scale than CLIP's (correct matches ~0.60–0.89). Thresholds must be based on the #1–#2 gap, not the raw score.
- SIFT re-ranking: best on synthetic images, but no gain on real photos. CLIP + SIFT picked Alakazam ex 188 instead of 201 because the two share a lot of identical text. It also needs the reference images (or precomputed descriptors, ~0.5–1GB) at scan time and adds ~0.5s/scan.

**Picked:** DINOv2-base alone.
**Would change my mind if:** more real scans show DINOv2 misses that CLIP gets right; same-art color variants prove common (planned fix: re-rank near-tied candidates by color similarity); or real-world misses appear that SIFT re-ranking would catch (cleanest deployment: reference images in S3, fetch top-10 per scan).

---

## 2.4 Low-confidence warning
**Date:** 2026-10-15
**Decision:** If the top match beats the runner-up by less than `LOW_CONFIDENCE_GAP = 0.03` (`pipeline.py`), `/scan` returns `low_confidence: true` plus the runner-up card, and the page shows "⚠️ Not sure. Could also be X."
**Tradeoffs:** Doesn't improve accuracy, but stops the app from quietly showing a possibly-wrong card. The threshold is based on DINOv2 gaps from the real test photos: correct matches were 0.088+, except Kakuna (0.014), which gets flagged even though it's correct. The gold vs pink Mew ex (gap 0.015) is also flagged, which is the right call since the two versions price differently.
**Would change my mind if:** real scans show the threshold flags too many correct matches (lower it) or lets wrong ones through (raise it).

---

## 3. Localization approach
**Date:** 2026-09-09 (updated 2026-10-15, see follow-up)
**Decision (current):** OpenCV contour detection + rotated-rect perspective warp to localize, de-skew, and crop the card before embedding: shrink to 1000px on the longest side → `cv2.Canny` → `cv2.dilate` → `cv2.findContours(RETR_LIST)` → pick the largest *card-shaped* contour (falling back to the largest overall) → `cv2.minAreaRect` / `cv2.boxPoints` → corner ordering → `cv2.getPerspectiveTransform` + `cv2.warpPerspective` to a fixed 500×700 output (matches a real card's 2.5"×3.5" aspect ratio).
**Original decision (2026-09-09):** same pipeline, but no resize, `RETR_EXTERNAL`, and simply the largest contour by area.
**Options considered:**
- `cv2.approxPolyDP` (reduce contour to its natural polygon, filter for exactly 4 points) vs. `cv2.minAreaRect` (always returns the smallest rotated rectangle around the contour, no filtering needed).
- Running `findContours` directly on raw Canny output vs. dilating the edge map first.
**Tradeoffs:**
- `approxPolyDP` is more general-purpose but requires a "did this reduce to ~4 points" check to confirm it actually found a rectangle. `minAreaRect` skips that entirely, which is a reasonable simplification here since the target object is known in advance to be rectangular (a card) — but it never verifies the contour *is* rectangular, so a bad/occluded contour would still silently produce a confident-looking wrong box.
- Raw Canny edges were not reliable input for contour detection: the card's outer border had small gaps (low-contrast points), so it wasn't one continuous closed loop. `findContours` split it into disconnected arcs with small enclosed area each, and a fully-closed *internal* shape (a decorative line under the card's ability text) won "largest contour" instead of the card border. Dilating the edge map with a small 3×3 kernel bridged the gaps and fixed this.
- `cv2.boxPoints()` does not return corners in a guaranteed top-left/top-right/bottom-right/bottom-left order, which `getPerspectiveTransform` needs to match against the destination rectangle. Fixed with a sum/diff sort: top-left = min(x+y), bottom-right = max(x+y), top-right = min(y−x), bottom-left = max(y−x).
**Picked:** The pipeline above — validated end-to-end (see below), not just visually.
**Would change my mind if:** the card-shape filter picks the wrong outline often (multiple cards in frame, heavy occlusion), or crops of sleeved cards keep including the sleeve. At that point, worth testing `approxPolyDP` corner checks or a learned card detector.

**Validation — does it actually improve matching (2026-09-09, with CLIP):** Fed the same physical card photo through the CLIP + cosine similarity pipeline, raw vs. localized/cropped.
- **Raw photo:** top-5 matches were all wrong cards, tightly clustered (0.8589–0.8648, ~0.006 spread). The correct card (`sv06.5/082`, Okidogi ex) didn't appear in the top 5 at all — the matcher was confidently wrong.
- **Localized/cropped photo:** correct card (`sv06.5/082`) ranked #1 at 0.8798, with a real ~0.04 gap to #2 (0.8391) — a meaningful confidence margin instead of the raw photo's ambiguous clustering.
- **Conclusion:** localization wasn't a cosmetic fix — it turned a confidently-wrong match into a confidently-right one with real separation.

**Follow-up — card-shaped contour selection (2026-10-15):** The original "would change my mind" condition (a busy background producing a bigger false contour) happened. A Zorua (`sv10.5w/142`) photographed on a laptop keyboard came back as Tranquill from a phone upload. "Largest contour" failed two ways:
- At some sizes, the sleeve's edge merged with the trackpad's edge into one big shape, so the box was much larger than the card.
- At phone resolution (~4000px), the 5×5 blur and fixed Canny thresholds (tuned for small images) left the card outline fragmented, and the largest piece was the energy symbol.

**Fix:** (1) shrink to 1000px on the longest side before edge detection (`max_side` parameter); (2) `RETR_EXTERNAL` → `RETR_LIST`, so the card's outline is found even when it's nested inside a merged shape; (3) `find_card_contour()` picks the largest contour whose `minAreaRect` is card-shaped (`CARD_RATIO = 63/88`, ±0.08, at least 5% of the image), falling back to the largest overall.
**Result:** Zorua correct at every size from 1000 to 5400px; the 10 existing test photos still matched the same cards.
**Tradeoff:** "Largest card-shaped" sometimes picks the sleeve/toploader outline instead of the card's own edge. On `test6` (Psyduck) the crop now includes the toploader, and its CLIP gap dropped 0.048 → 0.007 (DINOv2 later brought it to 0.111).
**Rejected:** "smallest card-shaped contour" (11/13; sometimes picks the inner art box) and "crop every candidate and keep the one the model is most confident about" (12/13; picked a crop CLIP was confidently wrong about). Resizing alone, or the shape filter alone, didn't fix Zorua; it took the combination with `RETR_LIST`.

**Follow-up — eval harness + candidate crops (2026-10-20):**
**Eval set:** 25 phone photos of my own cards in `eval/photos/` (not in git), labeled from the physical cards in `eval/labels.csv`, set up like a card show: penny sleeves, toploaders, a PSA and a CGC slab, a sealed promo wrapper, held in hand, overlapping other cards, tilted/rotated, busy backgrounds. `python eval.py` runs them through the same code as `/scan` (no pricing calls) and prints top-1/top-5 and the #1–#2 gap per photo; chosen crops are saved to `eval/crops/` for checking misses by eye.
**Baseline:** 16/25 top-1 (64%), 19/25 top-5. 8 of the 9 misses were the crop, not DINOv2: toploader or slab outline picked instead of the card, sideways crops (`order_points` can't tell a card's top from its side), and in-hand/desk-edge photos where the card's outline never forms one contour, so the fallback grabbed the desk or the whole photo.
**Decision:** `crop_candidates()` returns several crops and `match_candidates()` embeds them in one batch:
- crop #1 is the existing `crop_card()` result and stays the answer unless another crop's top score beats it by ≥ 0.04;
- card-shaped boxes from three Canny sensitivities (±0.10 ratio, ≥10% of the image), turned upright so the long side is vertical;
- a 180° flip, only for boxes tilted ≥ 20° (where "which way is up" is ambiguous);
- three card-shaped windows around the photo center (50/65/80% of the frame), for when no outline is found. Users center the card they're scanning.

| Variant | Top-1 | Top-5 | Mac ms/scan |
|---|---|---|---|
| Current `crop_card` | 16/25 | 19/25 | ~90 |
| Upright fix + largest box | 8/25 | 14/25 | ~100 |
| All candidates, highest score wins | 18/25 | 20/25 | ~255 |
| Candidates + 0.04 margin, flip only tilted | 21/25 | 22/25 | ~255 |
| **+ center windows** | **23/25** | **25/25** | ~370 |

**Tradeoffs:**
- Flipping *every* crop caused confident false matches (an upside-down Umbreon scored 0.80 as Mega Greninja ex), hence flips only for tilted boxes. Without the 0.04 margin, random patches sometimes beat the real crop.
- Center windows fixed both in-hand photos and the desk-edge Charmander, but pushed Canari (toploader) from #1 to #2.
- Slower: matching 6–13 crops is ~1.5–2s on CPU in Docker vs ~0.3s for one crop. Easy fix if needed: skip the extra crops when crop #1 is already confident.
- The margin, tilt cutoff and windows were tuned on these same 25 photos, so 92% is optimistic until checked on new photos.
- Why this works now when "keep the most confident crop" failed with CLIP: DINOv2 scores a bad crop much lower, and the margin keeps the default crop unless another is clearly better.
**Still wrong:** Clefairy in the CGC slab (crop is correct, slab glare puts the right card #3), Canari (#2). The 10 older test photos all give the same answer as before, except Umbreon ex `svp/176`, which has no reference image and was wrong both ways (it now scores 0.79 as Hydreigon ex, so the score alone can't flag unknown cards).
**Would change my mind if:** new photos show a much lower accuracy than 92% (re-tune on a bigger set), or Lambda scan time is too slow (add the early exit above, or fewer center windows). The long-term fix for the outline-not-found cases is a small trained card detector.

---

## 4. Pricing source(s) + demo terms
**Date:** 2026-10-14
**Decision:** PokemonPriceTracker API v2 (`/api/v2/cards`, `includeEbay=true`), one request per scan. Raw price = `prices.market`; PSA 9 / PSA 10 = `ebay.salesByGrade.psa9/psa10.smartMarketPrice.price`. The same response also includes CGC 9/9.5/10 and TAG 9 grades (unused).
**Demo terms:** 100 requests/day. Tests fake the API (monkeypatch) so they use none.
**Failure handling:** any `requests.RequestException` (missing/invalid key → 401, timeouts, 5xx, non-JSON) is caught in `scan_image`; the scan still returns the card with `prices: null` instead of a 500. Individual grades with no sales come back `null` (e.g. Kakuna has no graded sales). The API key is read from `.env`, which is excluded from the Docker image, so it must be passed at runtime (`--env-file` or the host's environment variables).
**Tradeoffs:**
- Graded prices are often "low" confidence and can diverge from the median sale (Okidogi ex PSA 10: smart price $82 vs median $36.50, weighted toward a few recent sales).
- PSA 9 can be below the raw price (Mega Greninja ex: PSA 9 $126.50, raw $128.06), a real data quirk.
**Would change my mind if:** the daily quota becomes a constraint (add ~24h price caching keyed by card), or price confidence proves too unreliable to show without a warning.

---

## 5. Grading ROI formula/thresholds
**Date:** 2026-10-14
**Decision:** `grading_roi(raw_price, psa10_price)` in `pricing.py` returns *profit* (not a ratio), assuming the card grades PSA 10.
**Formula:** profit = PSA 10 price − raw price − grading fee, with the fee picked by the PSA 10 value tier:

| PSA 10 price | Fee |
|---|---|
| ≤ $500 | $27 |
| ≤ $1,000 | $66 |
| ≤ $1,500 | $88 |
| ≤ $2,500 | $220 |
| > $2,500 | $380 |

Returns `None` if either price is missing (checked with `is None`, so a $0 price still counts). Returned by `/scan` as `profit` and shown on the page as "Profit if PSA 10"; negative values display as a loss (e.g. -$17.00). Covered by `tests/test_pricing.py`, including the 500/501 tier boundary.
**Picked:** a simple PSA 10-only profit for now. PSA 9 is displayed but intentionally not part of the formula.
**Would change my mind if:** I want a realistic expected value: weight PSA 10 and PSA 9 outcomes by an estimated chance of each grade. Fee tiers should also be re-checked against PSA's current pricing.

---

## 6. Deployment target
**Date:** 2026-10-20
**Decision:** AWS Lambda running the existing Docker image (container image from ECR), behind a public Function URL, in `us-east-2`. Function `PokeGradeAI`: arm64, 3008 MB memory, 180s timeout, env vars `POKEMONPRICETRACKER_API_KEY` and `OMP_NUM_THREADS=2`.
**Options considered:**
- **Lambda (container image) + Function URL:** about $0 at demo traffic (free tier), ECR storage ~$0.10–0.20/month for the ~790MB compressed image.
- **ECS on Fargate:** always on, no cold starts, but ~$25/month unless created and destroyed per demo.
- **Lambda Web Functions** (newer console option): only accepts code as a ZIP or from S3, no container images, and PyTorch + DINOv2 needs the container.
- **Mangum vs. Lambda Web Adapter** to run FastAPI on Lambda: Mangum needs a handler in `main.py` plus the Lambda base image or runtime client; the Web Adapter is one `COPY` line in the Dockerfile.
**Tradeoffs:**
- Lambda's near-zero idle cost keeps the demo link live for free; the cost is cold starts (below).
- Web Adapter keeps the app a normal uvicorn server, so the exact image tested locally is what runs on Lambda, and it would also run on Fargate unchanged. The model is downloaded to `/app/.hf_cache` instead of `/root/.cache` because Lambda doesn't run as root.
- arm64 because the image is built on an Apple Silicon Mac (and arm64 Lambda is cheaper).
- Uploads: Lambda caps requests at ~6MB and full-size iPhone JPEGs are ~3–4MB+, so the page shrinks photos in the browser before upload. Tested on the eval set: 2000px / JPEG 90 changed no answers (23/25), while 1600px/90 dropped to 21/25 and 1000px/90 to 22/25. The photos that flipped were ones already winning by tiny gaps (Mewtwo 0.003), so it's mostly borderline cases tipping, but 2000px was the only setting with no changes. Uploads are ~0.5MB.
**Picked:** Lambda + Web Adapter + Function URL.

**Cold starts (found on first deploy):**
- Lambda gives a function 10s to start; this app needed ~60s (import torch, read the 1.85GB image for the first time, load DINOv2 + embeddings). Lambda then retries the start inside the request, counted against the timeout. With a 60s timeout, the first request 502'd at 60.0s and the second finished at 59.9s. Fix: timeout → 180s (Lambda's default is 3s, raised to 60s at setup).
- Then the *first scan* on each new instance took ~56s while later scans took ~5s. Not CPU threads (`OMP_NUM_THREADS=2` didn't change it): Lambda loads container image files lazily, so the first pass over the model weights was slow. PyTorch's `Error in cpuinfo` / `Can't open MIDR_EL1` log lines are harmless warnings on Lambda's ARM machines.
- Fix: `main.py` runs one match on a blank image at startup, so the slow first pass happens during startup. Result: cold start ~16s, first scan 5.3s (was ~56s).
- Keep-warm: EventBridge Scheduler `pokegrade-keepwarm` invokes the function every 5 minutes (IAM role `pokegrade-keepwarm`, allowed to invoke only this function). The Web Adapter forwards these as `POST /events`, which returns a harmless 404. ~8,600 invocations/month, far inside the free tier. Two people scanning at the same moment can still hit a second, cold instance.

| | Before | After |
|---|---|---|
| Cold start | ~60s, first request often failed | ~16s |
| First scan on a new instance | ~56s | ~5s |
| Later scans | ~5s | ~5s |

**Would change my mind if:** real traffic makes cold starts common (try SnapStart, which snapshots the instance after the warm-up; it needs the Function URL on a published version/alias), or the demo needs steady traffic with no cold starts at all (Fargate). Cleanup still to do: move the setup into Terraform.

**Follow-up — least-privilege execution role (2026-10-22):** The function was created with the console's default role, `PowerUserRole` (AWS-managed `PowerUserAccess`: everything except managing IAM), behind a public URL. Switched it to a new role, `pokegrade-lambda-exec`, with only `AWSLambdaBasicExecutionRole` (write CloudWatch Logs). The app calls no other AWS service; pricing goes to PokemonPriceTracker over the internet.
- Why it matters: the public URL alone doesn't hand anyone those permissions, since visitors can only call the app's endpoints. The risk is a bug that lets someone run code inside the function (e.g. a malicious image exploiting OpenCV or Pillow); that code gets whatever the role allows. Logs-only limits the damage to writing logs.
- Checked after the swap: cold start, `/health` and `/` all 200, and logs still arrive. The keep-warm schedule uses its own role, so it's unaffected. `PowerUserRole` itself was left in place (part of the account's default setup), just no longer used here.

---

## 7. CI/CD
**Date:** 2026-10-22
**Decision:** GitHub Actions workflow (`.github/workflows/ci.yml`) runs `pytest` on every push and pull request: checkout → Python 3.13 (same as the Dockerfile, pip cache on) → install `requirements-dev.txt` with CPU-only torch (the default PyPI torch is a ~2GB GPU build) → `pytest -q`. First run took 46s.
**Endpoint test:** `tests/test_scan_endpoint.py` posts a real photo (`tests/fixtures/mew_ex_gold.jpg`, my gold Mew ex shrunk to 1000px, 113KB) to `/scan` through FastAPI's `TestClient`, with `get_card_prices` monkeypatched to raise a connection error. It asserts a 200, the right card (`sv03.5/205`), and `prices`/`profit` null. This guards the "pricing failure turns the scan into a 500" bug at the endpoint level (upload parsing, the try/except, JSON), which calling `scan_image()` directly would skip. It runs the real model, since `data/embeddings.pt` (15MB) is in git, so CI downloads DINOv2 each run (~70s total now).
**Tradeoffs:**
- Real model instead of a fake matcher: slower CI, but it tests the actual crop + match path. Caching the model between runs can bring the time back down.
- `httpx` pinned in `requirements-dev.txt` because `TestClient` needs it and it was only arriving indirectly through Hugging Face. Starlette now warns that `httpx` is deprecated in favor of `httpx2` (Pydantic's fork, first released May 2026); kept `httpx` for now since it's only a warning.
- The eval photos are gitignored, so the eval can't run in CI yet (options: commit 2000px copies ~12MB, keep them in S3, or run the eval by hand).
**Next:** build the Docker image in CI and run the tests inside it (test exactly what ships), then an OIDC role so GitHub can deploy without stored AWS keys, then deploy to Lambda on merge to `main`.
**Would change my mind if:** CI time grows enough to slow down pushes (cache the model, or split fast unit tests from the model test).

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
**Would change my mind if:** the daily quota becomes a constraint (*→ done: 24h cache in Postgres, section 8*), or price confidence proves too unreliable to show without a warning.

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
**Decision:** AWS Lambda running the existing Docker image (container image from ECR), behind a public Function URL, in `us-east-2`. Function `PokeGradeAI`: arm64, 3008 MB memory, 180s timeout, env vars `POKEMONPRICETRACKER_API_KEY` and `OMP_NUM_THREADS=2` (plus `DATABASE_URL` since 2026-10-24, section 8).
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

**Follow-up — Docker, deploy and eval in CI (2026-10-23):**
- **Tests run inside the image:** CI builds the Dockerfile and runs `pytest` in that container, so it tests exactly what ships. Before, it tested a separate pip install.
- **Deploy on push to `main`:** a `deploy` job runs only after `test` passes. It pushes the image to ECR, tagged with the commit SHA, and points `PokeGradeAI` at it. There are no stored AWS keys. GitHub gets short-lived credentials through OIDC by assuming `deploy-pokegrade`, whose inline policy only allows pushing to the `pokegrade` repo and updating this one function.
- **Gotcha:** GitHub's OIDC `sub` claim now includes IDs (`repo:bryanambrose076@68830798/PokeGradeAI@1336810682:ref:refs/heads/main`). The console wizard wrote the old `repo:owner/name:...` format, so every assume-role failed until I fixed the trust policy by hand. CloudTrail's `AssumeRoleWithWebIdentity` errors showed the real claim.
- **Account tradeoff:** OIDC setup required upgrading to a paid plan and activating advanced features. That can't be undone, and it created an AWS Organization. In exchange the account no longer expires after the free-plan period. The cost is that the free plan's hard spend cap is gone, so a $10 AWS Budget alert replaces it.
- **Slow first request after a deploy:** the first request on a new image takes ~2 min (uncached image plus the init retry). The deploy job ends with a `/health` curl (retries, 180s per try), so CI absorbs that wait instead of the first visitor, and a new image that never comes up healthy fails the deploy.
- **Eval in CI:** 2000px copies of the 25 eval photos (~12MB) are committed in `eval/photos_2000/`, with one photo's corner blacked out because a credit card was visible. CI runs `eval.py --min-top1 22 --min-top5 25` in the image (~1 min), so a matching regression fails the build before it deploys.
**Would change my mind if:** the eval set grows enough to slow CI (keep the photos in S3, or split them into tune and holdout sets with only holdout in CI).

---

## 8. Price cache (Postgres)
**Date:** 2026-10-24
**Decision:** Cache each card's prices in a Postgres table, `price_cache` (`schema.sql`), hosted on Neon (project `pokegrade`, AWS `us-east-2`, free plan, Postgres 18, pooled connection). `pipeline.py` checks the cache before calling PokemonPriceTracker and saves the answer after. Priced cards stay fresh for 24h; "no price found" answers for 6h, so a card that gets data later is retried sooner. API failures are never cached. The app reads `DATABASE_URL` (`.env` locally, a Lambda environment variable in production); without it, the cache is off and scans work exactly as before.
**Why now:** the API allows 100 calls/day (section 4), and the planned MCP tools and agents will call pricing far more often than a person scanning.
**Options considered:**
- **Neon vs. RDS:** RDS inside a VPC means Lambda also needs a NAT gateway (~$32/month) to keep reaching the pricing API. Neon is free at this size and reached over the internet with TLS. Neon is plain hosted Postgres, so moving off it later is one connection-string change.
- **Lambda memory or `/tmp` instead of a database:** free, but lost whenever Lambda replaces an instance, and not shared between instances.
- **Storing the full API response (`jsonb`):** would allow CGC/TAG grades later without new calls, but meant changing `get_card_prices` and its tests for data the page doesn't show. Skipped; easy to add.
- **pgvector:** not needed. Matching 4,585 vectors in memory takes under 1ms (2.2). Might revisit for finding similar past scans once scans are stored.
**Tradeoffs / details:**
- `price_cache.py` keeps one connection per Lambda instance and retries once on a dropped connection. Neon suspends after ~5 minutes idle and closes connections, while the keep-warm ping keeps Lambda alive much longer, so the saved connection is often dead on the next scan.
- Any database error counts as a cache miss, so a Neon outage costs API calls, never a failed scan.
- Prices are `double precision`, not `numeric`, so values come back as plain floats (no `Decimal` handling in the ROI math or JSON).
- A `found` column separates "API had no data" from "card has no graded sales" (Kakuna), which both have empty PSA columns.
- Freshness is computed with Postgres's clock (`now() - fetched_at`), so it doesn't depend on Lambda's clock.
**Testing:** `tests/test_price_cache.py`: miss then hit, 24h expiry, 6h "no price" expiry, database down, and two `/scan` calls for the same photo making one (faked) API call. Tests use `TEST_DATABASE_URL`, never `DATABASE_URL`; `tests/conftest.py` points every test at an empty test table, or turns the cache off, so a local run can't write to Neon even though `.env` points there. CI adds a `postgres:18` service container and runs the tests in the Docker image with `--network host`. Locally: `TEST_DATABASE_URL=postgresql:///pokegrade_test pytest` against Homebrew Postgres 16.
**Bug caught by the tests:** the endpoint test (pricing faked as down) started returning prices, because the cache test before it had left the gold Mew cached. Emptying the table before every test fixed it, and showed the intended behavior: a cached card still shows prices during an API outage, an uncached one doesn't.
**Result:** real API + Neon from my Mac: first gold Mew scan 1 API call, 3.1s; second scan 0 calls, 0.6s. On Lambda (2026-10-25): the gold Mew came back with the prices my Mac had cached, row unchanged, so Lambda read Neon without calling the API. A Lillipup (`sv10.5w/154`, eval photo `IMG_0275`) scanned twice created one row, and its `fetched_at` didn't change on the second scan, so only one API call was made. Lambda scans took ~4s either way; matching, not pricing, is most of the time there.
**Would change my mind if:** prices need to be fresher than a day (lower the TTL), Neon's free limits or wake-up time become a problem, or schema changes become frequent enough to need migrations (numbered SQL files plus a tool that tracks which ran) instead of one `schema.sql`.

---

## 9. Scan history (S3 + Postgres)
**Date:** 2026-10-25
**Decision:** Every successful scan is saved. `/scan` calls `save_scan()` (`pokegrade/scan_history.py`), which uploads the uploaded photo and the crop the matcher chose to a private S3 bucket (`pokegrade-scans-017421876144-us-east-2-an`, under `scans/<date>/<id>.jpg` and `<id>_crop.jpg`) and inserts a row into a `scans` table: random UUID, time, the top-match card ID, the top 5 with scores (`jsonb`), the #1–#2 gap, the low-confidence flag, both S3 keys, and how long the scan took. `/scan` returns the ID as `scan_id`, which the "Wrong card?" button (next step) will send back so a correction attaches to the right scan.
**Why:** the review agent and accuracy numbers need a record of what the app said on real scans. Before this, a wrong answer left no trace.
**Options considered:**
- **Save every scan vs. only low-confidence ones:** every scan. The app can't tell a right answer from a confidently wrong one (Umbreon `svp/176` scores 0.79 as Hydreigon ex with no warning), correct scans are needed for real accuracy numbers and new eval photos, and storage is cheap (~0.2–0.6MB per scan).
- **Photos in Postgres vs. S3:** S3. Databases are poor at storing files, and Neon's free tier is small; the row stores only the file names.
- **Save in the background after responding vs. before:** before. On Lambda the function is frozen once the response is sent, so background work may never finish. Costs a little time per scan, not counted in `duration_ms`.
- **Neon object storage vs. S3:** S3, reached through the Lambda role with no stored keys.
**Security / privacy:**
- Bucket: all public access blocked, SSE-S3 encryption, no versioning (so deletion is real).
- The Lambda role got one inline policy, `pokegrade-scans-upload`: `s3:PutObject` on `scans/*` only (no read, list or delete). Code running inside the function could at worst add files.
- Lifecycle rule `delete-scans-after-90-days` on `scans/`. Photos approved as labels will be copied to `labels/` (not covered by the rule).
- The page says "Photos are saved for 90 days to improve matching." under the upload button, before the photo is taken. Needs rewording once approved photos are kept longer.
**Failure handling:** saving only happens when both `SCANS_BUCKET` and `DATABASE_URL` are set (only on Lambda), so local runs don't fill Neon with rows that have no photos. If an S3 upload fails, the row is saved with that key empty; if the database fails, `scan_id` is `None`. Either way the user still gets their result.
**Also changed:** the Neon connection moved from `price_cache.py` into `db.py`, shared by the cache and scan history (one connection per Lambda instance, one copy of the reconnect logic). `scan_image()` now also returns the chosen crop. `boto3` added to requirements (~80–100MB in the image; the plain Python base image doesn't include it).
**Testing:** `tests/test_scan_history.py` goes through the real `/scan` with S3 faked and pricing faked: a scan saves a row plus both uploads; S3 down still saves the row; database down still returns the scan with `scan_id: None`; no `SCANS_BUCKET` saves nothing. `tests/conftest.py` removes `SCANS_BUCKET` and empties both tables before every test. 23 tests total.
**Result (live, 2026-10-25):** one gold Mew scan on Lambda returned a `scan_id`; Neon had its row (`sv03.5/205`, gap 0.044, 6,685ms, 5 matches) and S3 had both files (113KB photo, 106KB crop). The price came from the cache.
**Would change my mind if:** storage or privacy concerns grow (shorter retention, or save only scans that get a correction), or saving makes scans noticeably slower (save in a queue instead).

**Follow-up — code layout (2026-10-25):** the app files moved into `pokegrade/` and hand-run scripts into `scripts/`, with clearer names: `imageCrop` → `crop`, `cosineSimilarity` → `matcher`, `populateDatabase` → `fetch_sets` (it downloads card lists, not a database), `embedTest` → `build_embeddings`, `downloadImages` → `download_images`. Scripts run as `python -m scripts.eval`. Earlier sections keep the old names as they were written.

---

## 10. "Wrong card?" reports (corrections)
**Date:** 2026-10-26
**Decision:** After a saved scan, the page shows a **Wrong card?** button. It opens a search box; as the user types, `GET /search` returns up to 10 cards from `data/sets/` ("Did you mean…?"), and tapping one (or **None of these**) sends `POST /corrections` with the `scan_id`, the typed text and the chosen card ID (empty for "None of these"). Reports land in a new `corrections` table as `unverified`; approving or rejecting them is the review page (next step).
**Why:** the app can't tell when it's confidently wrong (Umbreon `svp/176`, section 9). Users who know the card can.
**Options considered:**
- **A "Looks right ✓" button too:** skipped. Unflagged scans are assumed correct. Simpler page, but accuracy from live scans ("share of scans nobody flagged") is an upper bound; the eval set stays the strict number.
- **Free-text only vs. pick from search:** search, so most reports carry a real card ID that can be checked. The typed text is kept anyway, to see how people describe cards.
- **Search library or database search vs. a loop in Python:** a loop. 4,619 cards take milliseconds and need no new dependency. `search_cards` will be reused as an MCP tool.
- **Live search vs. a Search button:** live, waiting until 300ms after the last keystroke so typing "mew" sends one request, not three. Search doesn't touch the pricing API, so request count isn't a cost concern.
**Search details:** a card scores one point per typed word that matches a whole word in its name, set name or number, so "mew" no longer matches Mewtwo, and "gold mew 151" returns the three Mew ex in 151 instead of Golduck. Numbers ignore leading zeros ("umbreon 59" finds `sv08.5/059`). Known gaps: typos find nothing, and names with symbols split into pieces ("Farfetch'd" must be typed "farfetch d"); an LLM fallback is an option if these matter (2d).
**Guardrails:**
- Text must be 1–100 characters; `/search` also cuts input at 100.
- A chosen card must exist in `data/sets/` (`is_real_card`).
- `scan_id REFERENCES scans (id)`, so Postgres rejects reports for scans that don't exist; non-ID text gets the same "Unknown scan" answer.
- One report per scan (`UNIQUE`): resubmitting replaces an unverified report, but approved or rejected ones can't be changed, and the page gets the same "saved" answer either way, so it doesn't reveal what's been reviewed.
- `status` is limited to `unverified`/`approved`/`rejected` by a `CHECK`.
- Card names are put on the page with `textContent`, never `innerHTML`.
- Not yet: rate limiting (2d).
**Testing:** `tests/test_corrections.py`: search finds the right card, ignores leading zeros, matches whole words only, and returns nothing for empty text; reports save as `unverified`, "None of these" saves an empty card, resubmitting replaces an unverified report but not an approved one, and a fake card, fake scan, malformed ID, too-long and blank text are each rejected with the right message. `tests/conftest.py` now empties `corrections` too (Postgres won't empty `scans` while another table references it). 35 tests total.
**Would change my mind if:** reports turn out noisy or abusive (require a reason, rate limit, or hide the button on high-confidence scans), or search misses too often (add typo tolerance).

---

## 11. Review page and labels
**Date:** 2026-10-26
**Decision:** "Wrong card?" reports are reviewed on a local page (`python -m scripts.review`, http://127.0.0.1:8932). Each report shows the photo and crop, the app's guess with its gap, what the user typed and picked, and three actions: **Approve** (as the user's card), **Approve as…** (a card ID I enter, found with the page's search box), and **Reject**. Approving adds a row to a new `labels` table and copies both photos from `scans/` to `labels/` in S3; rejecting only changes the report's status. The actions live in `pokegrade/review.py` so the MCP server can reuse them.
**Why:** reports are unverified (section 10). A label should only exist once I've confirmed the card, since labels will be used to measure and improve matching.
**Options considered:**
- **Local page vs. a password-protected page on the live site:** local. Nothing new is exposed to the internet, and Lambda's role stays upload-only (serving and copying photos would need `s3:GetObject`). It runs with my own AWS login. Cost: no reviewing from my phone, fine at a handful of reports. The MCP server and review agent will also run locally.
- **Using `corrections.status = 'approved'` as the label vs. a separate table:** separate. `corrections` is what users said; `labels` is what's confirmed, and labels may later come from other sources (scans nobody flagged, agent suggestions).
- **A "confirm recent scans" list:** dropped, since unflagged scans are assumed correct (section 10).
**Tune / holdout split:** each label is randomly assigned `tune` (~80%) or `holdout` (~20%) once, when approved, and stored, so a photo never changes sides. Tune labels can be looked at while improving matching; holdout labels are only used to measure, so the numbers stay honest.
**Details:**
- Photos are shown through presigned S3 links that expire after an hour; the bucket stays private.
- Copies under `labels/` aren't covered by the 90-day rule. The page's privacy line now says: "If you report a wrong card, that photo may be kept to train the scanner."
- User text is passed through `html.escape`, since the page shows text strangers wrote; the server only listens on `127.0.0.1`.
- `SCANS_BUCKET` is set inside `scripts/review.py`, not in `.env`, so running the main app locally still doesn't save scans (section 9).
- Approving a "None of these" report needs a card from me; approving or rejecting a report twice is refused.
- `db.py` gained `run_all` (all rows); `run` and `run_all` share one connect-and-retry function.
**Testing:** `tests/test_review.py` (S3 faked): pending reports come oldest first with the app's guess; approve saves the label, copies both photos and marks the report approved; a missing crop copies only the photo; a forced random value lands in `holdout`; "None of these" or a fake card is refused until a real card is given; approving twice or after rejecting is refused; the page escapes `<script>` in user text and shows errors after a redirect. CI now mounts `scripts/` for the tests too, since the image doesn't include it. 44 tests total.
**Would change my mind if:** reports become frequent enough that reviewing at my desk is a bottleneck (move the page online behind real auth), or labels need to include eval photos too (import `eval/labels.csv` into `labels`).

---

## 12. Guardrails
**Date:** 2026-10-26
**Decision:** Two guardrails on top of the ones from section 10: the Lambda function has **reserved concurrency of 3** (at most 3 requests run at once; extra ones get an error), and reports are only accepted within **24 hours** of the scan (`REPORT_WINDOW` in `pokegrade/corrections.py`; older scans get "This scan is too old to report").
**Why:** `/corrections` was already hard to abuse (one report per scan, and each scan costs real matching work), so the bigger risk is someone scripting `/scan`. The account allows 400 concurrent Lambda runs, and the $10 budget alert only emails after money is spent; the cap puts a fixed ceiling on cost. The 24-hour window stops old scan IDs being collected and reported later; real users report right after scanning.
**Options considered:**
- **Cap of 3:** enough for me and a few friends; a 4th scan at the exact same moment sees "Something went wrong. Please try again." Easy to raise in the console (Configuration → Concurrency).
- **Per-IP rate limit:** skipped. Each Lambda instance has its own memory, so a limit would need IP addresses stored in Neon, which is new personal data, for little benefit at this traffic.
- **LLM fallback for search:** skipped. The one real report so far found its card with plain search; revisit if reports show search missing.
**Testing:** `test_old_scan_rejected` ages a scan to 25 hours and checks the report is refused and nothing saved. 46 tests total. Concurrency set and checked from the CLI (`get-function-concurrency` returns 3); `/health` still ok.
**Would change my mind if:** real users hit the concurrency cap (raise it), or abuse shows up from one source (add a per-IP limit, or AWS WAF in front of the function).

## 13. MCP server and the two agents
**Date:** 2026-10-28
**Decision:** One local MCP server (`pokegrade/mcp_server.py`, `MCPServer` from `mcp` 2.x, run over stdio) is the review agent's toolbox. Two agents with **no overlap in what they can do**: the review agent reads scans, reports and labels and can only write rows in `suggestions`; the fix agent is Claude Code in the Docker image with no MCP server, no Neon/S3/GitHub credentials and no report text. The fix agent never pushes or opens a PR: it writes a report and candidate diffs, and I apply the winner myself.
**Why:** The review agent reads text users typed into "Wrong card?", which could be a prompt injection; it can't act, so a trick only produces a bad suggestion I'll reject. The fix agent has a shell and the code, so it only gets a suggestion I picked plus the target crops my script copies in. No GitHub token means a confused run can't reach `main` (merging deploys to Lambda).
**Options considered:**
- **One Claude Code agent doing both:** simpler, but the same agent would read untrusted user text and have a shell, and I wouldn't write an agent loop myself.
- **A general `list_scans` tool:** replaced by `list_unsure_scans` (low confidence, nobody reported). Every list tool returns only failures or likely failures, so the agent doesn't wade through normal scans.
- **Fix agent opens the PR:** dropped; it would need a GitHub token in the container that could also push to `main`.
**Testing:** `tests/test_mcp_server.py` calls each tool directly (filters, unknown scan IDs, tune-only label misses, every tool has a real description). Also started the server over stdio and called `search_cards` through an MCP client. 70 tests total.
**Login:** `db.py` and `pricing.py` load `.env` on import, so the server would quietly get the owner `DATABASE_URL`. It refuses to start without `AGENT_DATABASE_URL` and swaps it in; checked that `current_user` is `pokegrade_agent`. `.mcp.json` starts it from Claude Code and holds no secrets.
**Would change my mind if:** suggestions turn out to be mostly code fixes (then the review step adds little), or the holdout can't be kept out of the fix agent's container.

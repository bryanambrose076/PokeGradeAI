# Decisions Log — Pokémon Card Scanner

One entry per technical fork. Written as decisions were actually made.

---

## 1. Card ID reference source
**Decision:** TCGdex, via the `tcgdexsdk` Python package.
**Options considered:** TCGdex, pokemontcg.io.
**Tradeoffs:** TCGdex has an official SDK, making it straightforward to pull card name/set/number/image URLs into structured JSON per set.
**Picked:** TCGdex — SDK made the reference-data pull simple, and image URLs cover the full set list needed (Scarlet & Violet era sets + related).
**Would change my mind if:** image quality/resolution turned out too low for CLIP matching, or the SDK stopped being maintained.

**Follow-up — local image storage:** images saved as `data/images/{set_id}/{localId}.png`, nested by set. This means card numbers (which repeat across sets) stay unique without needing to edit the JSON to point at local files — the path is computed at runtime from `set_id` + `localId`, which are already in each JSON entry. No separate image-location field needed in the JSON.

---

## 2.1 Embedding model + vector index
**Decision:** CLIP `openai/clip-vit-base-patch32`, loaded via HuggingFace `transformers` (`CLIPModel` / `CLIPProcessor`).
**Options considered:** OpenAI's original `clip` package, `open_clip`, larger CLIP variants (ViT-B/16, ViT-L/14, ViT-L/14@336px).
**Tradeoffs:** ViT-B/32 and ViT-B/16 both produce 512-dim embeddings; ViT-L/14 variants produce 768-dim and score higher on general benchmarks but are slower and produce larger reference files. Coarser patch size (32px) may lose fine card detail (small text, set symbols) compared to smaller-patch or higher-resolution variants.
**Picked:** ViT-B/32 via `transformers` — fast, well-documented, easy to defend as a baseline; matches the "get something working end-to-end first" approach before optimizing.
**Would change my mind if:** real-world testing shows the model can't reliably distinguish visually similar cards (see investigation note below).

**Bug encountered:** `CLIPModel.get_image_features()` in the installed `transformers` version returns a `ModelOutput`-style object, not a plain tensor. Indexing it with `[0]` grabs `last_hidden_state` (shape `[1, 50, 768]` — full per-patch sequence), not the pooled embedding. The correct pooled 512-dim embedding is at `.pooler_output`. This mistake caused a ~300x-oversized `.pt` file (1.2GB+ instead of ~10MB) before being caught.

**Investigation — score separation / model resolution:** Ran a control test comparing a real query photo against (a) its correct top-5 CLIP matches and (b) one deliberately unrelated reference card. Correct top-5 scores clustered ~0.84–0.89; deliberately unrelated card scored ~0.76. Real but modest separation — top candidates are a legitimate tier above "wrong," but the gap *within* the top-5 is small (~0.03 between #1 and #2), leaving some risk of a wrong answer outranking the right one on a harder photo. Not yet enough evidence to justify switching models. Revisit after step 3 (localization) — if confidence issues persist on clean, localized photos (not just on messy background-heavy ones), that's stronger evidence for a resolution-limited model and a case to test a larger CLIP variant or a different embedding model.

---

## 2.2 Brute-force vs. FAISS benchmark
**Decision:** Brute-force cosine similarity (single matrix-vector multiply via PyTorch), no FAISS.
**Benchmark:** Full search over 4585 reference embeddings took ~0.27ms on real hardware (measured directly).
**Picked:** Brute-force — FAISS is built for the millions-of-vectors, high-throughput case; at 4585 vectors the search is already sub-millisecond, so FAISS would add complexity without solving a real bottleneck.
**Would change my mind if:** reference set grows by orders of magnitude (roughly 100k+ vectors) or query volume/latency requirements change significantly.

---

## 3. Localization approach
**Date:** 2026-09-09
**Decision:** OpenCV contour detection + rotated-rect perspective warp to localize, de-skew, and crop the card before embedding: `cv2.Canny` → `cv2.dilate` → `cv2.findContours` (largest by area) → `cv2.minAreaRect` / `cv2.boxPoints` → corner ordering → `cv2.getPerspectiveTransform` + `cv2.warpPerspective` to a fixed 500×700 output (matches a real card's 2.5"×3.5" aspect ratio).
**Options considered:**
- `cv2.approxPolyDP` (reduce contour to its natural polygon, filter for exactly 4 points) vs. `cv2.minAreaRect` (always returns the smallest rotated rectangle around the contour, no filtering needed).
- Running `findContours` directly on raw Canny output vs. dilating the edge map first.
**Tradeoffs:**
- `approxPolyDP` is more general-purpose but requires a "did this reduce to ~4 points" check to confirm it actually found a rectangle. `minAreaRect` skips that entirely, which is a reasonable simplification here since the target object is known in advance to be rectangular (a card) — but it never verifies the contour *is* rectangular, so a bad/occluded contour would still silently produce a confident-looking wrong box.
- Raw Canny edges were not reliable input for contour detection: the card's outer border had small gaps (low-contrast points), so it wasn't one continuous closed loop. `findContours` split it into disconnected arcs with small enclosed area each, and a fully-closed *internal* shape (a decorative line under the card's ability text) won "largest contour" instead of the card border. Dilating the edge map with a small 3×3 kernel bridged the gaps and fixed this.
- `cv2.boxPoints()` does not return corners in a guaranteed top-left/top-right/bottom-right/bottom-left order, which `getPerspectiveTransform` needs to match against the destination rectangle. Fixed with a sum/diff sort: top-left = min(x+y), bottom-right = max(x+y), top-right = min(y−x), bottom-left = max(y−x).
**Picked:** The pipeline above — validated end-to-end (see below), not just visually.
**Would change my mind if:** the "largest contour = the card" assumption breaks — multiple cards in frame, a heavily occluded card, or a busy/high-contrast background that produces a bigger false contour. At that point, worth adding a rectangle-ness check (e.g. via `approxPolyDP` corner count) before trusting `minAreaRect`'s output.

**Validation — does it actually improve matching:** Fed the same physical card photo through the existing CLIP + cosine similarity pipeline (`cosineSimilarity.py`), raw vs. localized/cropped.
- **Raw photo:** top-5 matches were all wrong cards, tightly clustered (0.8589–0.8648, ~0.006 spread). The correct card (`sv06.5/082`, Okidogi ex) didn't appear in the top 5 at all — the matcher was confidently wrong.
- **Localized/cropped photo:** correct card (`sv06.5/082`) ranked #1 at 0.8798, with a real ~0.04 gap to #2 (0.8391) — a meaningful confidence margin instead of the raw photo's ambiguous clustering.
- **Conclusion:** localization wasn't a cosmetic fix — it turned a confidently-wrong match into a confidently-right one with real separation.

---

## 4. Pricing source(s) + demo terms
**Date:**
**Decision:**
**Options considered:**
-
-
**Tradeoffs:**
-
**Picked:**
**Would change my mind if:**

---

## 5. Grading ROI formula/thresholds
**Date:**
**Decision:**
**Formula:**
**Picked:**
**Would change my mind if:**

---

## 6. Deployment target
**Date:**
**Decision:**
**Options considered:**
-
-
**Tradeoffs:**
-
**Picked:**
**Would change my mind if:**
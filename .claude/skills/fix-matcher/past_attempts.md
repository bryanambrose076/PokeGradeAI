# What has already been tried

From the project's decisions log. Don't repeat these without a new reason.

## Embedding model (DINOv2 was picked)

| Option | Real photos (12) | Synthetic (300) | Note |
|---|---|---|---|
| CLIP ViT-B/32 | 10/12 | 94.0% | Groups cards by subject, so different cards with similar subjects tie |
| CLIP ViT-B/16 | 3 new misses | n/a | |
| CLIP with a 4% / 8% inset crop | 9/12 | 93.3% / 88.7% | Worse |
| **DINOv2-base, whole card** | **12/12** | 90.3% | Picked; gaps to #2 were 3-6x larger |
| SIFT re-rank of the top 10 | 11-12/12 | 97-98% | No gain on real photos; ~0.5 s slower; confused cards sharing text (Alakazam ex 188 vs 201) |

## Cropping (the current crop_candidates design)

| Variant | Eval top-1 | Note |
|---|---|---|
| Plain crop_card | 16/25 | 8 of 9 misses were the crop: toploader or slab outline, sideways crops, no outline found |
| "Upright fix" + largest box | 8/25 | Much worse |
| All candidates, highest score wins | 18/25 | Random patches sometimes beat the real crop |
| Candidates + 0.04 margin, flip only tilted boxes | 21/25 | Flipping every crop caused confident false matches |
| **+ center windows (current)** | **23/25** | Fixed in-hand photos; pushed Canari (toploader) from #1 to #2 |

Also rejected: the *smallest* card-shaped contour (sometimes picks the inner art box), and "keep the crop the model is
most confident about" with CLIP (picked a confidently wrong crop).

## Ideas noted but never tried

- Re-rank near-tied candidates by color similarity (for same-art color variants; DINOv2 ignores color).
- Skip the extra crops when crop #1 is already confident (speed).
- A small trained card detector instead of edge detection (bigger project).

## Known misses (as of 2026-10-29)

- Canari (toploader, warm color cast, glare): right card #2, gap 0.018. Was #1 before the center windows were added.
  **Fixed 2026-10-30** by fix-1 (applied): now #1, gap 0.014, still flagged as unsure.
- Lillie's Clefairy ex (CGC slab, holo glare, warm tint): crop looks correct, right card #3, gap 0.017.

## Fix agent runs

Each run's outcome is added here, so the next run starts from what's already known.

### 2026-10-29 (Opus, high effort, 49 turns)

Note: the 2 tune scans are the same photos as eval misses IMG_0277 and IMG_0287, so each fix counts twice in the report.

**Canari (toploader, tilted):** the traced outline was right (#1, 0.666), but a center window scored 0.698 for Ludicolo and
beat crop #1 by more than the margin. Windows only had to beat crop #1, never the best outline.

| Candidate | Change | Eval | Tune | Speed | Outcome |
|---|---|---|---|---|---|
| fix-1 | A center window must beat *every* traced outline by the margin | 24/25 | 1/2 | +1% | Passed; recommended |
| fix-2 | Re-match the chosen crop after a per-channel 1st/99th percentile contrast stretch | 24/25 | 1/2 | +11% | Passed; untested on unseen cards |
| fix-3 | Both | 24/25 | 1/2 | +11% | Passed; no extra photo fixed |

Canari is right after each fix but still under the 0.03 confidence gap (0.006-0.022).

**Failed for Canari / overall:**
- Contrast-stretching every candidate before choosing: fixes Canari, breaks IMG_0291 (a wrong outline wins by 0.004).
- Averaging raw and stretched embeddings: no gain.
- Choosing the crop with the biggest #1-#2 gap: 18/27.
- Color re-rank of near ties: 20-22/27; broke a same-number reprint and other near ties.

**Lillie's Clefairy ex (CGC slab, holo glare, warm tint) is still #2-#3; the crop itself is correct.** Nothing tried moves it to #1:
contrast stretch, black-point lift, dehaze, glare inpainting, CLAHE, sharpening, more saturation, grayscale, the crop 6%
smaller to 10% bigger, exact four corners, score centering or hub correction, color-histogram re-ranking. Mega Clefable ex
and sv08.5/156 keep winning; sv08.5/156 is a "hub" card that scores high for many photos (it also came up for Canari).
Ideas not tried: glare-aware matching, or reading the printed card number to break near ties.

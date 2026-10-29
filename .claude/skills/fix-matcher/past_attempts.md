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
- Lillie's Clefairy ex (CGC slab, holo glare, warm tint): crop looks correct, right card #3, gap 0.017.

## Fix agent runs

Each run's outcome is added here, so the next run starts from what's already known.

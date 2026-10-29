# How a scan is matched

`pokegrade/pipeline.py` `match_photo(img, matcher)` is what the eval and the live site both call:

1. **Crop candidates** (`pokegrade/crop.py` `crop_candidates`). Several 500x700 crops of the card:
   - crop #1 is `crop_card`: shrink to 1000px, Canny edges, dilate, `findContours(RETR_LIST)`, then the largest
     *card-shaped* contour (`find_card_contour`, ratio `CARD_RATIO = 63/88` within 0.08, at least 5% of the image),
     falling back to the largest contour; corners sorted by `order_points`, then a perspective warp.
   - card-shaped boxes from three Canny sensitivities (`find_card_boxes`), turned upright (`portrait_corners`);
   - a 180 degree flip, only for boxes tilted 20 degrees or more;
   - three card-shaped windows around the photo center (50/65/80% of the frame), for when no outline is found.
2. **Embed and match** (`pokegrade/matcher.py` `match_candidates`). All crops are embedded in one batch with DINOv2
   (`embed_images`: the whole crop squashed to 224x224, no center crop) and compared by cosine similarity with the
   4,585 reference embeddings in `data/embeddings.pt`. Crop #1 stays the answer unless another traced-outline crop's top
   score beats it by at least `margin = 0.04`. The center windows (the last `len(CENTER_WINDOW_FRACS)` crops) are
   blind guesses, so a window only wins if it beats the *best* outline crop by the margin.
3. **Confidence** (`pipeline.py`). If #1 beats #2 by less than `LOW_CONFIDENCE_GAP = 0.03`, the scan is flagged as
   unsure. Correct matches usually have gaps of 0.09 or more; both known user misses had gaps of about 0.018.

## Useful facts

- DINOv2 is trained to ignore color shifts. That helps with lighting but makes same-art color variants nearly tie
  (gold vs pink Mew ex: gap 0.015).
- Raw scores aren't comparable across photos (correct matches range 0.60-0.89); use the #1-#2 gap.
- Reference images are clean official scans in `data/images/<set>/<number>.png` (read-only here).
- Cropping and matching take about 0.4 s per photo on a Mac; the live site runs on Lambda's CPU, slower.
- `embed_images` is shared by the photos and the reference cards. Changing it changes what the embeddings mean,
  and `data/embeddings.pt` would need rebuilding from all 4,585 images; prefer changes before it, on the crop.

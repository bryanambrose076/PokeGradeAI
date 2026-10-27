"""Find the card in a photo and straighten it."""
import cv2
import numpy as np
import os


# width / height of a Pokemon card (63mm x 88mm)
CARD_RATIO = 63 / 88


# Sort the 4 corners into top-left, top-right, bottom-right, bottom-left
def order_points(pts):
    pts = pts.astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


# picks the largest card-shaped contour, falling back to the largest contour overall
def find_card_contour(contours, img_area):
    best = None
    best_area = 0
    for contour in contours:
        _, (w, h), _ = cv2.minAreaRect(contour)
        if w == 0 or h == 0:
            continue
        ratio = min(w, h) / max(w, h)
        area = w * h
        if abs(ratio - CARD_RATIO) < 0.08 and area > 0.05 * img_area and area > best_area:
            best = contour
            best_area = area

    if best is None:
        return max(contours, key=cv2.contourArea)
    return best


# Main crop: find the card's outline and flatten it into a straight 500x700 image
def crop_card(image, card_width=500, card_height=700, max_side=1000, debug=False, save_path=None):
    """image: a file path, or a BGR numpy array (e.g. from cv2.imdecode)."""
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image}")
    else:
        img = image

    # shrink large photos so the blur and edge thresholds behave the same at any resolution
    scale = max_side / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Find edges (sudden brightness changes) in the photo
    edges = cv2.Canny(blurred, 50, 150)
    # dilate border to form connected outline
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # Trace the edges into outlines (contours)
    # RETR_LIST keeps inner contours too, so the card outline is found even when it merges with the background
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in image — check lighting/background")

    card_contour = find_card_contour(contours, img.shape[0] * img.shape[1])

    rect = cv2.minAreaRect(card_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    ordered_box = order_points(box)

    # define the shape of the card borders after transformation
    dst_points = np.array([
        [0, 0],
        [card_width - 1, 0],
        [card_width - 1, card_height - 1],
        [0, card_height - 1],
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered_box, dst_points)
    warped = cv2.warpPerspective(img, matrix, (card_width, card_height))

    if save_path:
        cv2.imwrite(save_path, warped)

    if debug:
        img_with_box = img.copy()
        cv2.drawContours(img_with_box, [box], 0, (0, 255, 0), 3)
        cv2.imshow("Grayscale", gray)
        cv2.imshow("Blurred", blurred)
        cv2.imshow("Edges", edges)
        cv2.imshow("Edges Dilated", edges_dilated)
        cv2.imshow("Detected Card", img_with_box)
        cv2.imshow("Warped Card", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return warped


# orders box corners clockwise from top-left, then rotates them so the long side is vertical
# (order_points can't tell a sideways or diagonal card's top from its side)
def portrait_corners(box):
    center = box.mean(axis=0)
    angles = np.arctan2(box[:, 1] - center[1], box[:, 0] - center[0])
    pts = box[np.argsort(angles)]
    pts = np.roll(pts, -np.argmin(pts.sum(axis=1)), axis=0)
    top = np.linalg.norm(pts[1] - pts[0])
    left = np.linalg.norm(pts[3] - pts[0])
    if top > left:
        pts = np.roll(pts, -1, axis=0)
    return pts.astype("float32")


# Find every card-shaped box in the photo using 3 edge sensitivities, largest first, duplicates removed
def find_card_boxes(img, ratio_tol=0.10, min_frac=0.10, max_boxes=8):
    img_area = img.shape[0] * img.shape[1]
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)

    boxes = []
    for low, high in [(50, 150), (20, 80), (100, 200)]:
        edges = cv2.dilate(cv2.Canny(gray, low, high), kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            if abs(min(w, h) / max(w, h) - CARD_RATIO) < ratio_tol and w * h > min_frac * img_area:
                boxes.append(rect)

    kept = []
    for rect in sorted(boxes, key=lambda r: -r[1][0] * r[1][1]):
        area = rect[1][0] * rect[1][1]
        duplicate = any(
            np.hypot(rect[0][0] - k[0][0], rect[0][1] - k[0][1]) <= 10
            and abs(area - k[1][0] * k[1][1]) <= 0.05 * k[1][0] * k[1][1]
            for k in kept
        )
        if not duplicate:
            kept.append(rect)
    return kept[:max_boxes]


def crop_candidates(image, card_width=500, card_height=700, max_side=1000):
    """Several possible crops of the card, for the matcher to choose between.
    The first one is always crop_card's result, so it can be kept as the default."""
    img = image
    scale = max_side / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    dst_points = np.array([
        [0, 0],
        [card_width - 1, 0],
        [card_width - 1, card_height - 1],
        [0, card_height - 1],
    ], dtype="float32")

    candidates = []
    try:
        candidates.append(crop_card(img, card_width, card_height, max_side))
    except ValueError:
        pass

    for rect in find_card_boxes(img):
        matrix = cv2.getPerspectiveTransform(portrait_corners(cv2.boxPoints(rect)), dst_points)
        warped = cv2.warpPerspective(img, matrix, (card_width, card_height))
        candidates.append(warped)
        # a box tilted well off-axis could be upside down, so try it flipped too
        tilt = abs(((rect[2] + 45) % 90) - 45)
        if tilt >= 20:
            candidates.append(cv2.rotate(warped, cv2.ROTATE_180))

    # card-shaped windows around the photo center, for when the card's outline isn't found
    # (fingers or a sleeve breaking the edge); users center the card they're scanning
    img_h, img_w = img.shape[:2]
    for frac in (0.5, 0.65, 0.8):
        h = min(img_h * frac, img_w * frac / CARD_RATIO)
        w = h * CARD_RATIO
        x0, y0 = int(img_w / 2 - w / 2), int(img_h / 2 - h / 2)
        window = img[y0:y0 + int(h), x0:x0 + int(w)]
        candidates.append(cv2.resize(window, (card_width, card_height)))

    return candidates


if __name__ == "__main__":
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    image_path = os.path.join(desktop, "test6.webp")

    warped = crop_card(image_path, debug=True, save_path="data/query_cropped.png")

    img = cv2.imread(image_path)
    rows, cols, _ = img.shape
    print(f"Image dimensions: {rows}x{cols}")

import cv2
import numpy as np
import os


# width / height of a Pokemon card (63mm x 88mm)
CARD_RATIO = 63 / 88


# ordered points of raw image corners
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

    # edge detection
    edges = cv2.Canny(blurred, 50, 150)
    # dilate border to form connected outline
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # extract corner contours from the dilated edges
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


if __name__ == "__main__":
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    image_path = os.path.join(desktop, "test6.webp")

    warped = crop_card(image_path, debug=True, save_path="data/query_cropped.png")

    img = cv2.imread(image_path)
    rows, cols, _ = img.shape
    print(f"Image dimensions: {rows}x{cols}")

import cv2
import numpy as np
import os


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


def crop_card(image, card_width=500, card_height=700, debug=False, save_path=None):
    """image: a file path, or a BGR numpy array (e.g. from cv2.imdecode)."""
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image}")
    else:
        img = image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # edge detection
    edges = cv2.Canny(blurred, 50, 150)
    # dilate border to form connected outline
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # extract corner contours from the dilated edges
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in image — check lighting/background")

    # finds the largest contour by area which should correspond to the card borders
    card_contour = max(contours, key=cv2.contourArea)

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

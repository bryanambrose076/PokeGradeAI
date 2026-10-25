"""Download card images into data/images/. Run: python -m scripts.download_images"""
import json
import os
import urllib.request

from PIL import Image, UnidentifiedImageError


def is_valid_png(path):
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except (UnidentifiedImageError, OSError):
        return False

# 1. Get the list of JSON files you already saved in data/sets/
set_files = os.listdir("data/sets")

# 2. Loop over each set's JSON file
for filename in set_files:

    # 3. Open and load that JSON file back into Python
    with open(f"data/sets/{filename}", "r") as f:
        cards = json.load(f)

    # 4. Loop over each card dict in that list
    for card in cards:
        if card["image_url"] is None:
            continue

        # 5. Build the folder path for this card's set, inside data/images/
        set_folder = f"data/images/{card['set_id']}"
        os.makedirs(set_folder, exist_ok=True)

        dest_path = f"{set_folder}/{card['localId']}.png"
        if os.path.exists(dest_path) and is_valid_png(dest_path):
            continue

        try:
            urllib.request.urlretrieve(card["image_url"], dest_path)
            if not is_valid_png(dest_path):
                raise OSError("downloaded file is not a valid image")
        except Exception as e:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            print(f"Failed to download {dest_path}: {e}")

    print(f"Done with {filename}")
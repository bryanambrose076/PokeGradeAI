"""Rebuild data/embeddings.pt from data/images/. Run: python -m scripts.build_embeddings"""
import torch
from PIL import Image
import os

from pokegrade.matcher import load_model, embed_images


# initialize DINOv2 (same model and preprocessing the matcher uses at scan time)
model, processor, device = load_model()

# Get the list of set folders in data/images
set_files = os.listdir("data/images")
embeddings = {}

for filename in set_files:

    if not os.path.isdir(f"data/images/{filename}"):
        continue
    # Loop over each image file in the set folder
    image_files = [f for f in os.listdir(f"data/images/{filename}") if f.lower().endswith(".png")]
    # Embed the set's images in batches of 32, which is much faster than one at a time
    for start in range(0, len(image_files), 32):
        batch = image_files[start:start + 32]
        images = [Image.open(f"data/images/{filename}/{name}") for name in batch]
        features = embed_images(images, model, processor, device)
        # Save each image's features with a key that includes the set and image name
        for name, feature in zip(batch, features):
            embeddings[f"{filename}/{name}"] = feature.cpu()
    print(f"Processed {filename} ({len(image_files)} cards)")

torch.save(embeddings, "data/embeddings.pt")
print("Done")

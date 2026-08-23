import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os



# initialize CLIP and set the device to GPU if available, otherwise CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Get the list of set folders in data/images
set_files = os.listdir("data/images")
embeddings = {}

for filename in set_files:

    if not os.path.isdir(f"data/images/{filename}"):
        continue
    # Loop over each image file in the set folder
    image_files = os.listdir(f"data/images/{filename}")
    for images in image_files:
        if not images.lower().endswith(".png"):
            continue
        # Open the image and process it with CLIP
        image = Image.open(f"data/images/{filename}/{images}")
        inputs = processor(images=image, return_tensors="pt").to(device)
        # Get the image features from the model
        with torch.no_grad():
            image_features = model.get_image_features(**inputs).pooler_output
        # Save the image features in the embeddings dictionary with a key that includes the set and image name
        key = f"{filename}/{images}"
        embeddings[key] = image_features[0].cpu()
        print(f"Processed {key}")
torch.save(embeddings, "data/embeddings.pt")
print("Done")


import sys
import torch
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModel
from PIL import Image


MODEL_NAME = "facebook/dinov2-base"


def load_model(model_name=MODEL_NAME):
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor, device


def embed_images(images, model, processor, device):
    """images: list of PIL.Images or BGR numpy arrays. Returns a [len(images), 768] tensor."""
    pil_images = []
    for image in images:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # squash the whole card to 224x224 instead of the processor's default center crop,
        # which would cut off the card's top (name) and bottom (set number)
        pil_images.append(image.convert("RGB").resize((224, 224), Image.BICUBIC))

    inputs = processor(images=pil_images, return_tensors="pt", do_resize=False, do_center_crop=False).to(device)
    with torch.no_grad():
        features = model(**inputs).pooler_output  # [batch, 768]
    return features


def load_matcher(embeddings_path="data/embeddings.pt", model_name=MODEL_NAME):
    model, processor, device = load_model(model_name)

    embeddings = torch.load(embeddings_path, map_location=device)
    keys = list(embeddings.keys())
    embedding_matrix = torch.stack([embeddings[k] for k in keys]).to(device)  # [num_cards, 768]
    normalized_matrix = torch.nn.functional.normalize(embedding_matrix, p=2, dim=1)

    return {
        "model": model,
        "processor": processor,
        "keys": keys,
        "normalized_matrix": normalized_matrix,
        "device": device,
    }


def match_card(image, matcher, top_k=5):
    """image: a PIL.Image, or a BGR numpy array (e.g. from imageCrop.crop_card)."""
    query_features = embed_images([image], matcher["model"], matcher["processor"], matcher["device"])[0]
    normalized_query = torch.nn.functional.normalize(query_features, p=2, dim=0)

    # cosine similarity = dot product of normalized vectors
    scores = matcher["normalized_matrix"] @ normalized_query
    top_scores, top_indices = torch.topk(scores, k=top_k)

    keys = matcher["keys"]
    return [(keys[i], s) for i, s in zip(top_indices.tolist(), top_scores.tolist())]


if __name__ == "__main__":
    query_path = sys.argv[1] if len(sys.argv) > 1 else "data/query.png"

    matcher = load_matcher()
    query_image = Image.open(query_path)
    results = match_card(query_image, matcher)

    print(f"Top 5 matches for {query_path}:")
    for key, score in results:
        print(f"  {key:<20} {score:.4f}")

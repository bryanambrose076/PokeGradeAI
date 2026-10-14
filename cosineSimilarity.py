import sys
import torch
import numpy as np
import cv2
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


def load_matcher(embeddings_path="data/embeddings.pt", model_name="openai/clip-vit-base-patch32"):
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    embeddings = torch.load(embeddings_path, map_location=device)
    keys = list(embeddings.keys())
    embedding_matrix = torch.stack([embeddings[k] for k in keys]).to(device)  # [num_cards, 512]
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
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    model = matcher["model"]
    device = matcher["device"]

    inputs = matcher["processor"](images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        query_features = model.get_image_features(**inputs).pooler_output[0]  # [512]
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

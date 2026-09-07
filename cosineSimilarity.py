import sys
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

embeddings = torch.load("data/embeddings.pt", map_location=device)


# 3. Turn the dict into:
#    - a list of keys (card identifiers), in a fixed order
#    - a single stacked tensor of shape [num_cards, 512], same order
#    (look up torch.stack for this)
keys = list(embeddings.keys())

embedding_list = []
for k in keys:
    embedding_list.append(embeddings[k])
embedding_matrix = torch.stack(embedding_list).to(device)  # [num_cards, 512]

# 4. Normalize that stacked tensor (unit length per row)
#    (look up torch.nn.functional.normalize, or do it manually with norms)
normalized_matrix = torch.nn.functional.normalize(embedding_matrix, p=2, dim=1)

# 5. Load one real photo you took of a physical card
#    Run it through processor + model.get_image_features(...).pooler_output
#    (same pipeline as embedTest.py — reuse that exact pattern)
#    Normalize this single embedding the same way
if len(sys.argv) > 1:
    query_path = sys.argv[1]
else:
    query_path = "data/query.png"
query_image = Image.open(query_path)
inputs = processor(images=query_image, return_tensors="pt").to(device)
with torch.no_grad():
    query_features = model.get_image_features(**inputs).pooler_output[0]  # [512]
normalized_query = torch.nn.functional.normalize(query_features, p=2, dim=0)

# 6. Compute similarity: normalized_matrix @ normalized_query
#    (matrix-vector multiply — gives you one score per reference card)
scores = normalized_matrix @ normalized_query  # [num_cards]

# 7. Find the top 5 highest scores and their indices
#    (look up torch.topk)
top_scores, top_indices = torch.topk(scores, k=5)

# 8. Print the corresponding keys (using your ordered key list) + their scores
print(f"Top 5 matches for {query_path}:")
top_scores = top_scores.tolist()
top_indices = top_indices.tolist()
for rank in range(len(top_indices)):
    index = top_indices[rank]
    score = top_scores[rank]
    print(f"  {keys[index]:<20} {score:.4f}")
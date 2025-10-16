from transformers import BertForSequenceClassification, BertTokenizer
import torch
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load your fine-tuned model (using BertForSequenceClassification)
model_path = "./models/best_patentbert_cpc_model.pt"  # relative path

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

# Load checkpoint and extract 'model_state_dict'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint["model_state_dict"])

# Path to CPC label dataset
CPC_FILE = "data/cpc_labels.json"

if not os.path.isfile(CPC_FILE):
    raise FileNotFoundError("❌ CPC label file not found. Please provide 'data/cpc_labels.json'.")

# Load CPC data
with open(CPC_FILE, "r") as f:
    cpc_data = json.load(f)

def encode(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model.bert(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

# Precompute CPC embeddings
cpc_embeddings = []
for item in cpc_data:
    emb = encode(item["description"])
    cpc_embeddings.append((item["code"], item["description"], emb))

def classify_cpc(abstract: str):
    abstract_emb = encode(abstract)
    similarities = []
    for code, desc, emb in cpc_embeddings:
        sim = cosine_similarity([abstract_emb], [emb])[0][0]
        similarities.append((code, desc, sim))
    best_match = max(similarities, key=lambda x: x[2])
    return f"[{best_match[0]}] - {best_match[1]}\nReason: Highest semantic similarity score ({best_match[2]:.3f})"

if __name__ == "__main__":
    test_abstract = "A method for processing digital data using electrical circuits"
    print("Classifying test abstract...")
    print(classify_cpc(test_abstract))

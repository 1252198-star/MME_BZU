import json
from sentence_transformers import SentenceTransformer

with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print("Loading embedding model...")

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

texts = [chunk["text"] for chunk in chunks]

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

for i, chunk in enumerate(chunks):
    chunk["embedding"] = embeddings[i].tolist()

with open("embedded_chunks.json", "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False)

print("Embeddings created successfully!")
print("Total embeddings:", len(embeddings))
import json
import chromadb

print("Loading embedded chunks...")

with open("embedded_chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

client = chromadb.PersistentClient(path="./chroma_db")

# حذف البيانات القديمة
try:
    client.delete_collection(name="bzu_documents")
except Exception:
    pass

collection = client.get_or_create_collection(
    name="bzu_documents"
)

print("Storing vectors...")

collection.add(
    ids=[str(chunk["id"]) for chunk in chunks],
    embeddings=[chunk["embedding"] for chunk in chunks],
    documents=[chunk["text"] for chunk in chunks],
    metadatas=[
        {
            "source": chunk["source"],
            "page": chunk["page"]
        }
        for chunk in chunks
    ]
)

print("Database updated successfully!")
print("Stored documents:", collection.count())
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="bzu_documents")

data = collection.get(
    include=["documents", "metadatas"]
)

for text, metadata in zip(data["documents"], data["metadatas"]):
    if "PHED127" in text:
        print("\nPAGE:", metadata["page"])
        print(text[:700])
        print("-" * 50)
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="bzu_documents")

data = collection.get(include=["documents", "metadatas"])

for i, doc in enumerate(data["documents"]):
    if "شروط التخرج التالية" in doc or "منح درجة البكالوريوس" in doc:
        print("\nPAGE:", data["metadatas"][i]["page"])
        print(doc[:2000])
        print("-" * 60)

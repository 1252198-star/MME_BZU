from pathlib import Path
import json

input_file = Path("extracted_text/BZU.json")

pages = json.loads(input_file.read_text(encoding="utf-8"))

chunk_size = 1200
overlap = 200

chunks = []
chunk_id = 1

for page_data in pages:

    text = page_data["text"]
    page_number = page_data["page"]

    start = 0

    while start < len(text):

        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "id": chunk_id,
                "text": chunk,
                "page": page_number,
                "source": "BZU.pdf"
            })

            chunk_id += 1

        start += chunk_size - overlap

output_file = Path("chunks.json")

output_file.write_text(
    json.dumps(chunks, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("Text splitting completed!")
print("Total chunks:", len(chunks))
print("Saved to:", output_file)

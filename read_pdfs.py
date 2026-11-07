
from pathlib import Path
import json
from pypdf import PdfReader

PDF_FOLDER = Path("documents")
OUTPUT_FOLDER = Path("extracted_text")

OUTPUT_FOLDER.mkdir(exist_ok=True)

pdf_files = list(PDF_FOLDER.glob("*.pdf"))

if not pdf_files:
    print("No PDF files found in documents folder.")

for pdf_file in pdf_files:

    reader = PdfReader(str(pdf_file))

    pages_data = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if text.strip():
            pages_data.append({
                "page": page_number,
                "text": text
            })

    output_file = OUTPUT_FOLDER / f"{pdf_file.stem}.json"

    output_file.write_text(
        json.dumps(pages_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"File: {pdf_file.name}")
    print(f"Total pages: {len(reader.pages)}")
    print(f"Extracted pages: {len(pages_data)}")
    print(f"Saved to: {output_file}")
    print("-" * 40)

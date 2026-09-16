import fitz  # pymupdf
from docx import Document
import os


def extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF."""
    full_text = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            full_text.append(page.get_text())
    return "\n".join(full_text)


def extract_docx_text(docx_path: str) -> str:
    """Extract all text from a Word document."""
    doc = Document(docx_path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def extract_txt_text(txt_path: str) -> str:
    """Extract all text from a plain text file, trying multiple encodings."""
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    for enc in encodings:
        try:
            with open(txt_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode text file: {txt_path}")


def extract_text(file_path: str) -> str:
    """Extract text from a document, detecting the type by extension.
    Raises ValueError for unsupported types, missing files, or extraction failures."""
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    try:
        if file_path.lower().endswith(".pdf"):
            return extract_pdf_text(file_path)
        elif file_path.lower().endswith(".docx"):
            return extract_docx_text(file_path)
        elif file_path.lower().endswith(".txt"):
            return extract_txt_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to extract text from {file_path}: {e}")


def list_documents(data_folder: str = "data") -> list[str]:
    """Return the paths of all supported documents in the data/ folder."""
    if not os.path.isdir(data_folder):
        return []
    supported_extensions = (".pdf", ".docx", ".txt")
    return [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if f.lower().endswith(supported_extensions)
    ]


if __name__ == "__main__":
    docs = list_documents("data")
    print(f"Found {len(docs)} document(s): {docs}")

    for path in docs:
        try:
            text = extract_text(path)
            print(f"\n--- {path} ---")
            print(f"Text length: {len(text)} characters")
            print("First 500 characters:\n")
            print(text[:500])
        except ValueError as e:
            print(f"\n--- {path} ---")
            print(f"Error: {e}")
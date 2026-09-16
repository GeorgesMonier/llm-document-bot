def chunk_text(text: str, max_chars: int = 3000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks of roughly max_chars characters.
    Overlap helps avoid cutting sentences/ideas at chunk boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap  # step back a bit to overlap

    return chunks
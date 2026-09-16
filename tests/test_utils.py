import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from text_utils import chunk_text


def test_chunk_text_short_text_returns_single_chunk():
    """Text shorter than max_chars should not be split."""
    text = "Short text."
    chunks = chunk_text(text, max_chars=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long_text_splits_into_multiple_chunks():
    """Text longer than max_chars should be split into multiple chunks."""
    text = "A" * 1000
    chunks = chunk_text(text, max_chars=300, overlap=50)
    assert len(chunks) > 1


def test_chunk_text_covers_full_text():
    """All chunks together should cover the entire original text."""
    text = "word " * 500
    chunks = chunk_text(text, max_chars=300, overlap=50)
    # The last chunk should reach the end of the text
    assert chunks[-1].endswith(text[-10:])
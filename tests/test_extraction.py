import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from extraction import extract_text, list_documents, extract_txt_text


def test_extract_text_unsupported_type(tmp_path):
    """extract_text should raise ValueError for unsupported file types."""
    fake_file = tmp_path / "document.xyz"
    fake_file.write_text("some content")

    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text(str(fake_file))


def test_extract_text_file_not_found():
    """extract_text should raise ValueError if the file doesn't exist."""
    with pytest.raises(ValueError, match="File not found"):
        extract_text("does_not_exist.pdf")


def test_extract_txt_text_reads_content(tmp_path):
    """extract_txt_text should return the exact content of a text file."""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Hello, this is a test document.")

    result = extract_txt_text(str(test_file))
    assert result == "Hello, this is a test document."


def test_list_documents_filters_by_extension(tmp_path):
    """list_documents should only return supported file types."""
    (tmp_path / "doc1.pdf").write_text("fake pdf")
    (tmp_path / "doc2.txt").write_text("fake txt")
    (tmp_path / "ignore.xyz").write_text("should be ignored")

    docs = list_documents(str(tmp_path))
    found_names = [os.path.basename(d) for d in docs]

    assert "doc1.pdf" in found_names
    assert "doc2.txt" in found_names
    assert "ignore.xyz" not in found_names


def test_list_documents_empty_folder_returns_empty_list():
    """list_documents should return an empty list for a nonexistent folder."""
    result = list_documents("nonexistent_folder_xyz")
    assert result == []
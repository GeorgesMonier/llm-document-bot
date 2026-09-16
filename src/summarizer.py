import ollama
from text_utils import chunk_text

MODEL = "llama3.2"
LONG_TEXT_THRESHOLD = 4000  # characters


class OllamaConnectionError(Exception):
    """Raised when Ollama is not running or unreachable."""
    pass


def _generate(prompt: str) -> str:
    """Internal helper: call Ollama and clean up the response."""
    try:
        response = ollama.generate(model=MODEL, prompt=prompt)
    except Exception as e:
        raise OllamaConnectionError(
            f"Could not reach Ollama. Is it running? ('ollama serve'). Details: {e}"
        )

    result = response["response"].strip()
    for marker in ["\nQuestion:", "\nQ:", "\nQUESTION"]:
        if marker in result:
            result = result.split(marker)[0].strip()
    return result


def summarize_text(text: str, max_words: int = 150) -> str:
    """Generate a summary of a single (short) chunk of text."""
    prompt = f"""Summarize the following text in a maximum of {max_words} words.

STRICT RULES:
- Output ONLY the summary, nothing else.
- Do NOT include questions, answers, titles, or any extra text.
- Do NOT explain what you are doing.

TEXT:
{text}

Write the summary now:"""

    return _generate(prompt)


def summarize_long_text(text: str, max_words: int = 150) -> str:
    """Summarize text of any length using map-reduce chunking.

    Short texts: summarized directly.
    Long texts: split into chunks, each chunk summarized individually
    ('map'), then those summaries are combined and summarized again
    ('reduce') to produce one final, coherent summary.
    """
    if len(text) <= LONG_TEXT_THRESHOLD:
        return summarize_text(text, max_words)

    chunks = chunk_text(text, max_chars=3000, overlap=200)
    print(f"    (long document: splitting into {len(chunks)} chunks...)")

    chunk_summaries = [
        summarize_text(chunk, max_words=80) for chunk in chunks
    ]

    combined = "\n\n".join(chunk_summaries)
    return summarize_text(combined, max_words=max_words)
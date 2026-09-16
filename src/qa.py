import ollama
from text_utils import chunk_text

MODEL = "llama3.2"
LONG_TEXT_THRESHOLD = 4000  # characters


class OllamaConnectionError(Exception):
    """Raised when Ollama is not running or unreachable."""
    pass


def _find_relevant_chunk(text: str, question: str) -> str:
    """Naive keyword-based retrieval: pick the chunk with the most
    question-word matches. For a full RAG setup, this would be
    replaced with embeddings + a vector store."""
    chunks = chunk_text(text, max_chars=3000, overlap=200)
    question_words = set(question.lower().split())

    best_chunk = chunks[0]
    best_score = -1
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_words & chunk_words)
        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk


def answer_question(text: str, question: str) -> str:
    """Answer a question based on the given text.
    For long documents, first narrows down to the most relevant chunk."""
    context = text
    if len(text) > LONG_TEXT_THRESHOLD:
        context = _find_relevant_chunk(text, question)

    prompt = f"""Answer the following question based ONLY on the provided text.
If the answer is not in the text, say "I cannot find that information in the document".

TEXT:
{context}

QUESTION: {question}

ANSWER:"""

    try:
        response = ollama.generate(model=MODEL, prompt=prompt)
        return response["response"].strip()
    except Exception as e:
        raise OllamaConnectionError(
            f"Could not reach Ollama. Is it running? ('ollama serve'). Details: {e}"
        )
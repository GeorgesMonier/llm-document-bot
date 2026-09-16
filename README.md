# Document Bot with Apache Spark

A Python tool that extracts text from documents (PDF, Word, TXT), generates summaries, and answers questions about their content — using a local LLM (no paid API required) and Apache Spark for parallel batch processing.

## Features

-  Multi-format extraction: PDF, DOCX, TXT
-  Local LLM summarization and Q&A via [Ollama](https://ollama.com) (llama3.2)
- Parallel document processing with PySpark
- Automatic chunking (map-reduce) for long documents that exceed the model's context
- Robust error handling — a corrupted file won't crash the whole batch
-  Unit tests with pytest

## How it works

1. Spark reads all documents in `data/` and distributes them across workers.
2. Each worker extracts the text and generates a summary (chunking automatically if the document is long).
3. Results are collected and shown to the user.
4. The user picks a document and can ask free-form questions about it in an interactive loop.

## Tech stack

Python · Apache Spark (PySpark) · Ollama (local LLM) · PyMuPDF · python-docx · pytest

## Installation

**Requirements:** Python 3.10+, Java (JDK 17), [Ollama](https://ollama.com/download)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/document-bot-spark.git
cd document-bot-spark

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download the local model
ollama pull llama3.2
```

## Usage

Place your documents (PDF, DOCX, or TXT) in the `data/` folder, then run:

```bash
python main.py
```

The bot will summarize every document in parallel, let you pick one, and open an interactive Q&A session.

## Running tests

```bash
pytest tests/ -v
```

## Known limitations

- Q&A retrieval for long documents uses simple keyword matching rather than semantic search. A natural next step would be embeddings + a vector store (e.g. ChromaDB) for true RAG-based retrieval.
- Small local models (like llama3.2) can occasionally add extra content beyond what's requested — mitigated with strict prompting, but not 100% guaranteed.

   ## Demo

   ![Demo](screenshots/demo.png)
   
## Project structure
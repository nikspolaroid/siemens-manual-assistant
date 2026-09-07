# Siemens Manual Assistant

A retrieval-augmented question answering system over Siemens industrial
automation documentation. Ask a question in plain English; get an answer
grounded in the manuals, with the source file and page number for every claim.

## Why

A maintenance technician troubleshooting a drive fault needs one paragraph
out of a 400-page manual. Keyword search fails when the technician's words
differ from the manual's. Semantic retrieval closes that gap, and mandatory
citations keep the model honest.

## Design decisions

**Embeddings run locally.** Document text never leaves the machine — only
the assembled prompt does. Relevant for plants with data-residency rules.

**Inference is abstracted.** A cloud model is used for iteration speed, but
`src/llm.py` switches to a local Ollama model with one line in `.env`.
Air-gapped OT networks cannot reach a cloud API.

**The model must refuse.** If the retrieved passages do not answer the
question, the system says so and names the manual that likely would. A
maintenance assistant that invents torque values is worse than none.

**Chunk size matches the embedding model.** 900 characters with 150 overlap,
sized to fit the encoder's input window so no text is silently truncated.

## Architecture

    PDFs -> page-tagged chunks -> local embeddings -> ChromaDB
    question -> embed -> top-k retrieval -> prompt with citations -> LLM -> answer

## Setup

    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env        # add your API key

Download manuals into `data/manuals/` (see the README there), then:

    python src/ingest.py    # inspect chunk counts
    python src/index.py     # build the vector index
    python src/ask.py       # command line
    python src/server.py    # browser UI at http://127.0.0.1:8000

## Current limitations

- Fixed-window chunking splits tables across chunk boundaries
- No reranking; retrieval quality is whatever the bi-encoder returns
- Single-turn only, no conversation memory
- Scanned PDFs without a text layer are skipped, not OCR'd

## Stack

Python, pypdf, sentence-transformers (all-MiniLM-L6-v2), ChromaDB,
Anthropic API / Ollama, Streamlit

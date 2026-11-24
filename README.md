# Document Intelligence Assistant

Chat with any PDF. A RAG (Retrieval-Augmented Generation) app: LLMs don't know your
private documents (contracts, reports, manuals), so we retrieve the relevant passages
at query time and ground the LLM's answer in them.

## How it works
1. **Ingest**: extract text per page (pypdf), split into 800-char chunks with 150 overlap.
2. **Embed**: `all-MiniLM-L6-v2` (sentence-transformers) turns chunks into vectors (runs locally).
3. **Index**: FAISS `IndexFlatIP` over normalized vectors (cosine similarity).
4. **Retrieve**: embed the question, fetch top-4 chunks.
5. **Generate**: the LLM answers using only those chunks, citing page numbers.

## LLM layer (provider-agnostic)
- **Gemini** (cloud) is the primary model, with retry + exponential backoff on 503/429.
- **Ollama** (local, e.g. `llama3.2:3b`) is the automatic fallback, and can be used alone
  for fully private, offline operation (documents never leave the machine).
- A sidebar switch selects Auto / Gemini only / Local only. Each answer shows which model produced it.

## Setup
```bash
pip install -r requirements.txt

# Cloud LLM (optional if using local only)
# Windows CMD:  set GEMINI_API_KEY=your_key
# Mac/Linux:    export GEMINI_API_KEY=your_key

# Local LLM (optional if using Gemini only)
# Install Ollama from ollama.com, then:
ollama pull llama3.2:3b

streamlit run app.py
```
Override models with env vars `GEMINI_MODEL` and `OLLAMA_MODEL`.

## Limitations / future work
- No OCR for scanned PDFs (add pytesseract)
- Index is in-memory per session (move to Chroma/pgvector for persistence)
- Add a reranker and hybrid (BM25 + vector) search to improve retrieval
- Add evaluation (RAGAS / hand-labelled Q&A set)
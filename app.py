"""Document Intelligence Assistant - chat with any PDF using RAG.

Pipeline: PDF -> page text -> overlapping chunks -> embeddings -> FAISS index
          question -> embed -> top-k chunks -> LLM answers grounded in them.

LLM layer is provider-agnostic: Gemini (cloud) with automatic fallback to a
local Ollama model if the API is overloaded, rate-limited or unavailable.
"""
import os
import time

import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # overlap so answers aren't cut at chunk borders
TOP_K = 4             # passages retrieved per question
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided document excerpts. "
    "Cite the page like (p. 3). If the answer is not in the excerpts, say "
    "'I couldn't find that in the document.' Do not use outside knowledge."
)


# ---------------- Retrieval ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")  # small, free, runs on CPU


def extract_pages(file):
    reader = PdfReader(file)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_pages(pages):
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for page_no, text in pages:
        text = " ".join(text.split())
        for start in range(0, max(len(text), 1), step):
            piece = text[start:start + CHUNK_SIZE]
            if len(piece) > 50:
                chunks.append({"page": page_no, "text": piece})
    return chunks


def build_index(chunks, embedder):
    vecs = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
    index = faiss.IndexFlatIP(vecs.shape[1])  # inner product == cosine (normalized)
    index.add(np.asarray(vecs, dtype="float32"))
    return index


def retrieve(question, index, chunks, embedder):
    q = embedder.encode([question], normalize_embeddings=True)
    _, ids = index.search(np.asarray(q, dtype="float32"), TOP_K)
    return [chunks[i] for i in ids[0] if i != -1]


# ---------------- LLM providers ----------------
def build_user_prompt(question, passages):
    context = "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in passages)
    return f"Document excerpts:\n{context}\n\nQuestion: {question}"


def ask_gemini(question, passages, history):
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    contents = []
    for m in history[-6:]:  # recent turns so follow-up questions work
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    contents.append(types.Content(
        role="user", parts=[types.Part(text=build_user_prompt(question, passages))]))

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT, max_output_tokens=800, temperature=0.2
    )
    last_err = None
    for attempt in range(3):  # retry with backoff on overload / rate limit
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL, contents=contents, config=config)
            return resp.text
        except Exception as e:
            last_err = e
            if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            break
    raise last_err


def ask_ollama(question, passages, history):
    import ollama

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history[-6:]
        + [{"role": "user", "content": build_user_prompt(question, passages)}]
    )
    resp = ollama.chat(model=OLLAMA_MODEL, messages=messages,
                       options={"temperature": 0.2})
    return resp["message"]["content"]


def answer(question, passages, history, mode):
    """Returns (answer_text, label_of_model_that_answered)."""
    gemini_label = f"Gemini ({GEMINI_MODEL})"
    local_label = f"Local ({OLLAMA_MODEL}, Ollama)"

    if mode == "Local only (Ollama)":
        return ask_ollama(question, passages, history), local_label
    if mode == "Gemini only":
        return ask_gemini(question, passages, history), gemini_label

    # Auto: try cloud first, fall back to local
    try:
        return ask_gemini(question, passages, history), gemini_label
    except Exception as gemini_err:
        try:
            text = ask_ollama(question, passages, history)
            return text, local_label + " - fallback"
        except Exception as ollama_err:
            return (
                "⚠️ Both models failed.\n\n"
                f"- Gemini: {type(gemini_err).__name__}: {str(gemini_err)[:150]}\n"
                f"- Ollama: {type(ollama_err).__name__}: {str(ollama_err)[:150]}\n\n"
                "Check your GEMINI_API_KEY and that Ollama is running.",
                "none",
            )


# ---------------- UI ----------------
st.set_page_config(page_title="Document Intelligence Assistant", page_icon="📄")
st.title("📄 Document Intelligence Assistant")
st.caption("Upload a PDF and chat with it. Answers are grounded in the document (RAG).")

embedder = load_embedder()

with st.sidebar:
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    mode = st.radio(
        "LLM mode",
        ["Auto (Gemini → local fallback)", "Gemini only", "Local only (Ollama)"],
        help="Auto uses Gemini and falls back to a local Ollama model if the API fails.",
    )
    if uploaded and st.session_state.get("doc_name") != uploaded.name:
        with st.spinner("Reading and indexing..."):
            chunks = chunk_pages(extract_pages(uploaded))
            if not chunks:
                st.error("No extractable text (scanned PDF?). Try a text-based PDF.")
                st.stop()
            st.session_state.chunks = chunks
            st.session_state.index = build_index(chunks, embedder)
            st.session_state.doc_name = uploaded.name
            st.session_state.messages = []
        st.success(f"Indexed {len(chunks)} chunks")

for m in st.session_state.get("messages", []):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if "index" in st.session_state:
    if q := st.chat_input("Ask something about your document"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            passages = retrieve(q, st.session_state.index, st.session_state.chunks, embedder)
            with st.spinner("Thinking..."):
                reply, source = answer(q, passages, st.session_state.messages[:-1], mode)
            st.markdown(reply)
            st.caption(f"Answered by: {source}")
            with st.expander("Source passages"):
                for p in passages:
                    st.markdown(f"**Page {p['page']}**: {p['text'][:300]}...")
        st.session_state.messages.append({"role": "assistant", "content": reply})
else:
    st.info("Upload a PDF in the sidebar to get started.")
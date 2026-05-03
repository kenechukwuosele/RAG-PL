# RAG-PL — RAG Research Assistant for LMS & e‑Learning Papers

A **Retrieval‑Augmented Generation (RAG)** research assistant built from scratch in **Python**. It answers questions **grounded in a local corpus of research papers** (PDFs) focused on **Learning Management Systems (LMS)** and **e‑learning**.

This project implements a complete RAG pipeline end‑to‑end:
- **PDF ingestion & chunking**
- **Vector embeddings + persistent indexing (ChromaDB)**
- **Hybrid retrieval** (dense + BM25)
- **Cross‑encoder reranking**
- **LLM generation** (via Groq)
- Optional **LLM‑based evaluation** (faithfulness / relevance / completeness)
- **Response caching** for repeated questions

---

## Why this repo exists

Most RAG tutorials skip important pieces (chunking, reranking, evaluation, caching, persistence) or hide them behind frameworks. This repo keeps the components **small, readable, and hackable**, making it a good base for:
- experimenting with retrieval strategies
- swapping embedding / reranking / generation models
- running offline RAG against your own PDF corpus

---

## Architecture at a glance

**High‑level flow**

1. **Ingest PDFs** from `papers/`
2. **Extract text** and **chunk** into passages
3. **Embed** chunks and **persist** them to ChromaDB (`db/`)
4. For a user question:
   - optionally **expand** the question into multiple search queries
   - retrieve candidates via **dense similarity** + **BM25 keyword scoring**
   - **rerank** results with a cross‑encoder
5. **Generate** an answer using the top‑ranked context
6. Optionally **evaluate** answer quality
7. **Cache** the response in `cache.json`

---

## Repository map

| Path | Purpose |
| --- | --- |
| `main.py` | Application entry point: orchestrates expansion → retrieval → rerank → generation → (optional) evaluation → caching |
| `parser.py` | PDF ingestion utilities: extract text, chunk passages for indexing |
| `embedder.py` | Embedding helper: converts chunks/queries into vectors |
| `retriever.py` | Hybrid retrieval layer: Chroma dense search + BM25 keyword scoring |
| `reranker.py` | Cross‑encoder reranker: reorders/filters retrieved candidates |
| `generator.py` | Groq‑backed query expansion + answer generation |
| `evaluator.py` | LLM‑based evaluation of faithfulness, relevance, completeness |
| `cache.py` | JSON‑backed cache utilities |
| `papers/` | Your source PDF corpus (LMS / e‑learning papers) |
| `db/` | Local ChromaDB persistence directory (runtime artifact) |
| `cache.json` | Persisted cache data (runtime artifact) |

---

## Quickstart

### 1) Prerequisites
- Python 3.10+ recommended
- A **Groq API key**

Set your Groq key in the environment:

```bash
export GROQ="<your_api_key>"
```

### 2) Add papers
Place PDFs into:

```text
papers/
```

### 3) Run
Start the pipeline via the main entry point:

```bash
python main.py
```

> The first run will typically take longer because it needs to parse PDFs, chunk them, embed them, and persist the index to `db/`.

---

## Configuration notes

- **Environment variables**
  - `GROQ` — required for query expansion and answer generation.

- **Runtime artifacts**
  - `db/` and `cache.json` are generated during use. If you want a clean re‑index, delete `db/` (and optionally `cache.json`) and rerun.

---

## What “grounded answers” means here

The generator is intended to answer using the **retrieved paper chunks** as context. The evaluator (if enabled) can help judge whether responses:
- stay faithful to the provided context
- actually address the question
- include sufficient detail

---

## Ideas for extending this project

- Add citation rendering (paper name + page) for each supporting chunk
- Expose as an API (FastAPI) or simple web UI
- Swap ChromaDB for FAISS / SQLite‑VSS / Elastic
- Add metadata filters (year, author, venue, topic)
- Add dataset‑based evaluation (ground truth Q/A)

---

## License

No license file was found in this repository. If you intend others to use or contribute, consider adding one (MIT/Apache‑2.0/etc.).

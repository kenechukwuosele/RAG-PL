# RAG-PL

This repository is a small retrieval-augmented generation pipeline built around a local paper corpus.

## File Map

- [main.py](main.py): application entry point that expands a question, retrieves supporting chunks, reranks them, generates an answer, and optionally evaluates the result.
- [parser.py](parser.py): PDF ingestion utilities for extracting text from papers and chunking it for indexing.
- [embedder.py](embedder.py): sentence embedding helper used to convert chunks and queries into vectors.
- [retriever.py](retriever.py): hybrid retrieval layer that combines Chroma dense search with BM25 keyword scoring.
- [reranker.py](reranker.py): cross-encoder reranker that filters the retrieved candidates down to the most relevant chunks.
- [generator.py](generator.py): Groq-backed query expansion and answer generation.
- [evaluator.py](evaluator.py): LLM-based evaluation of faithfulness, relevance, and completeness.
- [cache.py](cache.py): JSON-backed cache for previously answered questions.
- [cache.json](cache.json): persisted cache data for repeated questions.
- [db/](db/): local ChromaDB persistence directory for the indexed paper embeddings.
- [papers/](papers/): source PDF corpus used to build the retrieval index.

## Runtime Flow

1. Load and chunk PDFs from `papers/`.
2. Embed chunks and store them in ChromaDB under `db/`.
3. Expand the user question into a few search variants.
4. Retrieve candidate chunks with dense search plus BM25 reranking.
5. Rerank again with a cross-encoder.
6. Generate a grounded answer from the final context.
7. Cache the answer in `cache.json`.

## Notes

- The project expects the Groq API key in the `GROQ` environment variable.
- The existing `db/` and `cache.json` files are runtime artifacts, not source code.
"""Entry point for the RAG demo.

This module wires together query expansion, retrieval, reranking, answer
generation, caching, and optional evaluation for a few sample questions.
"""

from retriever import retrieve
from generator import generate_answer, expand_query
from reranker import rerank
from evaluator import evaluate
from cache import get_cached, set_cache

def ask(question, run_eval=False):
    print(f"\nQuestion: {question}")

    # Check cache first
    cached = get_cached(question)
    if cached:
        print(f"\n[CACHED] Answer: {cached['answer']}")
        print(f"Sources: {', '.join(cached['sources'])}")
        return

    queries = expand_query(question)

    all_docs, all_metas = [], []
    seen = set()
    for q in queries:
        docs, metas = retrieve(q, top_k=5)
        for doc, meta in zip(docs, metas):
            if doc not in seen:
                seen.add(doc)
                all_docs.append(doc)
                all_metas.append(meta)

    docs, metas = rerank(question, all_docs, all_metas, top_k=5, threshold=0.0)

    if not docs:
        print("\nAnswer: I don't have enough information in my knowledge base.")
        return

    answer = generate_answer(question, docs)
    sources = list(set(m["source"] for m in metas))

    print(f"\nAnswer: {answer}")
    print(f"\nSources: {', '.join(sources)}")

    # Store in cache
    set_cache(question, answer, sources)

    if run_eval:
        print("\n--- Evaluation ---")
        scores = evaluate(question, docs, answer)
        print(scores)

if __name__ == "__main__":
    ask("How does gamification improve student engagement in LMS?")
    ask("What role does AI play in automated grading?")
    ask("How does real-time feedback affect learning outcomes?")
    ask("What is the capital of France?")  # should fail gracefully
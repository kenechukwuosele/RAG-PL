from retriever import retrieve
from generator import generate_answer, expand_query
from reranker import rerank
from evaluator import evaluate
from cache import get_cached, set_cache

def ask(question, run_eval=False):
    cached = get_cached(question)
    if cached:
        return {
            "question": question,
            "answer": cached["answer"],
            "sources": cached["sources"],
            "cached": True,
            "evaluation": None
        }

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
        return {
            "question": question,
            "answer": "I don't have enough information in my knowledge base.",
            "sources": [],
            "cached": False,
            "evaluation": None
        }

    answer = generate_answer(question, docs)
    sources = list(set(m["source"] for m in metas))
    set_cache(question, answer, sources)

    eval_result = None
    if run_eval:
        eval_result = evaluate(question, docs, answer)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "cached": False,
        "evaluation": eval_result
    }

if __name__ == "__main__":
    result = ask("What are the challenges of e-learning?")
    print(f"\nAnswer: {result['answer']}")
    print(f"Sources: {', '.join(result['sources'])}")